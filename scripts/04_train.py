#!/usr/bin/env python3
"""
Pipeline BumbaLM - Etapa 4: Treinamento Contrastivo do Modelo.

Executa:
1. Leitura dos hiperparametros de treinamento e hardware ('config_train.yaml').
2. Construcao da arquitetura do Qwen3-Embedding-4B (Full Fine-Tuning sem LoRA).
3. Carregamento dos tripletos minerados ('train_triplets.jsonl').
4. Configuracao da CachedMultipleNegativesRankingLoss e do SentenceTransformerTrainer.
5. Execucao do fine-tuning com telemetria cientifica de hardware em tempo real.
6. Salvamento do melhor modelo consolidado em disco ('best_model_4b/').

Uso:
    python scripts/04_train.py
    python scripts/04_train.py --batch_size 128 --epochs 1
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path
from typing import Any

# Adiciona a raiz do repositorio ao PYTHONPATH
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import torch

from src.training.model_builder import construir_modelo_embedding
from src.training.trainer import carregar_dataset_tripletos, configurar_treinador


def carregar_yaml(caminho: Path) -> dict[str, Any]:
    """Le arquivo YAML com seguranca."""
    if not caminho.is_file():
        raise FileNotFoundError(f"Arquivo de configuracao nao encontrado: {caminho}")
    try:
        import yaml
        with caminho.open(encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        return {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config_base",
        type=Path,
        default=BASE_DIR / "configs" / "config_base.yaml",
        help="Caminho do config_base.yaml",
    )
    parser.add_argument(
        "--config_train",
        type=Path,
        default=BASE_DIR / "configs" / "config_train.yaml",
        help="Caminho do config_train.yaml",
    )
    parser.add_argument(
        "--triplets_file",
        type=Path,
        default=None,
        help="Caminho do arquivo train_triplets.jsonl gerado na Etapa 3.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Diretorio de saida do experimento e modelo treinado.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Sobrescreve o batch_size_train configurado no YAML.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Sobrescreve a quantidade de epocas configurada no YAML.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Sobrescreve a taxa de aprendizado (learning_rate).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg_base = carregar_yaml(args.config_base)
    cfg_train = carregar_yaml(args.config_train)

    seed = cfg_base.get("projeto", {}).get("seed", 42)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Localizacao do arquivo de dados de tripletos
    dir_processados = Path(cfg_base.get("dados", {}).get("diretorio_processados", "data/processed"))
    if not dir_processados.is_absolute():
        dir_processados = BASE_DIR / dir_processados

    caminho_tripletos = args.triplets_file or (dir_processados / "train_triplets.jsonl")
    if not Path(caminho_tripletos).is_absolute():
        caminho_tripletos = BASE_DIR / caminho_tripletos

    # Diretorio de saida do experimento
    exp_dir_base = Path(cfg_base.get("saidas", {}).get("diretorio_experimentos", "experiments"))
    exp_nome = cfg_base.get("saidas", {}).get("nome_experimento", "exp_001_h200_4b")
    dir_saida = args.output_dir or (exp_dir_base / exp_nome)
    if not Path(dir_saida).is_absolute():
        dir_saida = BASE_DIR / dir_saida
    dir_saida.mkdir(parents=True, exist_ok=True)

    # Parametros do modelo
    modelo_cfg = cfg_train.get("modelo", {})
    model_id = modelo_cfg.get("model_id", "Qwen/Qwen3-Embedding-4B")
    max_seq_len = modelo_cfg.get("max_seq_length", 768)
    query_instruction = modelo_cfg.get(
        "query_instruction",
        "Instruct: Dado um trecho de um documento jurídico brasileiro, recupere o trecho mais relevante semanticamente.\nQuery: ",
    )
    attn_impl = modelo_cfg.get("attn_implementation", "sdpa")

    # Parametros de treino
    train_cfg = cfg_train.get("treinamento", {})
    batch_size = args.batch_size or train_cfg.get("batch_size_train", 96)
    mini_batch_size = train_cfg.get("mini_batch_size", batch_size)
    grad_accum = train_cfg.get("gradient_accumulation_steps", 1)
    lr = args.lr or train_cfg.get("learning_rate", 2.0e-5)
    weight_decay = train_cfg.get("weight_decay", 0.01)
    epochs = args.epochs or train_cfg.get("num_epochs", 1)
    warmup_ratio = train_cfg.get("warmup_ratio", 0.05)
    bf16 = train_cfg.get("bf16", True)
    grad_checkpointing = train_cfg.get("gradient_checkpointing", False)
    num_workers = train_cfg.get("dataloader_num_workers", 8)
    pin_memory = train_cfg.get("dataloader_pin_memory", True)
    eval_steps = train_cfg.get("eval_steps", 250)
    save_steps = train_cfg.get("save_steps", 250)
    loss_scale = cfg_train.get("loss", {}).get("scale", 50.0)

    print("=" * 64)
    print("BumbaLM - Etapa 4: Treinamento Contrastivo em GPU H200")
    print("=" * 64)
    print(f"Arquivo de Tripletos : {caminho_tripletos}")
    print(f"Diretorio de Saida   : {dir_saida}")
    print(f"Modelo de Base       : {model_id}")
    print(f"Batch Size Real      : {batch_size} (Acumulacao: {grad_accum})")
    print(f"Taxa de Aprendizado  : {lr} | Epocas: {epochs}")
    print(f"Precisao bfloat16    : {bf16} | Atencao: {attn_impl}")
    print(f"Dataloader Workers   : {num_workers} | Pin Memory: {pin_memory}")
    print("-" * 64)

    # 1. Construcao do Modelo de Embedding
    model = construir_modelo_embedding(
        model_id=model_id,
        max_seq_length=max_seq_len,
        attn_implementation=attn_impl,
        gradient_checkpointing=grad_checkpointing,
    )

    # 2. Carregamento do Dataset de Tripletos
    train_dataset = carregar_dataset_tripletos(
        caminho_arquivo=caminho_tripletos,
        n_negativos=5,
        query_instruction=query_instruction,
    )

    # 3. Configuracao do Trainer e Telemetria
    trainer, telemetry = configurar_treinador(
        model=model,
        train_dataset=train_dataset,
        output_dir=dir_saida,
        batch_size=batch_size,
        mini_batch_size=mini_batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        weight_decay=weight_decay,
        num_epochs=epochs,
        warmup_ratio=warmup_ratio,
        eval_steps=eval_steps,
        save_steps=save_steps,
        bf16=bf16,
        dataloader_num_workers=num_workers,
        dataloader_pin_memory=pin_memory,
        scale=loss_scale,
        evaluator=None,
    )

    # 4. Execucao do Fine-Tuning
    print("-" * 64)
    print("Disparando treinamento contrastivo com telemetria cientifica...")
    trainer.train()

    # 5. Salvamento do Modelo Final
    best_model_path = dir_saida / "best_model_4b"
    model.save(str(best_model_path))

    print("=" * 64)
    print(f"Treinamento concluido com sucesso!")
    print(f"Melhor modelo persistido em: {best_model_path}")
    print(f"Relatorio executivo JSON   : {telemetry.json_path}")
    print(f"Metricas tabulares CSV     : {telemetry.csv_path}")
    print("=" * 64)


if __name__ == "__main__":
    main()
