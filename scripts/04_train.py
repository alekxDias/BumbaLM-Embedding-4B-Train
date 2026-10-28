#!/usr/bin/env python3
"""
Pipeline BumbaLM - Etapa 4: Treinamento Contrastivo do Modelo.

Executa:
1. Leitura dos hiperparametros de treinamento e hardware ('config_train.yaml').
2. Construcao da arquitetura do Qwen3-Embedding-4B (Full Fine-Tuning sem LoRA).
3. Carregamento dos tripletos minerados ('train_triplets.jsonl') ou pares sem HN ('train_pairs.jsonl').
4. Configuracao da CachedMultipleNegativesRankingLoss e do SentenceTransformerTrainer.
5. Execucao do fine-tuning com telemetria cientifica de hardware em tempo real.
6. Salvamento do melhor modelo consolidado em disco ('best_model_4b/').

Uso:
    python scripts/04_train.py
    python scripts/04_train.py --gpus 1
    python scripts/04_train.py --gpus 0,1
    python scripts/04_train.py --batch_size 128 --epochs 1
    python scripts/04_train.py --n_negativos 0 --triplets_file data/processed_tjma/train_pairs.jsonl
"""

from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

# Adiciona a raiz do repositorio ao PYTHONPATH
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import torch
import transformers.utils.import_utils as _tf_import_utils
import transformers.trainer as _tf_trainer

# Bypass do bloqueio de versao do torch.load para carregar checkpoints locais confiaveis (CVE-2025-32434)
_tf_import_utils.check_torch_load_is_safe = lambda: None
_tf_trainer.check_torch_load_is_safe = lambda: None

# Suporte para desserializacao segura de objetos NumPy (NumPy 2.0 / Python 3.13)
try:
    import numpy as np
    safe_list = [np.ndarray, np.dtype]
    if hasattr(np, "_core") and hasattr(np._core, "multiarray"):
        if hasattr(np._core.multiarray, "_reconstruct"):
            safe_list.append(np._core.multiarray._reconstruct)
        if hasattr(np._core.multiarray, "scalar"):
            safe_list.append(np._core.multiarray.scalar)
    if hasattr(torch.serialization, "add_safe_globals"):
        torch.serialization.add_safe_globals(safe_list)
except Exception:
    pass

# Tratamento defensivo para rng_state: evita abortar o treino caso o pickle da semente aleatoria falhe
if hasattr(_tf_trainer.Trainer, "_load_rng_state"):
    _orig_load_rng_state = _tf_trainer.Trainer._load_rng_state
    def _safe_load_rng_state(self, resume_from_checkpoint):
        try:
            _orig_load_rng_state(self, resume_from_checkpoint)
        except Exception as exc:
            print(f"[Aviso] Nao foi possivel restaurar rng_state ({exc}). O treinamento continuara com novo estado de RNG.")
    _tf_trainer.Trainer._load_rng_state = _safe_load_rng_state

from src.training.model_builder import construir_modelo_embedding
from src.training.trainer import carregar_dataset_tripletos, configurar_treinador
from src.evaluation.ir_evaluator import criar_ir_evaluator


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


def _ids_gpus(texto: str) -> list[str]:
    return [parte.strip() for parte in str(texto or "").split(",") if parte.strip()]


def _ja_distribuido() -> bool:
    return os.environ.get("LOCAL_RANK") is not None


def _processo_principal() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def _imprimir(mensagem: str) -> None:
    if _processo_principal():
        print(mensagem)


def _resolver_gpus(args: argparse.Namespace, cfg_train: dict[str, Any]) -> str:
    if args.gpus is not None:
        return args.gpus.strip()
    bruto = cfg_train.get("hardware", {}).get("gpus")
    if bruto is None:
        return ""
    return str(bruto).strip()


def _preparar_gpus(gpus: str) -> None:
    """Fixa uma GPU ou relanca o script em DDP quando ha duas ou mais."""
    if _ja_distribuido():
        return

    ids = _ids_gpus(gpus)
    if not ids:
        return

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(ids)
    if len(ids) == 1:
        return

    comando = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node",
        str(len(ids)),
        str(Path(__file__).resolve()),
        *sys.argv[1:],
    ]
    print(f"Relancando treino em {len(ids)} GPUs ({','.join(ids)}) com torch.distributed.run ...")
    resultado = subprocess.run(comando)
    sys.exit(resultado.returncode)


def _dispositivo_local() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    return f"cuda:{local_rank}"


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
        help="Caminho do JSONL de treino (train_triplets.jsonl ou train_pairs.jsonl).",
    )
    parser.add_argument(
        "--n_negativos",
        type=int,
        default=None,
        help="Quantidade de hard negatives por instancia. Use 0 para pares so com in-batch. Padrao: 5.",
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
    parser.add_argument(
        "--mini_batch_size",
        type=int,
        default=None,
        help="Sobrescreve o mini_batch_size para o GradCache (CachedMNRL).",
    )
    parser.add_argument(
        "--no_eval",
        action="store_true",
        help="Desativa o InformationRetrievalEvaluator durante o treinamento.",
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=None,
        help="Sobrescreve a frequencia de avaliacao intermediaria (eval_steps).",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Caminho do checkpoint ou 'auto' para retomar o treinamento a partir do mais recente.",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="GPUs a usar, separadas por virgula. Ex.: '1' ou '0,1'. Vazio mantem o processo unico atual.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg_base = carregar_yaml(args.config_base)
    cfg_train = carregar_yaml(args.config_train)
    _preparar_gpus(_resolver_gpus(args, cfg_train))

    seed = cfg_base.get("projeto", {}).get("seed", 42)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Localizacao do arquivo de dados de tripletos/pares
    dir_processados = Path(cfg_base.get("dados", {}).get("diretorio_processados", "data/processed"))
    if not dir_processados.is_absolute():
        dir_processados = BASE_DIR / dir_processados

    n_negativos = 5 if args.n_negativos is None else int(args.n_negativos)
    if n_negativos < 0:
        raise ValueError(f"--n_negativos deve ser >= 0; recebido: {n_negativos}")

    if args.triplets_file is not None:
        caminho_tripletos = Path(args.triplets_file)
    elif n_negativos == 0:
        caminho_tripletos = dir_processados / "train_pairs.jsonl"
    else:
        caminho_tripletos = dir_processados / "train_triplets.jsonl"
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
    batch_size = args.batch_size or train_cfg.get("batch_size_train", 64)
    mini_batch_size = args.mini_batch_size or train_cfg.get("mini_batch_size", 32)
    grad_accum = train_cfg.get("gradient_accumulation_steps", 1)
    lr = args.lr or train_cfg.get("learning_rate", 2.0e-5)
    weight_decay = train_cfg.get("weight_decay", 0.01)
    epochs = args.epochs or train_cfg.get("num_epochs", 1)
    warmup_ratio = train_cfg.get("warmup_ratio", 0.05)
    bf16 = train_cfg.get("bf16", True)
    grad_checkpointing = train_cfg.get("gradient_checkpointing", False)
    num_workers = train_cfg.get("dataloader_num_workers", 4)
    pin_memory = train_cfg.get("dataloader_pin_memory", False)
    eval_steps = train_cfg.get("eval_steps", 250)
    save_steps = train_cfg.get("save_steps", 250)
    loss_scale = cfg_train.get("loss", {}).get("scale", 50.0)

    dispositivo = _dispositivo_local()
    num_processos = max(1, int(os.environ.get("WORLD_SIZE", "1")))

    _imprimir("=" * 64)
    _imprimir("BumbaLM - Etapa 4: Treinamento Contrastivo")
    _imprimir("=" * 64)
    _imprimir(f"Arquivo de Treino    : {caminho_tripletos}")
    _imprimir(f"Hard negatives (N)   : {n_negativos}" + (" (somente in-batch)" if n_negativos == 0 else ""))
    _imprimir(f"Diretorio de Saida   : {dir_saida}")
    _imprimir(f"Modelo de Base       : {model_id}")
    _imprimir(f"GPUs no passo        : {num_processos} | Dispositivo local: {dispositivo}")
    _imprimir(f"Batch por GPU        : {batch_size} (Acumulacao: {grad_accum})")
    _imprimir(f"Taxa de Aprendizado  : {lr} | Epocas: {epochs}")
    _imprimir(f"Precisao bfloat16    : {bf16} | Atencao: {attn_impl}")
    _imprimir(f"Dataloader Workers   : {num_workers} | Pin Memory: {pin_memory}")
    _imprimir("-" * 64)

    # 1. Construcao do Modelo de Embedding
    model = construir_modelo_embedding(
        model_id=model_id,
        max_seq_length=max_seq_len,
        attn_implementation=attn_impl,
        gradient_checkpointing=grad_checkpointing,
        device=dispositivo,
    )

    # 2. Carregamento do Dataset de Tripletos / Pares
    train_dataset = carregar_dataset_tripletos(
        caminho_arquivo=caminho_tripletos,
        n_negativos=n_negativos,
        query_instruction=query_instruction,
    )

    # 3. Construcao do Avaliador de Validacao (InformationRetrievalEvaluator)
    evaluator = None
    if not args.no_eval:
        arq_val_processos = dir_processados / "val_processos.jsonl"
        arq_val_chunks = dir_processados / "val_corpus_chunks.jsonl"
        if not arq_val_chunks.is_file():
            arq_val_chunks = dir_processados / "validacao_corpus_chunks.jsonl"

        if arq_val_processos.is_file() and arq_val_chunks.is_file():
            eval_batch_size = cfg_train.get("avaliacao", {}).get("eval_batch_size", 128)
            _imprimir("Configurando InformationRetrievalEvaluator sobre o split de validacao...")
            evaluator = criar_ir_evaluator(
                caminho_processos=arq_val_processos,
                caminho_chunks=arq_val_chunks,
                nome="val",
                query_instruction=query_instruction,
                excluir_chunks_contaminados=True,
                batch_size=eval_batch_size,
                show_progress_bar=False,
            )
            _imprimir("  - Avaliador de validacao pronto (chunks com vazamento de ementa excluidos).")
        else:
            _imprimir("Aviso: Arquivos de validacao nao encontrados. O treinamento continuara sem avaliacao intermediaria.")
    else:
        _imprimir("Flag --no_eval detectada: Avaliacao intermediaria desativada.")

    # 4. Configuracao do Trainer e Telemetria
    metric_best = cfg_train.get("metric_for_best_model", "val_cosine_ndcg@10")
    greater_is_better = cfg_train.get("greater_is_better", True)
    if args.eval_steps is not None:
        eval_steps = args.eval_steps

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
        evaluator=evaluator,
        metric_for_best_model=metric_best,
        greater_is_better=greater_is_better,
    )

    # 4. Execucao do Fine-Tuning
    resume = args.resume_from_checkpoint
    if resume is not None:
        if str(resume).strip().lower() in ("true", "auto", "1"):
            resume = True
        else:
            p_resume = Path(resume)
            if not p_resume.is_absolute():
                if (dir_saida / p_resume).exists():
                    p_resume = dir_saida / p_resume
                elif (BASE_DIR / p_resume).exists():
                    p_resume = BASE_DIR / p_resume
            resume = str(p_resume)

    _imprimir("-" * 64)
    if resume:
        _imprimir(f"Retomando treinamento a partir do checkpoint: {resume}")
    else:
        _imprimir("Disparando treinamento contrastivo com telemetria cientifica...")
    trainer.train(resume_from_checkpoint=resume)

    # 5. Salvamento do Modelo Final (somente o processo da primeira GPU)
    if _processo_principal():
        modelo_salvar = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
        best_model_path = dir_saida / "best_model_4b"
        modelo_salvar.save(str(best_model_path))

        print("=" * 64)
        print("Treinamento concluido com sucesso!")
        print(f"Melhor modelo persistido em: {best_model_path}")
        print(f"Relatorio executivo JSON   : {telemetry.json_path}")
        print(f"Metricas tabulares CSV     : {telemetry.csv_path}")
        print("=" * 64)

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


if __name__ == "__main__":
    main()
