#!/usr/bin/env python3
"""
Pipeline BumbaLM - Etapa 5: Avaliacao e Comparativo com Baseline.

Executa:
1. Leitura das configuracoes do projeto e splits de avaliacao ('test_processos.jsonl' e 'test_corpus_chunks.jsonl').
2. Carregamento do modelo treinado ('best_model_4b/').
3. Carregamento do modelo base para comparativo.
4. Execucao da avaliacao de Information Retrieval (IR).
5. Emissao de relatorios consolidados (JSON e TXT) com metricas nDCG@10, MRR@10, Acc@1, 5, 10 e MAP@100.

Uso:
    python scripts/05_evaluate.py
    python scripts/05_evaluate.py --compare_baseline
    python scripts/05_evaluate.py --split val
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Optional

# Adiciona a raiz do repositorio ao PYTHONPATH
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from sentence_transformers import SentenceTransformer

from src.evaluation.ir_evaluator import (
    criar_ir_evaluator,
    executar_avaliacao_completa,
    formatar_metricas_ir,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("bumbalm_evaluate")


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
        "--model_path",
        type=Path,
        default=None,
        help="Caminho do modelo treinado. Padrao: experiments/<nome_experimento>/best_model_4b",
    )
    parser.add_argument(
        "--compare_baseline",
        action="store_true",
        help="Se informado, avalia tambem o modelo base zero-shot para medir o ganho do fine-tuning.",
    )
    parser.add_argument(
        "--baseline_id",
        type=str,
        default="Qwen/Qwen3-Embedding-4B",
        help="Identificador HuggingFace do modelo base zero-shot.",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["test", "val", "both"],
        default="test",
        help="Particao a ser avaliada: test, val ou both. Padrao: test.",
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=None,
        help="Diretorio com os dados processados. Padrao: data/processed.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Diretorio para gravacao dos relatorios de avaliacao.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size para inferencia durante a avaliacao.",
    )
    return parser.parse_args()


def gerar_relatorio_comparativo(
    metricas_treinado: dict[str, float],
    metricas_baseline: Optional[dict[str, float]],
    nome_split: str,
    caminho_saida: Path,
) -> None:
    """Gera relatorio tabular em texto comparando modelo treinado contra baseline."""
    with caminho_saida.open("w", encoding="utf-8") as f:
        f.write("=" * 72 + "\n")
        f.write(f"RELATORIO COMPARATIVO DE RECUPERACAO JURIDICA (SPLIT: {nome_split.upper()})\n")
        f.write("=" * 72 + "\n\n")

        if metricas_baseline:
            cabecalho = f"{'Metrica':<20} | {'Baseline (Zero-Shot)':<22} | {'BumbaLM (Trained)':<20} | {'Delta Abs':<10}"
            f.write(cabecalho + "\n")
            f.write("-" * len(cabecalho) + "\n")

            todas_metricas = list(dict.fromkeys(list(metricas_treinado.keys()) + list(metricas_baseline.keys())))
            for met in todas_metricas:
                val_base = metricas_baseline.get(met, 0.0)
                val_ft = metricas_treinado.get(met, 0.0)
                delta = val_ft - val_base
                sinal = "+" if delta >= 0 else ""
                f.write(f"{met:<20} | {val_base:<22.4f} | {val_ft:<20.4f} | {sinal}{delta:<9.4f}\n")
        else:
            cabecalho = f"{'Metrica':<25} | {'BumbaLM (Trained)':<20}"
            f.write(cabecalho + "\n")
            f.write("-" * len(cabecalho) + "\n")
            for met, val_ft in metricas_treinado.items():
                f.write(f"{met:<25} | {val_ft:<20.4f}\n")

        f.write("\n" + "=" * 72 + "\n")


def main() -> None:
    args = parse_args()

    cfg_base = carregar_yaml(args.config_base)
    cfg_train = carregar_yaml(args.config_train)

    # Diretorio de dados processados
    dir_processados = args.data_dir or Path(cfg_base.get("dados", {}).get("diretorio_processados", "data/processed"))
    if not dir_processados.is_absolute():
        dir_processados = BASE_DIR / dir_processados

    # Identificacao do modelo treinado
    exp_dir_base = Path(cfg_base.get("saidas", {}).get("diretorio_experimentos", "experiments"))
    exp_nome = cfg_base.get("saidas", {}).get("nome_experimento", "exp_001_h200_4b")
    caminho_modelo = args.model_path or (exp_dir_base / exp_nome / "best_model_4b")
    if not Path(caminho_modelo).is_absolute():
        caminho_modelo = BASE_DIR / caminho_modelo

    if not caminho_modelo.exists():
        raise FileNotFoundError(
            f"Diretorio do modelo treinado nao encontrado: {caminho_modelo}\n"
            "Certifique-se de executar 'python scripts/04_train.py' primeiro ou passe --model_path."
        )

    # Diretorio de saida da avaliacao
    dir_saida = args.output_dir or (caminho_modelo.parent / "avaliacoes")
    if not Path(dir_saida).is_absolute():
        dir_saida = BASE_DIR / dir_saida
    dir_saida.mkdir(parents=True, exist_ok=True)

    # Parametros de consulta
    query_instruction = cfg_train.get("modelo", {}).get(
        "query_instruction",
        "Instruct: Dado um trecho de um documento jurídico brasileiro, recupere o trecho mais relevante semanticamente.\nQuery: ",
    )

    splits_para_avaliar = ["test"] if args.split == "test" else (["val"] if args.split == "val" else ["val", "test"])

    print("=" * 64)
    print("BumbaLM - Etapa 5: Avaliacao In-Domain de Recuperacao Juridica")
    print("=" * 64)
    print(f"Modelo sob Avaliacao : {caminho_modelo}")
    print(f"Splits Alvo          : {', '.join(splits_para_avaliar)}")
    print(f"Diretorio de Dados   : {dir_processados}")
    print(f"Diretorio de Saida   : {dir_saida}")
    print(f"Comparar Baseline    : {args.compare_baseline} ({args.baseline_id if args.compare_baseline else 'N/A'})")
    print(f"Batch Size Avaliacao : {args.batch_size}")
    print("-" * 64)

    # 1. Carregamento do modelo treinado
    print(f"\nCarregando modelo treinado: {caminho_modelo} ...")
    modelo_treinado = SentenceTransformer(str(caminho_modelo), trust_remote_code=True)

    # 2. Carregamento opcional do baseline zero-shot
    modelo_baseline: Optional[SentenceTransformer] = None
    if args.compare_baseline:
        print(f"Carregando modelo baseline zero-shot: {args.baseline_id} ...")
        modelo_baseline = SentenceTransformer(args.baseline_id, trust_remote_code=True)

    relatorio_consolidado_geral: dict[str, Any] = {}

    for split_nome in splits_para_avaliar:
        arq_processos = dir_processados / f"{split_nome}_processos.jsonl"
        arq_chunks = dir_processados / f"{split_nome}_corpus_chunks.jsonl"

        if not arq_processos.is_file() or not arq_chunks.is_file():
            log.warning("Arquivos para o split '%s' nao encontrados em %s. Pulando...", split_nome, dir_processados)
            continue

        print(f"\n--- Avaliando Split: {split_nome.upper()} ---")

        # Avalia Modelo Treinado
        print("Executando avaliacao do modelo BumbaLM treinado...")
        res_treinado = executar_avaliacao_completa(
            model=modelo_treinado,
            caminho_processos=arq_processos,
            caminho_chunks=arq_chunks,
            output_dir=dir_saida / split_nome / "treinado",
            nome_avaliacao=f"{split_nome}_bumbalm",
            query_instruction=query_instruction,
            excluir_chunks_contaminados=True,
            batch_size=args.batch_size,
            avaliar_por_tribunal=True,
        )

        res_baseline: Optional[dict[str, Any]] = None
        if modelo_baseline is not None:
            print("\nExecutando avaliacao do modelo baseline zero-shot...")
            res_baseline = executar_avaliacao_completa(
                model=modelo_baseline,
                caminho_processos=arq_processos,
                caminho_chunks=arq_chunks,
                output_dir=dir_saida / split_nome / "baseline",
                nome_avaliacao=f"{split_nome}_baseline",
                query_instruction=query_instruction,
                excluir_chunks_contaminados=True,
                batch_size=args.batch_size,
                avaliar_por_tribunal=True,
            )

        # Gera relatorio comparativo
        caminho_relatorio_txt = dir_saida / f"comparativo_{split_nome}.txt"
        gerar_relatorio_comparativo(
            metricas_treinado=res_treinado.get("metricas_global", {}),
            metricas_baseline=res_baseline.get("metricas_global", {}) if res_baseline else None,
            nome_split=split_nome,
            caminho_saida=caminho_relatorio_txt,
        )

        relatorio_consolidado_geral[split_nome] = {
            "treinado": res_treinado,
            "baseline": res_baseline,
        }

        # Exibicao no terminal
        print("\n" + "=" * 64)
        print(f"RESUMO DOS RESULTADOS GLOBAIS - SPLIT: {split_nome.upper()}")
        print("=" * 64)
        with caminho_relatorio_txt.open(encoding="utf-8") as f:
            print(f.read())

    # Grava JSON unificado
    caminho_consolidado_json = dir_saida / "avaliacao_consolidada.json"
    with caminho_consolidado_json.open("w", encoding="utf-8") as f:
        json.dump(relatorio_consolidado_geral, f, indent=2, ensure_ascii=False)

    print("=" * 64)
    print("Avaliacao finalizada com sucesso!")
    print(f"Relatorios e metricas gravados em: {dir_saida}")
    print("=" * 64)


if __name__ == "__main__":
    main()
