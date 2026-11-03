#!/usr/bin/env python3
"""
Pipeline BumbaLM - Etapa 3b: Tuplas com negativos aleatorios (ablacao).

Executa:
1. Leitura de 'train_triplets.jsonl', gerado na Etapa 3 com hard negatives.
2. Leitura de 'train_corpus_chunks.jsonl'.
3. Troca dos 5 negativos de cada tupla por chunks sorteados de outros processos.
   Ancora e positivo nao mudam.
4. Gravacao de 'train_tuplas_aleatorias.jsonl'.

Nao usa GPU.

Uso:
    python scripts/03b_random_negatives.py
    python scripts/03b_random_negatives.py --seed 7
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import json

import numpy as np

from src.evaluation.ir_evaluator import carregar_chunks_jsonl, carregar_processos_jsonl
from src.mining.random_negatives import gerar_tuplas_negativos_aleatorios


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
    parser.add_argument("--config_base", type=Path, default=BASE_DIR / "configs" / "config_base.yaml")
    parser.add_argument("--config_data", type=Path, default=BASE_DIR / "configs" / "config_data.yaml")
    parser.add_argument("--data_dir", type=Path, default=None, help="Diretorio dos arquivos processados.")
    parser.add_argument("--seed", type=int, default=None, help="Semente do sorteio. Padrao: projeto.seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg_base = carregar_yaml(args.config_base)
    cfg_data = carregar_yaml(args.config_data)

    dir_processados = args.data_dir or Path(cfg_base.get("dados", {}).get("diretorio_processados", "data/processed"))
    if not dir_processados.is_absolute():
        dir_processados = BASE_DIR / dir_processados

    arq_tuplas = dir_processados / "train_triplets.jsonl"
    arq_chunks = dir_processados / "train_corpus_chunks.jsonl"
    arq_saida = dir_processados / "train_tuplas_aleatorias.jsonl"
    n_neg = cfg_data.get("mineracao_negativos", {}).get("n_negativos", 5)
    seed = args.seed if args.seed is not None else cfg_base.get("projeto", {}).get("seed", 42)

    print("=" * 64)
    print("BumbaLM - Etapa 3b: Negativos aleatorios (ablacao)")
    print("=" * 64)
    print(f"Tuplas com hard negatives : {arq_tuplas}")
    print(f"Chunks de treino          : {arq_chunks}")
    print(f"Saida                     : {arq_saida}")
    print(f"Negativos por tupla       : {n_neg} | Semente: {seed}")
    print("-" * 64)

    chunks = carregar_chunks_jsonl(arq_chunks)
    print(f"Chunks carregados: {len(chunks):,}")
    # Grupos de quase duplicatas gravados pela Etapa 1
    processos = carregar_processos_jsonl(dir_processados / "train_processos.jsonl")
    grupo_por_processo = {p.id_global: p.grupo_duplicata for p in processos}

    # Teto relativo: vetores do corpus (Etapa 3) e das consultas (gravados pela mineracao)
    max_relativo = cfg_data.get("mineracao_negativos", {}).get("max_relativo_positivo", 0.95)
    vetores_consultas = None
    corpus_embs = None
    pasta_consultas = dir_processados / "train_triplets_consultas"
    arq_corpus = dir_processados / "corpus_embeddings_f16.npy"
    if max_relativo:
        if not pasta_consultas.is_dir() or not arq_corpus.is_file():
            raise FileNotFoundError(
                f"Teto relativo ligado, mas faltam {pasta_consultas} ou {arq_corpus}. Rode a Etapa 3 antes."
            )
        corpus_embs = np.load(arq_corpus, mmap_mode="r")
        vetores_consultas = {}
        for arq in sorted(pasta_consultas.glob("lote_*.npy")):
            ids = json.loads(arq.with_suffix(".json").read_text(encoding="utf-8"))
            for pid, vetor in zip(ids, np.load(arq).astype(np.float32)):
                vetores_consultas[pid] = vetor
        print(f"Teto relativo: negativo < {float(max_relativo):.0%} do positivo | "
              f"{len(vetores_consultas):,} consultas com vetor")

    stats = gerar_tuplas_negativos_aleatorios(
        arquivo_tuplas_dificeis=arq_tuplas,
        chunks_corpus=chunks,
        arquivo_saida=arq_saida,
        n_negativos=n_neg,
        seed=seed,
        grupo_por_processo=grupo_por_processo,
        vetores_consultas=vetores_consultas,
        corpus_embs=corpus_embs,
        max_relativo_positivo=float(max_relativo) if max_relativo else None,
    )

    print("-" * 64)
    print(f"Tuplas lidas          : {stats['tuplas_lidas']:,}")
    print(f"Tuplas gravadas       : {stats['tuplas_gravadas']:,}")
    print(f"Tuplas sem negativos  : {stats['tuplas_sem_negativos']:,}")
    print(f"Sorteios rejeitados pelo teto relativo: {stats['rejeitados_teto_relativo']:,}")
    if stats["tuplas_sem_vetor_da_consulta"]:
        print(f"AVISO: {stats['tuplas_sem_vetor_da_consulta']:,} tuplas sem vetor da consulta (sorteadas sem o teto)")
    print(f"Arquivo salvo em      : {arq_saida}")
    print("Etapa 3b finalizada com sucesso.")


if __name__ == "__main__":
    main()
