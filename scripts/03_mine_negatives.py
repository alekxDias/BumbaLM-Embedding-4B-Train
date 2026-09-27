#!/usr/bin/env python3
"""
Pipeline BumbaLM - Etapa 3: Mineracao de Hard Negatives em GPU para o Treinamento.

Executa:
1. Leitura dos processos de treino ('train_processos.jsonl') e dos chunks ('train_corpus_chunks.jsonl').
2. Inicializacao do modelo base para inferencia vetorial acelerada (Qwen3).
3. Geracao de embeddings normalizados para todo o corpus de chunks do treino.
4. Comparacao matricial em GPU e selecao de 5 hard negatives por positivo (na faixa [0.60, 0.85]).
5. Persistencia das instancias de treino no formato de tripletos ('train_triplets.jsonl').
6. Emissao de relatorio com distribuicao de margens e cobertura de chunks.

Uso:
    python scripts/03_mine_negatives.py
    python scripts/03_mine_negatives.py --device cuda
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

# Adiciona a raiz do repositorio ao PYTHONPATH
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.data.schema import ChunkProcesso, ProcessoJudicial
from src.mining.hard_negatives import (
    carregar_modelo_mineracao,
    minerar_tripletos_contrastivos,
)


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
        "--config_data",
        type=Path,
        default=BASE_DIR / "configs" / "config_data.yaml",
        help="Caminho do config_data.yaml",
    )
    parser.add_argument(
        "--config_train",
        type=Path,
        default=BASE_DIR / "configs" / "config_train.yaml",
        help="Caminho do config_train.yaml",
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=None,
        help="Diretorio onde estao os arquivos processados.",
    )
    parser.add_argument(
        "--output_file",
        type=Path,
        default=None,
        help="Caminho do arquivo de tripletos de saida (padrao: data/processed/train_triplets.jsonl).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Dispositivo de execucao ('cuda' ou 'cpu'). Padrao: automatico.",
    )
    return parser.parse_args()


def carregar_processos_jsonl(caminho: Path) -> list[ProcessoJudicial]:
    """Le arquivo de processos persistido pelo splitter."""
    if not caminho.is_file():
        raise FileNotFoundError(f"Arquivo nao encontrado: {caminho}")

    processos = []
    with caminho.open(encoding="utf-8") as f:
        for num_linha, linha in enumerate(f, start=1):
            linha = linha.strip()
            if not linha:
                continue
            dado = json.loads(linha)
            p = ProcessoJudicial(
                tribunal=dado.get("str_tribunal", ""),
                numero_processo=dado.get("str_numero_unico", ""),
                ementa=dado.get("str_ementa", ""),
                inteiro_teor=dado.get("str_inteiro_teor", ""),
                metadados=dado.get("metadados", {}),
            )
            processos.append(p)
    return processos


def carregar_chunks_jsonl(caminho: Path) -> list[ChunkProcesso]:
    """Le colecao de chunks persistida pelo chunker."""
    if not caminho.is_file():
        raise FileNotFoundError(f"Arquivo de chunks nao encontrado: {caminho}")

    chunks = []
    with caminho.open(encoding="utf-8") as f:
        for linha in f:
            linha = linha.strip()
            if not linha:
                continue
            dado = json.loads(linha)
            ch = ChunkProcesso(
                chunk_id=dado.get("chunk_id", ""),
                id_global_processo=dado.get("id_global_processo", ""),
                tribunal=dado.get("str_tribunal", ""),
                numero_processo=dado.get("str_numero_unico", ""),
                chunk_idx=int(dado.get("chunk_idx", 0)),
                n_tokens=int(dado.get("n_tokens", 0)),
                texto=dado.get("texto", ""),
                contem_vazamento_ementa=bool(dado.get("contem_vazamento_ementa", False)),
                motivo_vazamento=dado.get("motivo_vazamento"),
            )
            chunks.append(ch)
    return chunks


def main() -> None:
    args = parse_args()

    cfg_base = carregar_yaml(args.config_base)
    cfg_data = carregar_yaml(args.config_data)
    cfg_train = carregar_yaml(args.config_train)

    dir_processados_str = str(args.data_dir) if args.data_dir else cfg_base.get("dados", {}).get("diretorio_processados", "data/processed")
    dir_processados = Path(dir_processados_str)
    if not dir_processados.is_absolute():
        dir_processados = BASE_DIR / dir_processados

    # Localiza arquivo de processos de treino (suporta train_processos ou treino_processos)
    arq_processos = dir_processados / "train_processos.jsonl"
    if not arq_processos.is_file():
        arq_processos = dir_processados / "treino_processos.jsonl"

    # Localiza arquivo de chunks de treino
    arq_chunks = dir_processados / "train_corpus_chunks.jsonl"
    if not arq_chunks.is_file():
        arq_chunks = dir_processados / "treino_corpus_chunks.jsonl"

    # Arquivo de tripletos final de saida
    arq_saida = args.output_file or (dir_processados / "train_triplets.jsonl")
    if not Path(arq_saida).is_absolute():
        arq_saida = BASE_DIR / arq_saida

    # Parametros de mineracao
    mining_cfg = cfg_data.get("mineracao_negativos", {})
    n_neg = mining_cfg.get("n_negativos", 5)
    lim_min = mining_cfg.get("limiar_similaridade_min", 0.60)
    lim_max = mining_cfg.get("limiar_similaridade_max", 0.85)
    top_k = mining_cfg.get("top_k_candidatos", 200)
    encode_batch = mining_cfg.get("encode_batch_size", 256)
    query_batch = mining_cfg.get("query_batch_size", 256)

    # Parametros do modelo e instrucao
    modelo_cfg = cfg_train.get("modelo", {})
    model_id = modelo_cfg.get("model_id", "Qwen/Qwen3-Embedding-4B")
    query_instruction = modelo_cfg.get(
        "query_instruction",
        "Instruct: Dado um trecho de um documento jurídico brasileiro, recupere o trecho mais relevante semanticamente.\nQuery: ",
    )
    attn_impl = modelo_cfg.get("attn_implementation", "sdpa")

    print("=" * 64)
    print("BumbaLM - Etapa 3: Mineracao de Hard Negatives em GPU")
    print("=" * 64)
    print(f"Processos de Treino : {arq_processos}")
    print(f"Chunks de Treino    : {arq_chunks}")
    print(f"Arquivo de Saida    : {arq_saida}")
    print(f"Modelo de Mineracao : {model_id}")
    print(f"Faixa de Cosseno    : [{lim_min}, {lim_max}] | TOP_K: {top_k}")
    print(f"Negativos por Chunk : {n_neg}")
    print(f"Batch Inferencia    : {encode_batch} | Batch Queries: {query_batch}")
    print("-" * 64)

    # 1. Carregamento dos dados em memoria
    print("Carregando processos e chunks de treino...")
    processos = carregar_processos_jsonl(arq_processos)
    chunks = carregar_chunks_jsonl(arq_chunks)

    print(f"Processos carregados: {len(processos):,}")
    print(f"Chunks carregados   : {len(chunks):,}")

    # 2. Inicializacao do modelo
    modelo = carregar_modelo_mineracao(
        model_id=model_id,
        device=args.device,
        attn_implementation=attn_impl,
    )

    # 3. Execucao da mineracao
    stats = minerar_tripletos_contrastivos(
        modelo=modelo,
        processos=processos,
        chunks_corpus=chunks,
        arquivo_saida=arq_saida,
        n_negativos=n_neg,
        limiar_min=lim_min,
        limiar_max=lim_max,
        top_k=top_k,
        encode_batch_size=encode_batch,
        query_batch_size=query_batch,
        query_instruction=query_instruction,
        ignorar_chunk_0_como_positivo=True,
    )

    # 4. Emissao de relatorio de mineracao
    relatorio_txt = dir_processados / "relatorio_mineracao_tripletos.txt"
    with relatorio_txt.open("w", encoding="utf-8") as f:
        f.write("=" * 64 + "\n")
        f.write("RELATORIO DE MINERACAO DE HARD NEGATIVES - BUMBALM\n")
        f.write("=" * 64 + "\n\n")
        f.write(f"Modelo Utilizado         : {model_id}\n")
        f.write(f"Faixa de Similaridade    : [{lim_min}, {lim_max}]\n")
        f.write(f"Negativos por Instancia  : {n_neg}\n")
        f.write(f"Processos Avaliados      : {stats['processos_avaliados']:,}\n")
        f.write(f"Positivos Avaliados      : {stats['chunks_positivos_avaliados']:,}\n")
        f.write(f"Tripletos Gerados        : {stats['tripletos_salvos']:,}\n")
        f.write(f"Chunks sem 5 Negativos   : {stats['chunks_sem_negativos_suficientes']:,}\n")
        f.write(f"Margem Media (Pos - Neg) : {stats['media_margem_tripletos']:.4f}\n")

    print("\n" + "=" * 64)
    print("Resumo da Mineracao de Tripletos:")
    print(f"  - Tripletos gerados com sucesso : {stats['tripletos_salvos']:,}")
    print(f"  - Chunks sem 5 negativos        : {stats['chunks_sem_negativos_suficientes']:,}")
    print(f"  - Margem media (pos - neg max)  : {stats['media_margem_tripletos']:.4f}")
    print(f"Arquivo de saida salvo em         : {arq_saida}")
    print(f"Relatorio salvo em                : {relatorio_txt}")
    print("=" * 64)
    print("Etapa 3 finalizada com sucesso.")


if __name__ == "__main__":
    main()
