#!/usr/bin/env python3
"""
Pipeline BumbaLM - Etapa 2: Geracao de Chunks por Tokens e Auditoria de Vazamento.

Executa:
1. Leitura dos processos particionados ('train_processos.jsonl', 'val_processos.jsonl', 'test_processos.jsonl').
2. Carregamento do tokenizador do modelo base (Qwen3).
3. Fragmentacao do inteiro teor de cada processo em janelas de 768 tokens (com overlap de 128).
4. Auditoria sistematica contra vazamento da ementa (marcando o Chunk 0 contaminado).
5. Gravacao das colecoes de chunks e geracao do relatorio estatistico de auditoria.

Uso:
    python scripts/02_generate_chunks.py
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

from transformers import AutoTokenizer

from src.data.chunker import fragmentar_texto_em_chunks, salvar_chunks_jsonl
from src.data.schema import ChunkProcesso, ProcessoJudicial
from src.evaluation.leakage_audit import auditar_e_marcar_chunks, gerar_relatorio_auditoria


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
        "--data_dir",
        type=Path,
        default=None,
        help="Diretorio onde estao os arquivos train/val/test_processos.jsonl.",
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


def processar_split(
    nome_split: str,
    processos: list[ProcessoJudicial],
    tokenizer: AutoTokenizer,
    chunk_size: int,
    overlap: int,
    trecho_minimo_auditoria: int,
) -> tuple[list[ChunkProcesso], dict]:
    """
    Executa chunking e auditoria de vazamento para uma determinada particao.
    """
    chunks_split: list[ChunkProcesso] = []

    for proc in processos:
        chunks_proc = fragmentar_texto_em_chunks(
            texto=proc.inteiro_teor,
            processo=proc,
            tokenizer=tokenizer,
            chunk_size=chunk_size,
            overlap=overlap,
        )
        if not chunks_proc:
            continue

        # Realiza a marcacao de vazamento de ementa em cada chunk
        chunks_auditados = auditar_e_marcar_chunks(
            processo=proc,
            chunks=chunks_proc,
            trecho_minimo=trecho_minimo_auditoria,
        )
        chunks_split.extend(chunks_auditados)

    relatorio = gerar_relatorio_auditoria(chunks_split)
    return chunks_split, relatorio


def main() -> None:
    args = parse_args()

    cfg_base = carregar_yaml(args.config_base)
    cfg_data = carregar_yaml(args.config_data)

    dir_processados_str = str(args.data_dir) if args.data_dir else cfg_base.get("dados", {}).get("diretorio_processados", "data/processed")
    dir_processados = Path(dir_processados_str)
    if not dir_processados.is_absolute():
        dir_processados = BASE_DIR / dir_processados

    chunk_cfg = cfg_data.get("chunking", {})
    model_id_tokenizer = chunk_cfg.get("model_id_tokenizer", "Qwen/Qwen3-Embedding-4B")
    chunk_size = chunk_cfg.get("chunk_size_tokens", 768)
    overlap = chunk_cfg.get("chunk_overlap_tokens", 128)
    trecho_minimo = chunk_cfg.get("min_caracteres_contaminacao", 200)

    print("=" * 64)
    print("BumbaLM - Etapa 2: Chunking por Tokens e Auditoria de Ementa")
    print("=" * 64)
    print(f"Diretorio de Dados  : {dir_processados}")
    print(f"Modelo Tokenizador  : {model_id_tokenizer}")
    print(f"Tamanho do Chunk    : {chunk_size} tokens | Overlap: {overlap} tokens")
    print(f"Trecho Minimo Audit : {trecho_minimo} caracteres normalizados")
    print("-" * 64)

    # 1. Carregamento do tokenizador
    print(f"Inicializando tokenizador: {model_id_tokenizer} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_id_tokenizer, trust_remote_code=True)

    arquivos_splits = {
        "treino": dir_processados / "train_processos.jsonl",
        "validacao": dir_processados / "val_processos.jsonl",
        "teste": dir_processados / "test_processos.jsonl",
    }

    relatorios_splits = {}
    arquivos_gerados = []

    # 2. Processamento de cada particao
    for nome, caminho in arquivos_splits.items():
        if not caminho.is_file():
            print(f"Aviso: Arquivo {caminho.name} nao encontrado. Pulei o split '{nome}'.")
            continue

        print(f"\nProcessando particao '{nome}' ({caminho.name}) ...")
        processos = carregar_processos_jsonl(caminho)
        chunks, relatorio = processar_split(
            nome_split=nome,
            processos=processos,
            tokenizer=tokenizer,
            chunk_size=chunk_size,
            overlap=overlap,
            trecho_minimo_auditoria=trecho_minimo,
        )

        caminho_saida_chunks = dir_processados / f"{nome}_corpus_chunks.jsonl"
        salvar_chunks_jsonl(caminho_saida_chunks, chunks)
        arquivos_gerados.append(caminho_saida_chunks)

        relatorios_splits[nome] = relatorio

        print(f"  - Total de processos  : {len(processos):,}")
        print(f"  - Total de chunks     : {relatorio['total_chunks']:,}")
        print(f"  - Chunks limpos       : {relatorio['limpos']:,} ({(1 - relatorio['taxa_contaminacao']):.1%})")
        print(f"  - Chunks contaminados : {relatorio['contaminados']:,} ({relatorio['taxa_contaminacao']:.1%})")
        print(f"    (Chunks com ementa identificados principalmente no Chunk 0)")

    # 3. Emissao do relatorio de auditoria consolidado
    relatorio_txt = dir_processados / "relatorio_auditoria_chunks.txt"
    with relatorio_txt.open("w", encoding="utf-8") as f:
        f.write("=" * 64 + "\n")
        f.write("RELATORIO DE AUDITORIA DE CHUNKING E VAZAMENTO DE EMENTA\n")
        f.write("=" * 64 + "\n\n")
        f.write(f"Configuracao: chunk_size={chunk_size} tokens | overlap={overlap} tokens\n")
        f.write(f"Tokenizador : {model_id_tokenizer}\n\n")

        for split_nome, rel in relatorios_splits.items():
            f.write(f"--- SPLIT: {split_nome.upper()} ---\n")
            f.write(f"Total de Chunks       : {rel['total_chunks']:,}\n")
            f.write(f"Chunks Limpos         : {rel['limpos']:,}\n")
            f.write(f"Chunks Contaminados   : {rel['contaminados']:,} ({rel['taxa_contaminacao']:.1%})\n")
            f.write("Motivos da Contaminacao:\n")
            for mot, qtd in rel["motivos"].items():
                f.write(f"  - {mot}: {qtd:,}\n")
            f.write("Contaminacao por Posicao do Chunk:\n")
            for ch_idx, qtd in rel["distribuicao_por_indice_chunk"].items():
                f.write(f"  - {ch_idx}: {qtd:,}\n")
            f.write("\n")

    arquivos_gerados.append(relatorio_txt)

    print("\n" + "=" * 64)
    print("Arquivos de Chunks e Auditoria Gerados:")
    for arq in arquivos_gerados:
        print(f"  - {arq}")
    print("=" * 64)
    print("Etapa 2 finalizada com sucesso.")


if __name__ == "__main__":
    main()
