#!/usr/bin/env python3
"""
Pipeline BumbaLM - Etapa 3b: Geracao de pares (ementa, chunk) sem hard negatives.

Monta instancias de treino apenas com ancora (ementa) e positivo (chunk do mesmo
processo), deixando negatives vazio. A CachedMNRL usara somente negativos in-batch.

Regras alinhadas a mineracao (03_mine_negatives / hard_negatives.py):
- Exclui chunk com vazamento de ementa quando o processo tem outros chunks.
- Nao usa GPU nem embeddings.

Uso:
    python scripts/03b_build_pairs.py
    python scripts/03b_build_pairs.py --data_dir data/processed_tjma
    python scripts/03b_build_pairs.py --output_file data/processed_tjma/train_pairs.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator

from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.data.schema import ChunkProcesso, ProcessoJudicial


def carregar_yaml(caminho: Path) -> dict[str, Any]:
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
        "--data_dir",
        type=Path,
        default=None,
        help="Diretorio com train_processos.jsonl e train_corpus_chunks.jsonl.",
    )
    parser.add_argument(
        "--output_file",
        type=Path,
        default=None,
        help="Saida JSONL (padrao: <data_dir>/train_pairs.jsonl).",
    )
    parser.add_argument(
        "--incluir_chunks_com_vazamento",
        action="store_true",
        help="Se informado, nao aplica o filtro de vazamento de ementa.",
    )
    return parser.parse_args()


def _ler_jsonl(caminho: Path, desc: str) -> Iterator[dict[str, Any]]:
    tamanho = caminho.stat().st_size
    with caminho.open("rb") as arquivo, tqdm(
        total=tamanho,
        desc=desc,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as barra:
        for linha_bytes in arquivo:
            barra.update(len(linha_bytes))
            texto = linha_bytes.decode("utf-8").strip()
            if texto:
                yield json.loads(texto)


def carregar_processos_jsonl(caminho: Path) -> list[ProcessoJudicial]:
    if not caminho.is_file():
        raise FileNotFoundError(f"Arquivo nao encontrado: {caminho}")
    processos = []
    for dado in _ler_jsonl(caminho, "Carregando processos"):
        dado.pop("str_inteiro_teor", None)
        processos.append(
            ProcessoJudicial(
                tribunal=dado.get("str_tribunal", ""),
                numero_processo=dado.get("str_numero_unico", ""),
                ementa=dado.get("str_ementa", ""),
                inteiro_teor="",
                metadados=dado.get("metadados", {}),
            )
        )
    return processos


def carregar_chunks_jsonl(caminho: Path) -> list[ChunkProcesso]:
    if not caminho.is_file():
        raise FileNotFoundError(f"Arquivo de chunks nao encontrado: {caminho}")
    chunks = []
    for dado in _ler_jsonl(caminho, "Carregando chunks"):
        chunks.append(
            ChunkProcesso(
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
        )
    return chunks


def gerar_pares(
    processos: list[ProcessoJudicial],
    chunks: list[ChunkProcesso],
    arquivo_saida: Path,
    filtrar_vazamento: bool = True,
) -> dict[str, int]:
    """
    Gera train_pairs.jsonl alinhado a regra de positivos da mineracao.
    """
    indices_por_processo: dict[str, list[int]] = defaultdict(list)
    for idx, ch in enumerate(chunks):
        indices_por_processo[ch.id_global_processo].append(idx)

    processos_validos = [p for p in processos if p.id_global in indices_por_processo]
    arquivo_saida.parent.mkdir(parents=True, exist_ok=True)

    stats = {
        "processos_com_chunks": len(processos_validos),
        "pares_salvos": 0,
        "chunks_pulados_vazamento": 0,
        "chunks_sem_texto": 0,
    }

    with arquivo_saida.open("w", encoding="utf-8") as f:
        for proc in tqdm(processos_validos, desc="Gerando pares", unit="proc"):
            query = str(proc.ementa or "").strip()
            if not query:
                continue

            pos_indices = indices_por_processo[proc.id_global]
            for pos_idx in pos_indices:
                ch = chunks[pos_idx]
                # Mesma regra da mineracao: pula leak se houver outro chunk no processo.
                if (
                    filtrar_vazamento
                    and ch.contem_vazamento_ementa
                    and len(pos_indices) > 1
                ):
                    stats["chunks_pulados_vazamento"] += 1
                    continue

                positive = str(ch.texto or "").strip()
                if not positive:
                    stats["chunks_sem_texto"] += 1
                    continue

                registro = {
                    "id_global": proc.id_global,
                    "str_tribunal": proc.tribunal,
                    "str_numero_unico": proc.numero_processo,
                    "query": query,
                    "positive": positive,
                    "positive_chunk_id": ch.chunk_id,
                    "negatives": [],
                    "negative_chunk_ids": [],
                }
                f.write(json.dumps(registro, ensure_ascii=False) + "\n")
                stats["pares_salvos"] += 1

    return stats


def main() -> None:
    args = parse_args()
    cfg_base = carregar_yaml(args.config_base)

    dir_processados_str = (
        str(args.data_dir)
        if args.data_dir
        else cfg_base.get("dados", {}).get("diretorio_processados", "data/processed")
    )
    dir_processados = Path(dir_processados_str)
    if not dir_processados.is_absolute():
        dir_processados = BASE_DIR / dir_processados

    arq_processos = dir_processados / "train_processos.jsonl"
    if not arq_processos.is_file():
        arq_processos = dir_processados / "treino_processos.jsonl"

    arq_chunks = dir_processados / "train_corpus_chunks.jsonl"
    if not arq_chunks.is_file():
        arq_chunks = dir_processados / "treino_corpus_chunks.jsonl"

    arq_saida = args.output_file or (dir_processados / "train_pairs.jsonl")
    if not Path(arq_saida).is_absolute():
        arq_saida = BASE_DIR / arq_saida

    print("=" * 64)
    print("BumbaLM - Etapa 3b: Pares sem hard negatives (in-batch)")
    print("=" * 64)
    print(f"Processos : {arq_processos}")
    print(f"Chunks    : {arq_chunks}")
    print(f"Saida     : {arq_saida}")
    print(f"Filtro vazamento: {not args.incluir_chunks_com_vazamento}")
    print("-" * 64)

    processos = carregar_processos_jsonl(arq_processos)
    chunks = carregar_chunks_jsonl(arq_chunks)
    stats = gerar_pares(
        processos=processos,
        chunks=chunks,
        arquivo_saida=arq_saida,
        filtrar_vazamento=not args.incluir_chunks_com_vazamento,
    )

    relatorio = dir_processados / "relatorio_pares_sem_hn.txt"
    with relatorio.open("w", encoding="utf-8") as f:
        f.write("Relatorio de pares sem hard negatives\n")
        f.write(f"Processos com chunks : {stats['processos_com_chunks']:,}\n")
        f.write(f"Pares salvos         : {stats['pares_salvos']:,}\n")
        f.write(f"Pulados (vazamento)  : {stats['chunks_pulados_vazamento']:,}\n")
        f.write(f"Sem texto            : {stats['chunks_sem_texto']:,}\n")

    print("=" * 64)
    print(f"Pares salvos              : {stats['pares_salvos']:,}")
    print(f"Chunks pulados (vazamento): {stats['chunks_pulados_vazamento']:,}")
    print(f"Relatorio                 : {relatorio}")
    print("=" * 64)


if __name__ == "__main__":
    main()
