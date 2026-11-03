"""
Negativos aleatorios para a ablacao contra os hard negatives.

Le as tuplas da mineracao e troca apenas os negativos. Ancora e positivo ficam iguais,
para a comparacao medir so o efeito de como os negativos foram escolhidos.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from tqdm import tqdm

from src.data.schema import ChunkProcesso


def _sortear_negativos(
    rng: random.Random,
    id_processo: str,
    chunks_corpus: list[ChunkProcesso],
    n_negativos: int,
    grupo_por_processo: dict[str, str] | None = None,
    max_tentativas: int = 1000,
) -> list[int]:
    # Sorteio com rejeicao. Mesmas regras dos hard negatives: fora o proprio processo,
    # os processos quase duplicados e os chunks com a ementa copiada.
    grupos = grupo_por_processo or {}
    grupo_alvo = grupos.get(id_processo, id_processo)
    escolhidos: list[int] = []
    vistos: set[int] = set()
    tentativas = 0
    while len(escolhidos) < n_negativos and tentativas < max_tentativas:
        tentativas += 1
        idx = rng.randrange(len(chunks_corpus))
        if idx in vistos:
            continue
        vistos.add(idx)
        ch = chunks_corpus[idx]
        if ch.contem_vazamento_ementa:
            continue
        if grupos.get(ch.id_global_processo, ch.id_global_processo) == grupo_alvo:
            continue
        escolhidos.append(idx)
    return escolhidos


def gerar_tuplas_negativos_aleatorios(
    arquivo_tuplas_dificeis: Path | str,
    chunks_corpus: list[ChunkProcesso],
    arquivo_saida: Path | str,
    n_negativos: int = 5,
    seed: int = 42,
    grupo_por_processo: dict[str, str] | None = None,
) -> dict[str, Any]:
    """
    Grava um JSONL com o mesmo formato de train_triplets.jsonl.
    Os negativos vem de processos diferentes, sorteados de forma uniforme no corpus de treino.
    grupo_por_processo ({id_global: grupo_duplicata}) exclui tambem os quase duplicados.
    """
    entrada = Path(arquivo_tuplas_dificeis)
    saida = Path(arquivo_saida)
    saida.parent.mkdir(parents=True, exist_ok=True)
    if not entrada.is_file():
        raise FileNotFoundError(f"Tuplas com hard negatives nao encontradas: {entrada}")
    if not chunks_corpus:
        raise ValueError("Corpus de chunks vazio: nao ha de onde sortear negativos.")

    rng = random.Random(seed)
    stats = {"tuplas_lidas": 0, "tuplas_gravadas": 0, "tuplas_sem_negativos": 0}
    temporario = saida.with_name(saida.name + ".tmp")

    with entrada.open(encoding="utf-8") as f_in, temporario.open("w", encoding="utf-8", newline="\n") as f_out:
        for linha in tqdm(f_in, desc="Sorteando negativos aleatorios", unit="tupla"):
            linha = linha.strip()
            if not linha:
                continue
            stats["tuplas_lidas"] += 1
            item = json.loads(linha)
            id_processo = str(item.get("id_global", ""))

            indices = _sortear_negativos(rng, id_processo, chunks_corpus, n_negativos, grupo_por_processo)
            if len(indices) < n_negativos:
                stats["tuplas_sem_negativos"] += 1
                continue

            item["negatives"] = [chunks_corpus[i].texto for i in indices]
            item["negative_chunk_ids"] = [chunks_corpus[i].chunk_id for i in indices]
            # Os scores e a margem eram do modelo base sobre os hard negatives. Aqui nao valem.
            item["scores_negativos"] = None
            item["margin"] = None
            item["estrategia_negativos"] = "aleatorio"
            f_out.write(json.dumps(item, ensure_ascii=False) + "\n")
            stats["tuplas_gravadas"] += 1

    temporario.replace(saida)
    return stats
