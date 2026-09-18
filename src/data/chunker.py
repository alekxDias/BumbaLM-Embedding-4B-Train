"""
Fragmentacao do inteiro teor em janelas deslizantes baseadas em tokens (Chunking).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tqdm import tqdm
from transformers import AutoTokenizer

from src.data.schema import ChunkProcesso, ProcessoJudicial


def fragmentar_texto_em_chunks(
    texto: str,
    processo: ProcessoJudicial,
    tokenizer: AutoTokenizer,
    chunk_size: int = 768,
    overlap: int = 128,
) -> list[ChunkProcesso]:
    """
    Divide o texto do inteiro teor em janelas de tokens com sobreposicao fixa.

    Args:
        texto: Texto integral do acordao.
        processo: Instancia do ProcessoJudicial de origem.
        tokenizer: Tokenizador do modelo de embedding (Qwen3).
        chunk_size: Tamanho maximo de cada fragmento em tokens (padrao: 768).
        overlap: Quantidade de tokens compartilhados entre janelas consecutivas (padrao: 128).

    Returns:
        Lista de instancias de ChunkProcesso.
    """
    if overlap >= chunk_size:
        raise ValueError("O overlap deve ser estritamente menor que chunk_size.")

    ids = tokenizer.encode(texto, add_special_tokens=False)
    if not ids:
        return []

    stride = chunk_size - overlap
    chunks: list[ChunkProcesso] = []
    start = 0
    idx = 0

    while start < len(ids):
        end = min(start + chunk_size, len(ids))
        piece_ids = ids[start:end]
        piece_text = tokenizer.decode(piece_ids, skip_special_tokens=True).strip()

        if piece_text:
            chunk_id = f"{processo.id_global}__{idx}"
            chunks.append(
                ChunkProcesso(
                    chunk_id=chunk_id,
                    id_global_processo=processo.id_global,
                    tribunal=processo.tribunal,
                    numero_processo=processo.numero_processo,
                    chunk_idx=idx,
                    n_tokens=len(piece_ids),
                    texto=piece_text,
                )
            )
            idx += 1

        if end >= len(ids):
            break
        start += stride

    return chunks


def fragmentar_colecao_processos(
    processos: list[ProcessoJudicial],
    tokenizer: AutoTokenizer,
    chunk_size: int = 768,
    overlap: int = 128,
    desc_progresso: str = "Processando Chunks",
) -> tuple[list[ChunkProcesso], dict[str, Any]]:
    """
    Fragmenta uma lista de processos judiciais gerando a colecao consolidada de chunks.

    Returns:
        Tupla com (lista_de_ChunkProcesso, relatorio_estatistico).
    """
    todos_chunks: list[ChunkProcesso] = []
    chunks_por_processo: list[int] = []
    processos_com_um_chunk = 0

    for proc in tqdm(processos, desc=desc_progresso):
        chunks_proc = fragmentar_texto_em_chunks(
            texto=proc.inteiro_teor,
            processo=proc,
            tokenizer=tokenizer,
            chunk_size=chunk_size,
            overlap=overlap,
        )
        if not chunks_proc:
            continue

        n = len(chunks_proc)
        chunks_por_processo.append(n)
        if n == 1:
            processos_com_um_chunk += 1

        todos_chunks.extend(chunks_proc)

    estatisticas = {
        "total_processos": len(processos),
        "processos_com_chunks": len(chunks_por_processo),
        "total_chunks_gerados": len(todos_chunks),
        "media_chunks_por_processo": round(len(todos_chunks) / max(1, len(chunks_por_processo)), 2),
        "processos_chunk_unico": processos_com_um_chunk,
    }

    return todos_chunks, estatisticas


def salvar_chunks_jsonl(caminho_arquivo: Path | str, chunks: list[ChunkProcesso]) -> None:
    """Grava a lista de chunks em arquivo JSONL."""
    caminho = Path(caminho_arquivo)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with caminho.open("w", encoding="utf-8") as f:
        for ch in chunks:
            f.write(json.dumps(ch.to_dict(), ensure_ascii=False) + "\n")
