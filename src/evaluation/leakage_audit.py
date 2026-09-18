"""
Auditoria de vazamento de ementa no inteiro teor (Problema do Chunk 0).
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Optional

from src.data.schema import ChunkProcesso, ProcessoJudicial


_ROTULOS_INICIAIS = re.compile(
    r"^(?:ementa|acordao|relatorio|voto(?: do relator)?|decisao|"
    r"vistos?|relatados?|discutidos?|e acordam)\s+"
)


def normalizar_texto_auditoria(texto: str) -> str:
    """
    Remove acentos, pontuacoes e espacos extras para comparacao lexico-semantica uniforme.
    """
    texto = unicodedata.normalize("NFKD", str(texto or "").lower())
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", " ", texto).strip()


def remover_rotulos_iniciais(texto: str) -> str:
    """Remove palavras de cabecalho judiciario no inicio do texto."""
    while True:
        novo = _ROTULOS_INICIAIS.sub("", texto)
        if novo == texto:
            return texto
        texto = novo


def classificar_contaminacao(
    ementa: str,
    texto_chunk: str,
    trecho_minimo: int = 200,
    min_chunk_na_ementa: int = 100,
) -> Optional[str]:
    """
    Classifica se o chunk contem vazamento espurio do texto da ementa.

    Retornos possiveis:
      - None: O chunk e genuina fundamentacao sem vazamento.
      - "inteira": A ementa completa esta reproduzida dentro do chunk.
      - "chunk_na_ementa": O chunk e curto e esta contido dentro da ementa.
      - "trecho": Uma janela contigua longa (>= trecho_minimo caracteres) da ementa
        esta presente de forma identica dentro do chunk.
    """
    consulta = remover_rotulos_iniciais(normalizar_texto_auditoria(ementa))
    chunk = remover_rotulos_iniciais(normalizar_texto_auditoria(texto_chunk))

    if not consulta or not chunk:
        return None

    if consulta in chunk:
        return "inteira"

    if len(chunk) >= min_chunk_na_ementa and chunk in consulta:
        return "chunk_na_ementa"

    if len(consulta) < trecho_minimo or len(chunk) < trecho_minimo:
        return None

    # Verificacao por janela deslizante de caracteres
    janelas = {
        chunk[i : i + trecho_minimo]
        for i in range(len(chunk) - trecho_minimo + 1)
    }
    for i in range(len(consulta) - trecho_minimo + 1):
        if consulta[i : i + trecho_minimo] in janelas:
            return "trecho"

    return None


def auditar_e_marcar_chunks(
    processo: ProcessoJudicial,
    chunks: list[ChunkProcesso],
    trecho_minimo: int = 200,
) -> list[ChunkProcesso]:
    """
    Analisa todos os chunks de um determinado processo judicial, marcando
    os campos 'contem_vazamento_ementa' e 'motivo_vazamento'.
    """
    for ch in chunks:
        motivo = classificar_contaminacao(
            ementa=processo.ementa,
            texto_chunk=ch.texto,
            trecho_minimo=trecho_minimo,
        )
        if motivo is not None:
            ch.contem_vazamento_ementa = True
            ch.motivo_vazamento = motivo

    return chunks


def gerar_relatorio_auditoria(chunks: list[ChunkProcesso]) -> dict:
    """
    Consolida metricas estatisticas de contaminacao de ementa sobre uma colecao de chunks.
    """
    total = len(chunks)
    contaminados = sum(1 for c in chunks if c.contem_vazamento_ementa)
    limpos = total - contaminados

    por_motivo = Counter(c.motivo_vazamento for c in chunks if c.contem_vazamento_ementa)
    por_chunk_idx = Counter(c.chunk_idx for c in chunks if c.contem_vazamento_ementa)

    return {
        "total_chunks": total,
        "limpos": limpos,
        "contaminados": contaminados,
        "taxa_contaminacao": round(contaminados / max(1, total), 4),
        "motivos": dict(por_motivo),
        "distribuicao_por_indice_chunk": {f"chunk_{k}": v for k, v in sorted(por_chunk_idx.items())},
    }
