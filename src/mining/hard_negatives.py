"""
Mineracao vetorial acelerada de Hard Negatives em GPU para aprendizado contrastivo.
"""

from __future__ import annotations

import gc
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.data.schema import ChunkProcesso, ProcessoJudicial


DEFAULT_QUERY_INSTRUCTION = (
    "Instruct: Dado um trecho de um documento jurídico brasileiro, "
    "recupere o trecho mais relevante semanticamente.\nQuery: "
)


def preparar_query_com_instrucao(query: str, instrucao: str = DEFAULT_QUERY_INSTRUCTION) -> str:
    """
    Aplica a instrucao formal na ementa (query).
    Os documentos candidatos (positivos e negativos) nao devem receber o prefixo.
    """
    texto = str(query or "").strip()
    if texto.startswith("Instruct:"):
        return texto
    return f"{instrucao}{texto}"


def carregar_modelo_mineracao(
    model_id: str = "Qwen/Qwen3-Embedding-4B",
    max_seq_len: int = 1024,
    device: Optional[str] = None,
    attn_implementation: str = "sdpa",
) -> SentenceTransformer:
    """
    Inicializa o modelo SentenceTransformer para inferencia vetorial e calculo de similaridades.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Carregando modelo de mineracao: {model_id} no dispositivo: {device}...")
    modelo = SentenceTransformer(
        model_id,
        model_kwargs={
            "torch_dtype": torch.bfloat16 if device == "cuda" else torch.float32,
            "trust_remote_code": True,
            "attn_implementation": attn_implementation if device == "cuda" else "eager",
        },
        tokenizer_kwargs={"padding_side": "left"},
        device=device,
    )
    modelo.max_seq_length = max_seq_len
    return modelo


def encodar_textos_corpus(
    modelo: SentenceTransformer,
    textos: list[str],
    batch_size: int = 256,
    show_progress: bool = True,
) -> np.ndarray:
    """
    Gera representacoes vetoriais normalizadas (norma L2 = 1.0) para todos os textos do corpus.
    """
    embeddings = modelo.encode(
        textos,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return embeddings


def selecionar_hard_negatives(
    sims_q: np.ndarray,
    sims_p: np.ndarray,
    indices_mesmo_processo: list[int],
    n_negativos: int = 5,
    limiar_min: float = 0.60,
    limiar_max: float = 0.85,
    top_k: int = 200,
) -> list[tuple[int, float]]:
    """
    Seleciona os melhores hard negatives para um determinado chunk positivo.

    Estrategia de selecao:
    1. Mascara estritamente todos os chunks pertencentes ao mesmo processo (evita falsos negativos).
    2. Identifica os top_k chunks mais proximos vetorialmente da consulta (ementa).
    3. Filtra apenas os candidatos dentro da faixa informatica [limiar_min, limiar_max].
    4. Reordena os candidatos pela similaridade direta com o chunk positivo.
    5. Retorna os n_negativos com maior similaridade com o chunk positivo dentro da faixa permitida.
    """
    sims_q_mascarado = sims_q.copy()
    sims_q_mascarado[indices_mesmo_processo] = -np.inf
    candidatos_top_k = np.argsort(sims_q_mascarado)[::-1][:top_k]

    candidatos_validos = []
    for idx in candidatos_top_k:
        score_q = float(sims_q[idx])
        if score_q < limiar_min or score_q > limiar_max:
            continue
        score_p = float(sims_p[idx])
        candidatos_validos.append((int(idx), score_q, score_p))

    # Ordena prioritariamente pela similaridade com o positivo, seguido pela query
    candidatos_validos.sort(key=lambda x: (x[2], x[1]), reverse=True)
    return [(idx, score_q) for idx, score_q, _ in candidatos_validos[:n_negativos]]


def minerar_tripletos_contrastivos(
    modelo: SentenceTransformer,
    processos: list[ProcessoJudicial],
    chunks_corpus: list[ChunkProcesso],
    arquivo_saida: Path | str,
    n_negativos: int = 5,
    limiar_min: float = 0.60,
    limiar_max: float = 0.85,
    top_k: int = 200,
    encode_batch_size: int = 256,
    query_batch_size: int = 256,
    query_instruction: str = DEFAULT_QUERY_INSTRUCTION,
    ignorar_chunk_0_como_positivo: bool = True,
) -> dict[str, Any]:
    """
    Executa a mineracao de tripletos em lote utilizando aceleracao matricial na GPU.

    Args:
        modelo: Modelo SentenceTransformer carregado.
        processos: Lista de processos judiciais de treino.
        chunks_corpus: Colecao consolidada de chunks do treino.
        arquivo_saida: Caminho do arquivo JSONL onde os tripletos serao salvos.
        n_negativos: Quantidade de hard negatives por positivo (padrao: 5).
        limiar_min: Similaridade minima com a query (padrao: 0.60).
        limiar_max: Similaridade maxima com a query (padrao: 0.85).
        top_k: Numero de candidatos preliminares examinados por query.
        encode_batch_size: Lote de inferencia para vetorizar chunks.
        query_batch_size: Lote de processos para comparacao matricial na GPU.
        query_instruction: Prefixo de instrucao aplicado na query.
        ignorar_chunk_0_como_positivo: Se True e houver outros chunks limpos no processo,
            evita usar o Chunk 0 contaminado como documento positivo principal.

    Returns:
        Dicionario com estatisticas consolidadas da mineracao.
    """
    saida = Path(arquivo_saida)
    saida.parent.mkdir(parents=True, exist_ok=True)

    # 1. Mapeamentos e indexacao rapida
    indices_por_processo: dict[str, list[int]] = defaultdict(list)
    for idx, ch in enumerate(chunks_corpus):
        indices_por_processo[ch.id_global_processo].append(idx)

    processos_validos = [p for p in processos if p.id_global in indices_por_processo]
    textos_corpus = [ch.texto for ch in chunks_corpus]

    print(f"Total de chunks a vetorizar: {len(chunks_corpus):,}")
    print(f"Total de processos para consulta: {len(processos_validos):,}")

    # 2. Vetorizacao de todo o corpus de chunks
    print("Gerando embeddings de todos os chunks do corpus ...")
    corpus_embs = encodar_textos_corpus(
        modelo=modelo,
        textos=textos_corpus,
        batch_size=encode_batch_size,
        show_progress=True,
    )

    device = modelo.device
    print(f"Carregando matriz de embeddings ({corpus_embs.shape}) na memoria de calculo ({device})...")
    corpus_embs_tensor = torch.from_numpy(corpus_embs).to(device=device, dtype=torch.float32)

    stats = {
        "processos_avaliados": 0,
        "tripletos_salvos": 0,
        "chunks_positivos_avaliados": 0,
        "chunks_sem_negativos_suficientes": 0,
        "media_margem_tripletos": 0.0,
    }

    soma_margens = 0.0

    # 3. Processamento em lote das consultas
    with saida.open("w", encoding="utf-8") as f_out:
        for start in tqdm(range(0, len(processos_validos), query_batch_size), desc="Minerando Tripletos"):
            batch_procs = processos_validos[start : start + query_batch_size]
            queries_formatadas = [
                preparar_query_com_instrucao(p.ementa, query_instruction)
                for p in batch_procs
            ]

            with torch.no_grad():
                q_embs = modelo.encode(
                    queries_formatadas,
                    batch_size=encode_batch_size,
                    normalize_embeddings=True,
                    convert_to_tensor=True,
                    show_progress_bar=False,
                ).to(dtype=torch.float32)

                # Multiplicacao de matrizes: cos(queries_batch, corpus) -> Shape (B, N)
                sims_queries_gpu = torch.matmul(q_embs, corpus_embs_tensor.T)
                sims_queries = sims_queries_gpu.cpu().numpy()

            for i_local, proc in enumerate(batch_procs):
                stats["processos_avaliados"] += 1
                pos_indices = indices_por_processo[proc.id_global]
                sims_q = sims_queries[i_local]

                # Obtem embeddings dos positivos deste processo para cos(pos, corpus)
                pos_embs_gpu = corpus_embs_tensor[pos_indices]
                sims_p_gpu = torch.matmul(pos_embs_gpu, corpus_embs_tensor.T)
                sims_p_matriz = sims_p_gpu.cpu().numpy()

                for j_local, pos_idx in enumerate(pos_indices):
                    ch_positivo = chunks_corpus[pos_idx]

                    # Filtro inteligente do Chunk 0: se o chunk contem vazamento de ementa
                    # e o processo possui outros chunks limpos, nao o usa como positivo
                    if (
                        ignorar_chunk_0_como_positivo
                        and ch_positivo.contem_vazamento_ementa
                        and len(pos_indices) > 1
                    ):
                        continue

                    stats["chunks_positivos_avaliados"] += 1

                    negs = selecionar_hard_negatives(
                        sims_q=sims_q,
                        sims_p=sims_p_matriz[j_local],
                        indices_mesmo_processo=pos_indices,
                        n_negativos=n_negativos,
                        limiar_min=limiar_min,
                        limiar_max=limiar_max,
                        top_k=top_k,
                    )

                    if len(negs) < n_negativos:
                        stats["chunks_sem_negativos_suficientes"] += 1
                        continue

                    score_pos = round(float(sims_q[pos_idx]), 4)
                    neg_textos = [textos_corpus[idx] for idx, _ in negs]
                    neg_scores = [round(s, 4) for _, s in negs]
                    neg_ids = [chunks_corpus[idx].chunk_id for idx, _ in negs]
                    margem = round(score_pos - max(neg_scores), 4)

                    registro = {
                        "id_global": proc.id_global,
                        "str_tribunal": proc.tribunal,
                        "str_numero_unico": proc.numero_processo,
                        "query": proc.ementa,
                        "positive": ch_positivo.texto,
                        "positive_chunk_id": ch_positivo.chunk_id,
                        "positive_chunk_idx": ch_positivo.chunk_idx,
                        "negatives": neg_textos,
                        "negative_chunk_ids": neg_ids,
                        "score_positive": score_pos,
                        "scores_negativos": neg_scores,
                        "margin": margem,
                    }

                    f_out.write(json.dumps(registro, ensure_ascii=False) + "\n")
                    stats["tripletos_salvos"] += 1
                    soma_margens += margem

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if stats["tripletos_salvos"] > 0:
        stats["media_margem_tripletos"] = round(soma_margens / stats["tripletos_salvos"], 4)

    return stats
