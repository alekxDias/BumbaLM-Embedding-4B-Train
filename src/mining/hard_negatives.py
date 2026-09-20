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

    dtype_modelo = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else (torch.float16 if device == "cuda" else torch.float32)

    print(f"Carregando modelo de mineracao: {model_id} no dispositivo: {device} (dtype: {dtype_modelo})...")
    modelo = SentenceTransformer(
        model_id,
        model_kwargs={
            "torch_dtype": dtype_modelo,
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
    Gera representacoes vetoriais normalizadas para todos os textos do corpus.
    """
    embeddings = modelo.encode(
        textos,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return embeddings


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
    Executa a mineracao em lote de tripletos.

    Otimizacoes:
    - Processamento Vetorial Massivo: Multiplicacao matricial nativa na GPU utilizando precisao bfloat16.
    - Mascaramento In-Place: Anulação imediata de candidatos pertencentes ao mesmo processo diretamente na memoria da GPU.
    - Mitigacao de Gargalo de I/O: Execucao do torch.topk diretamente na GPU. Apenas os indices estritamente necessarios sao transferidos para a CPU.
    - Avaliacao Focada: O calculo da similaridade de cosseno contra o texto positivo (ancora) e restrito ao subconjunto Top-K.
    """
    saida = Path(arquivo_saida)
    saida.parent.mkdir(parents=True, exist_ok=True)

    device = modelo.device
    dtype_tensor = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else (torch.float16 if device.type == "cuda" else torch.float32)

    # 1. Indexacao rapida de processos
    indices_por_processo: dict[str, list[int]] = defaultdict(list)
    for idx, ch in enumerate(chunks_corpus):
        indices_por_processo[ch.id_global_processo].append(idx)

    processos_validos = [p for p in processos if p.id_global in indices_por_processo]
    textos_corpus = [ch.texto for ch in chunks_corpus]
    total_chunks = len(chunks_corpus)

    print(f"Total de chunks a vetorizar: {total_chunks:,}")
    print(f"Total de processos para consulta: {len(processos_validos):,}")

    # 2. Vetorizacao de todo o corpus de chunks
    print("Gerando embeddings de todos os chunks do corpus ...")
    corpus_embs_cpu = encodar_textos_corpus(
        modelo=modelo,
        textos=textos_corpus,
        batch_size=encode_batch_size,
        show_progress=True,
    )

    print(f"Transferindo matriz de embeddings ({corpus_embs_cpu.shape}) para {device} em {dtype_tensor}...")
    corpus_embs_gpu = torch.from_numpy(corpus_embs_cpu).to(device=device, dtype=dtype_tensor)

    stats = {
        "processos_avaliados": 0,
        "tripletos_salvos": 0,
        "chunks_positivos_avaliados": 0,
        "chunks_sem_negativos_suficientes": 0,
        "media_margem_tripletos": 0.0,
    }

    soma_margens = 0.0
    k_candidatos_gpu = min(max(top_k * 2, n_negativos * 10), total_chunks)

    # 3. Processamento em lote das consultas
    with saida.open("w", encoding="utf-8") as f_out:
        for start in tqdm(range(0, len(processos_validos), query_batch_size), desc="Minerando Tripletos (GPU Otimizada)"):
            batch_procs = processos_validos[start : start + query_batch_size]
            queries_formatadas = [
                preparar_query_com_instrucao(p.ementa, query_instruction)
                for p in batch_procs
            ]

            with torch.no_grad():
                # Vetorizacao em lote das consultas
                q_embs_gpu = modelo.encode(
                    queries_formatadas,
                    batch_size=encode_batch_size,
                    normalize_embeddings=True,
                    convert_to_tensor=True,
                    show_progress_bar=False,
                ).to(dtype=dtype_tensor, device=device)

                # Multiplicacao de matrizes em lote na GPU: (B, D) x (D, N) -> Shape (B, N)
                sims_batch_gpu = torch.matmul(q_embs_gpu, corpus_embs_gpu.T)

                # Mascara chunks do mesmo processo judicial diretamente na GPU
                for i_local, proc in enumerate(batch_procs):
                    pos_idx_list = indices_por_processo[proc.id_global]
                    sims_batch_gpu[i_local, pos_idx_list] = -1e9

                # Top-K direto na GPU: extrai apenas os candidatos mais proximos
                top_scores_gpu, top_indices_gpu = torch.topk(sims_batch_gpu, k=k_candidatos_gpu, dim=-1)

                # Transfere para CPU apenas os indices e scores necessarios (array compacto)
                top_scores_cpu = top_scores_gpu.cpu().float().numpy()
                top_indices_cpu = top_indices_gpu.cpu().numpy()
                q_embs_cpu = q_embs_gpu.cpu().float().numpy()

            # Processamento rapido dos candidatos selecionados
            for i_local, proc in enumerate(batch_procs):
                stats["processos_avaliados"] += 1
                pos_indices = indices_por_processo[proc.id_global]
                q_emb_vec = q_embs_cpu[i_local]

                cands_idx = top_indices_cpu[i_local]
                cands_score_q = top_scores_cpu[i_local]

                # Filtra candidatos na faixa [limiar_min, limiar_max]
                candidatos_validos = []
                for c_idx, score_q in zip(cands_idx, cands_score_q):
                    if limiar_min <= score_q <= limiar_max:
                        candidatos_validos.append((int(c_idx), float(score_q)))

                if len(candidatos_validos) < n_negativos:
                    stats["chunks_sem_negativos_suficientes"] += len(pos_indices)
                    continue

                cand_indices_array = np.array([idx for idx, _ in candidatos_validos], dtype=np.int64)
                cand_scores_q_array = np.array([s for _, s in candidatos_validos], dtype=np.float32)
                cand_embs_mat = corpus_embs_cpu[cand_indices_array]  # Shape (K, D)

                for pos_idx in pos_indices:
                    ch_positivo = chunks_corpus[pos_idx]

                    # Filtro do Chunk 0: evita usar cabeçalho se houver outros chunks limpos
                    if (
                        ignorar_chunk_0_como_positivo
                        and ch_positivo.contem_vazamento_ementa
                        and len(pos_indices) > 1
                    ):
                        continue

                    stats["chunks_positivos_avaliados"] += 1

                    pos_emb_vec = corpus_embs_cpu[pos_idx]  # Shape (D,)

                    # Similaridade de cosseno com o positivo apenas sobre os candidatos: (K,)
                    sims_p_array = np.dot(cand_embs_mat, pos_emb_vec)

                    # Ordenacao pelos negativos mais desafiadores (prioriza maior similaridade com o positivo)
                    ordem = np.lexsort((cand_scores_q_array, sims_p_array))[::-1][:n_negativos]

                    if len(ordem) < n_negativos:
                        stats["chunks_sem_negativos_suficientes"] += 1
                        continue

                    neg_indices_finais = cand_indices_array[ordem]
                    neg_scores_finais = [round(float(s), 4) for s in cand_scores_q_array[ordem]]
                    neg_textos = [textos_corpus[idx] for idx in neg_indices_finais]
                    neg_ids = [chunks_corpus[idx].chunk_id for idx in neg_indices_finais]

                    # Calcula score exato da query com o positivo
                    score_pos = round(float(np.dot(q_emb_vec, pos_emb_vec)), 4)
                    margem = round(score_pos - max(neg_scores_finais), 4)

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
                        "scores_negativos": neg_scores_finais,
                        "margin": margem,
                    }

                    f_out.write(json.dumps(registro, ensure_ascii=False) + "\n")
                    stats["tripletos_salvos"] += 1
                    soma_margens += margem

            if device.type == "cuda":
                torch.cuda.empty_cache()

    if stats["tripletos_salvos"] > 0:
        stats["media_margem_tripletos"] = round(soma_margens / stats["tripletos_salvos"], 4)

    return stats
