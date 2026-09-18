#!/usr/bin/env python3
"""
Mineracao de Hard Negatives para Treinamento Contrastivo.

Objetivo:
1. Carregar a colecao de chunks (corpus_chunks.jsonl) e os pares filtrados (pares_train_filtrado.jsonl).
2. Gerar embeddings vetoriais normalizados para todo o corpus de chunks utilizando o modelo base.
3. Para cada ementa (query com instrucao especifica para o dominio juridico brasileiro):
   a. Calcular a similaridade de cosseno com todos os chunks do corpus via multiplicacao de matrizes na GPU.
   b. Excluir estritamente chunks pertencentes ao mesmo processo (evitar falsos negativos).
   c. Selecionar candidatos com similaridade de cosseno na faixa informatica [LIMIAR_MIN, LIMIAR_MAX].
   d. Reordenar os candidatos pela similaridade com o chunk positivo de referencia.
   e. Selecionar os 5 negativos mais desafiadores (hard negatives).
4. Persistir as instancias no formato de tripletos para fine-tuning contrastivo.

Entradas esperadas:
- corpus_chunks.jsonl: colecao de chunks gerada na Etapa 1.
- pares_train_filtrado.jsonl: pares ementa + metadados de processo gerados na Etapa 1.

Saida gerada:
- tripletos_train_chunks.jsonl contendo:
  - query: texto da ementa (com instrucao aplicada)
  - positive: texto do chunk do mesmo processo
  - negatives: lista com os textos dos 5 hard negatives selecionados
  - score_positive: similaridade da ementa com o chunk positivo
  - scores_negativos: lista com as similaridades dos negativos
  - margin: diferenca de score entre o positivo e o negativo mais similar
  - str_numero_unico: numero do processo
  - positive_chunk_id: ID do chunk positivo
  - negative_chunk_ids: IDs dos chunks negativos

Uso:
    python minerar_tripletos_chunks_train.py
"""

from __future__ import annotations

import gc
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


# Configuracao de caminhos
BASE_DIR = Path(__file__).resolve().parent

ARQUIVO_CHUNKS = BASE_DIR / "corpus_chunks.jsonl"
ARQUIVO_PARES = BASE_DIR / "pares_train_filtrado.jsonl"
ARQUIVO_SAIDA = BASE_DIR / "tripletos_train_chunks.jsonl"

# Modelo base para geracao dos embeddings de mineracao
MODEL_ID = "Qwen/Qwen3-Embedding-4B"

# Instrucao recomendada pela arquitetura Qwen3 para tarefas de retrieval
QUERY_INSTRUCTION = (
    "Instruct: Dado um trecho de um documento jurídico brasileiro, "
    "recupere o trecho mais relevante semanticamente.\nQuery: "
)
MAX_SEQ_LEN = 1024  # Suporta chunks de 768 tokens com margem de seguranca

# Parametros de selecao de hard negatives
N_NEGATIVOS = 5       # Quantidade de negativos minerados por instancia
LIMIAR_MIN = 0.50     # Similaridade minima com a query (evita negativos triviais)
LIMIAR_MAX = 0.90     # Similaridade maxima com a query (evita falsos negativos/casos identicos)
                      # Nota: No artigo publicado, a faixa de corte foi definida entre 0.60 e 0.85.
TOP_K = 200           # Quantidade de candidatos mais proximos da ementa a avaliar

# Parametros de processamento em lote para aceleracao em GPU
ENCODE_BATCH = 128    # Tamanho do lote para inferencia dos embeddings
QUERY_BATCH = 256     # Tamanho do lote de processos para calculo de similaridade em matriz


def ler_jsonl(caminho: Path) -> list[dict]:
    """Lê um arquivo JSONL e retorna uma lista de dicionarios."""
    rows = []
    with caminho.open(encoding="utf-8") as f:
        for num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSON invalido na linha {num} em {caminho}") from exc
    return rows


def carregar_corpus() -> tuple[list[dict], dict[str, list[int]], dict[str, str], list[str], list[str], dict]:
    """
    Carrega chunks e ementas, construindo os indices necessarios para acesso rapido.
    
    Returns:
        Tupla com (corpus_chunks, indices_por_processo, ementa_por_processo,
                   corpus_texts, processos, estatisticas_pulados).
    """
    if not ARQUIVO_CHUNKS.is_file():
        raise FileNotFoundError(f"Arquivo de chunks nao encontrado: {ARQUIVO_CHUNKS}. Execute a Etapa 1 primeiro.")
    if not ARQUIVO_PARES.is_file():
        raise FileNotFoundError(f"Arquivo de pares nao encontrado: {ARQUIVO_PARES}. Execute a Etapa 1 primeiro.")

    chunks_raw = ler_jsonl(ARQUIVO_CHUNKS)
    pares_raw = ler_jsonl(ARQUIVO_PARES)

    ementa_por_processo = {
        str(p["str_numero_unico"]).strip(): str(p["str_ementa"]).strip()
        for p in pares_raw
        if str(p.get("str_ementa", "")).strip()
    }

    corpus_chunks: list[dict] = []
    indices_por_processo: dict[str, list[int]] = defaultdict(list)
    pulados = {"sem_ementa": 0, "sem_texto": 0}

    for row in chunks_raw:
        numero = str(row.get("str_numero_unico", "")).strip()
        texto = str(row.get("texto", "")).strip()
        if not numero or not texto:
            pulados["sem_texto"] += 1
            continue
        if numero not in ementa_por_processo:
            pulados["sem_ementa"] += 1
            continue

        idx = len(corpus_chunks)
        corpus_chunks.append({
            "corpus_idx": idx,
            "chunk_id": str(row.get("chunk_id", f"{numero}__{row.get('chunk_idx', 0)}")).strip(),
            "str_numero_unico": numero,
            "chunk_idx": int(row.get("chunk_idx", 0)),
            "texto": texto,
        })
        indices_por_processo[numero].append(idx)

    processos = sorted(indices_por_processo.keys())
    corpus_texts = [c["texto"] for c in corpus_chunks]
    return corpus_chunks, indices_por_processo, ementa_por_processo, corpus_texts, processos, pulados


def carregar_modelo() -> SentenceTransformer:
    """
    Instancia o modelo Qwen3-Embedding-4B em precisao bfloat16 para mineracao.
    """
    print(f"Carregando modelo de mineracao: {MODEL_ID} ...")
    modelo = SentenceTransformer(
        MODEL_ID,
        model_kwargs={"torch_dtype": torch.bfloat16, "trust_remote_code": True},
        tokenizer_kwargs={"padding_side": "left"},
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    modelo.max_seq_length = MAX_SEQ_LEN
    print(f"Dispositivo de execucao: {modelo.device}")
    return modelo


def encodar_corpus(modelo: SentenceTransformer, corpus_texts: list[str]) -> np.ndarray:
    """
    Gera representacoes vetoriais normalizadas (norma L2 = 1.0) para todos os chunks.
    """
    print("Gerando representacoes vetoriais para todos os chunks do corpus ...")
    embeddings = modelo.encode(
        corpus_texts,
        batch_size=ENCODE_BATCH,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    print(f"Dimensoes da matriz de embeddings: {embeddings.shape}")
    return embeddings


def query_para_encode(texto: str) -> str:
    """
    Aplica o prefixo de instrucao formal na ementa (query).
    Documentos candidatos (positivos e negativos) nao recebem a instrucao.
    """
    texto = str(texto or "").strip()
    if texto.startswith(QUERY_INSTRUCTION):
        return texto
    return QUERY_INSTRUCTION + texto


def selecionar_negativos_por_chunk(
    sims_q: np.ndarray,
    sims_p: np.ndarray,
    indices_excluir: list[int],
    top_k: int,
) -> list[tuple[int, float]]:
    """
    Seleciona os 5 melhores hard negatives para um determinado chunk positivo.

    Estrategia de selecao:
    1. Mascara estritamente todos os chunks pertencentes ao mesmo processo judicial.
    2. Localiza os top_k chunks mais proximos semantica e vetorialmente da ementa.
    3. Descarta chunks fora da faixa [LIMIAR_MIN, LIMIAR_MAX].
    4. Reordena os sobreviventes pela similaridade direta com o chunk positivo.
    5. Retorna os N_NEGATIVOS mais similares ao positivo dentro da faixa permitida.
    """
    sims_q_m = sims_q.copy()
    sims_q_m[indices_excluir] = -np.inf
    ranked_q = np.argsort(sims_q_m)[::-1][:top_k]

    candidatos = []
    for idx in ranked_q:
        score_q = float(sims_q[idx])
        if score_q < LIMIAR_MIN or score_q > LIMIAR_MAX:
            continue
        candidatos.append((int(idx), score_q, float(sims_p[idx])))

    # Ordena prioritariamente pela proximidade com o positivo, seguido pelo score da query
    candidatos.sort(key=lambda x: (x[2], x[1]), reverse=True)
    return [(idx, score_q) for idx, score_q, _ in candidatos[:N_NEGATIVOS]]


def minerar_tripletos(
    modelo: SentenceTransformer,
    processos: list[str],
    ementa_por_processo: dict[str, str],
    indices_por_processo: dict[str, list[int]],
    corpus_chunks: list[dict],
    corpus_texts: list[str],
    corpus_embeddings: np.ndarray,
    f_out,
) -> dict:
    """
    Minera hard negatives em lote utilizando aceleracao em GPU via torch.matmul.
    Salva diretamente no arquivo de saida a cada processo concluido.
    """
    stats = {
        "processos": 0,
        "tripletos_salvos": 0,
        "chunks_sem_5_negs": 0,
    }

    # Transfere a matriz de embeddings do corpus para a memoria da GPU
    device = modelo.device
    print(f"Carregando matriz de embeddings na VRAM da GPU ({device}) para busca rapida...")
    corpus_embeddings_gpu = torch.from_numpy(corpus_embeddings).to(device=device, dtype=torch.float32)

    for start in tqdm(range(0, len(processos), QUERY_BATCH), desc="Minerando Tripletos"):
        batch_proc = processos[start : start + QUERY_BATCH]
        queries = [ementa_por_processo[p] for p in batch_proc]

        with torch.no_grad():
            # Vetorizacao do lote de consultas com instrucao
            q_embs_gpu = modelo.encode(
                [query_para_encode(q) for q in queries],
                batch_size=ENCODE_BATCH,
                normalize_embeddings=True,
                convert_to_tensor=True,
                show_progress_bar=False,
            ).to(dtype=torch.float32)

            # Produto escalar: cos(ementa, corpus) para todo o lote
            sims_batch_gpu = torch.matmul(q_embs_gpu, corpus_embeddings_gpu.T)
            sims_batch = sims_batch_gpu.cpu().numpy()

        for local_i, numero in enumerate(batch_proc):
            stats["processos"] += 1
            pos_indices = indices_por_processo[numero]
            sims_q = sims_batch[local_i]

            # Produto escalar: cos(chunk_positivo, corpus)
            pos_embs_gpu = corpus_embeddings_gpu[pos_indices]
            sims_p_gpu = torch.matmul(pos_embs_gpu, corpus_embeddings_gpu.T)
            sims_p_por_chunk = sims_p_gpu.cpu().numpy()

            query = queries[local_i]

            for j, pos_idx in enumerate(pos_indices):
                negs = selecionar_negativos_por_chunk(
                    sims_q,
                    sims_p_por_chunk[j],
                    pos_indices,
                    TOP_K,
                )
                if len(negs) < N_NEGATIVOS:
                    stats["chunks_sem_5_negs"] += 1
                    continue

                ch = corpus_chunks[pos_idx]
                neg_textos = [corpus_texts[i] for i, _ in negs]
                neg_scores = [round(s, 4) for _, s in negs]
                neg_ids = [corpus_chunks[i]["chunk_id"] for i, _ in negs]
                score_pos = round(float(sims_q[pos_idx]), 4)
                margin = round(score_pos - max(neg_scores), 4)

                registro = {
                    "query": query,
                    "positive": ch["texto"],
                    "negatives": neg_textos,
                    "score_positive": score_pos,
                    "scores_negativos": neg_scores,
                    "margin": margin,
                    "str_numero_unico": numero,
                    "positive_chunk_id": ch["chunk_id"],
                    "positive_chunk_idx": ch["chunk_idx"],
                    "negative_chunk_ids": neg_ids,
                }
                f_out.write(json.dumps(registro, ensure_ascii=False) + "\n")
                stats["tripletos_salvos"] += 1

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return stats


def inspecionar_saida(n: int = 5) -> None:
    """Imprime os primeiros registros gerados para auditoria visual imediata."""
    amostra = ler_jsonl(ARQUIVO_SAIDA)[:n]
    if not amostra:
        print("Nenhum tripleto gerado. Verifique os limiares ou o parametro TOP_K.")
        return

    print("\nAuditoria Rapida dos Primeiros Tripletos:")
    for i, r in enumerate(amostra):
        print(
            f"[{i}] Chunk Positivo: {r['positive_chunk_id']} | "
            f"Score Positivo: {r['score_positive']} | Margem: {r['margin']} | "
            f"Scores Negativos: {r['scores_negativos']}"
        )


def main() -> None:
    """Execucao principal da mineracao de hard negatives."""
    ARQUIVO_SAIDA.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 64)
    print("BumbaLM - Etapa 2: Mineracao de Hard Negatives em GPU")
    print("=" * 64)
    print(f"Corpus de Chunks      : {ARQUIVO_CHUNKS}")
    print(f"Pares de Processos    : {ARQUIVO_PARES}")
    print(f"Arquivo de Saida      : {ARQUIVO_SAIDA}")
    print(f"Limiares Negativos    : [{LIMIAR_MIN}, {LIMIAR_MAX}] | TOP_K={TOP_K}")
    print(f"Negativos por Chunk   : {N_NEGATIVOS}")
    print("-" * 64)

    # 1. Carregamento dos dados
    (
        corpus_chunks,
        indices_por_processo,
        ementa_por_processo,
        corpus_texts,
        processos,
        pulados,
    ) = carregar_corpus()

    n_tasks = sum(len(indices_por_processo[p]) for p in processos)
    print(f"Total de chunks no corpus : {len(corpus_chunks):,}")
    print(f"Total de processos        : {len(processos):,}")
    print(f"Total maximo de tripletos : {n_tasks:,} (1 por chunk)")
    print(f"Registros pulados         : {pulados}")

    if not processos:
        raise ValueError("Nenhum processo valido encontrado para mineracao.")

    # 2. Inicializacao do modelo e calculo dos embeddings do corpus
    modelo = carregar_modelo()
    corpus_embeddings = encodar_corpus(modelo, corpus_texts)

    # 3. Limpeza de memoria e mineracao dos tripletos
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"Iniciando mineracao de ate {n_tasks:,} tripletos...")
    with ARQUIVO_SAIDA.open("w", encoding="utf-8") as f_out:
        stats = minerar_tripletos(
            modelo,
            processos,
            ementa_por_processo,
            indices_por_processo,
            corpus_chunks,
            corpus_texts,
            corpus_embeddings,
            f_out,
        )

    print("-" * 64)
    print("Resumo da Mineracao:")
    print(f"  - Processos avaliados      : {stats['processos']:,}")
    print(f"  - Tripletos validos salvos : {stats['tripletos_salvos']:,}")
    print(f"  - Chunks sem 5 negativos   : {stats['chunks_sem_5_negs']:,}")
    print(f"Arquivo final persistido em  : {ARQUIVO_SAIDA}")
    print("=" * 64)

    inspecionar_saida()


if __name__ == "__main__":
    main()