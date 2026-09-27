"""
Modulo de avaliacao de Information Retrieval (IR) in-domain para acordaos judiciais.

Constroi o InformationRetrievalEvaluator da biblioteca SentenceTransformers garantindo:
1. Queries formadas pelas ementas dos processos de teste/validacao com instrucao formal.
2. Corpus formado exclusivamente por chunks de inteiro teor limpos (descontaminados).
3. Exclusao rigorosa de chunks com vazamento de ementa (is_ementa_leak=True) do corpus
   e dos documentos relevantes para eliminar atalhos sintaticos triviais.
4. Calculo de metricas consolidadas: nDCG@10, MRR@10, MAP@100 e Recall/Accuracy@1, 5, 10.
5. Suporte a avaliacao global e segmentada por tribunal.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional, Union

try:
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.evaluation import InformationRetrievalEvaluator
    from sentence_transformers.util import cos_sim
except ImportError:
    SentenceTransformer = Any  # type: ignore
    InformationRetrievalEvaluator = Any  # type: ignore
    cos_sim = None

from src.data.schema import ChunkProcesso, ProcessoJudicial

log = logging.getLogger("bumbalm_ir_evaluator")


def carregar_processos_jsonl(caminho: Path | str) -> list[ProcessoJudicial]:
    """Le processos judiciais gravados em arquivo JSONL."""
    caminho_path = Path(caminho)
    if not caminho_path.is_file():
        raise FileNotFoundError(f"Arquivo de processos nao encontrado: {caminho_path}")

    processos = []
    with caminho_path.open(encoding="utf-8") as f:
        for linha in f:
            linha = linha.strip()
            if not linha:
                continue
            item = json.loads(linha)
            processos.append(ProcessoJudicial.from_dict(item))
    return processos


def carregar_chunks_jsonl(caminho: Path | str) -> list[ChunkProcesso]:
    """Le chunks de processos gravados em arquivo JSONL com flag de contaminacao."""
    caminho_path = Path(caminho)
    if not caminho_path.is_file():
        raise FileNotFoundError(f"Arquivo de chunks nao encontrado: {caminho_path}")

    chunks = []
    with caminho_path.open(encoding="utf-8") as f:
        for linha in f:
            linha = linha.strip()
            if not linha:
                continue
            item = json.loads(linha)
            chunks.append(ChunkProcesso.from_dict(item))
    return chunks


def montar_estruturas_ir(
    processos: list[ProcessoJudicial],
    chunks: list[ChunkProcesso],
    query_instruction: str = "Instruct: Dado um trecho de um documento jurídico brasileiro, recupere o trecho mais relevante semanticamente.\nQuery: ",
    excluir_chunks_contaminados: bool = True,
    limite_queries: Optional[int] = None,
    filtro_tribunal: Optional[str] = None,
) -> tuple[dict[str, str], dict[str, str], dict[str, set[str]]]:
    """
    Monta as estruturas de dados fundamentais para o InformationRetrievalEvaluator:
    - queries: dict[qid, query_text]
    - corpus: dict[cid, chunk_text]
    - relevant_docs: dict[qid, set[cid]]

    Garante descontaminacao metodologica:
    - Se excluir_chunks_contaminados=True, chunks marcados com is_ementa_leak=True
      sao integralmente descartados do corpus e da lista de relevantes.
    - Processos que nao possuirem nenhum chunk limpo remanescente sao omitidos do benchmark.
    """
    # 1. Filtra tribunal se especificado
    if filtro_tribunal:
        tribunal_alvo = filtro_tribunal.upper().strip()
        processos = [p for p in processos if p.tribunal.upper().strip() == tribunal_alvo]
        chunks = [c for c in chunks if c.tribunal.upper().strip() == tribunal_alvo]

    # 2. Construtor do corpus e mapeamento processo -> chunks validos
    corpus: dict[str, str] = {}
    processo_para_chunk_ids: dict[str, set[str]] = defaultdict(set)
    chunks_descartados_vazamento = 0

    for ch in chunks:
        if excluir_chunks_contaminados and ch.is_ementa_leak:
            chunks_descartados_vazamento += 1
            continue

        cid = ch.chunk_id
        corpus[cid] = ch.texto
        processo_para_chunk_ids[ch.processo_id].add(cid)

    # 3. Construtor de queries e mapa de relevancias
    queries: dict[str, str] = {}
    relevant_docs: dict[str, set[str]] = {}
    processos_sem_chunk_limpo = 0

    for proc in processos:
        pid = proc.id
        chunks_relevantes = processo_para_chunk_ids.get(pid, set())

        if not chunks_relevantes:
            processos_sem_chunk_limpo += 1
            continue

        ementa_limpa = proc.ementa.strip()
        if not ementa_limpa:
            continue

        qid = f"query_{pid}"
        query_formatada = (
            ementa_limpa
            if ementa_limpa.startswith("Instruct:")
            else f"{query_instruction}{ementa_limpa}"
        )

        queries[qid] = query_formatada
        relevant_docs[qid] = set(chunks_relevantes)

        if limite_queries is not None and len(queries) >= limite_queries:
            break

    log.info(
        "Estruturas IR montadas: %d queries | %d docs no corpus | %d descartados por vazamento | %d procs sem chunk limpo",
        len(queries),
        len(corpus),
        chunks_descartados_vazamento,
        processos_sem_chunk_limpo,
    )

    return queries, corpus, relevant_docs


def criar_ir_evaluator(
    caminho_processos: Path | str,
    caminho_chunks: Path | str,
    nome: str = "juridico",
    query_instruction: str = "Instruct: Dado um trecho de um documento jurídico brasileiro, recupere o trecho mais relevante semanticamente.\nQuery: ",
    excluir_chunks_contaminados: bool = True,
    limite_queries: Optional[int] = None,
    filtro_tribunal: Optional[str] = None,
    batch_size: int = 64,
    show_progress_bar: bool = True,
) -> InformationRetrievalEvaluator:
    """
    Instancia e retorna o InformationRetrievalEvaluator configurado para o dominio juridico.
    """
    if cos_sim is None:
        raise ImportError(
            "A biblioteca sentence_transformers e necessaria para criar o InformationRetrievalEvaluator. "
            "Instale via 'pip install sentence-transformers'."
        )

    processos = carregar_processos_jsonl(caminho_processos)
    chunks = carregar_chunks_jsonl(caminho_chunks)

    queries, corpus, relevant_docs = montar_estruturas_ir(
        processos=processos,
        chunks=chunks,
        query_instruction=query_instruction,
        excluir_chunks_contaminados=excluir_chunks_contaminados,
        limite_queries=limite_queries,
        filtro_tribunal=filtro_tribunal,
    )

    if not queries:
        raise ValueError(
            f"Nenhuma query valida encontrada para o avaliador '{nome}'. "
            f"Verifique os arquivos {caminho_processos} e {caminho_chunks}."
        )

    if not corpus:
        raise ValueError(f"Corpus de chunks vazio para o avaliador '{nome}'.")

    evaluator = InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        name=nome,
        batch_size=batch_size,
        show_progress_bar=show_progress_bar,
        score_functions={"cosine": cos_sim},
        main_score_function="cosine",
        mrr_at_k=[10],
        ndcg_at_k=[10],
        accuracy_at_k=[1, 5, 10],
        precision_recall_at_k=[1, 5, 10],
        map_at_k=[100],
    )

    return evaluator


def formatar_metricas_ir(metricas: dict[str, float], prefixo_nome: str = "") -> dict[str, float]:
    """
    Normaliza os nomes das metricas retornadas pelo evaluator para um formato canonico.
    Exemplo: 'val_ir_eval_cosine_ndcg@10' -> 'ndcg@10'
    """
    metricas_limpas: dict[str, float] = {}
    for chave, valor in metricas.items():
        chave_normalizada = chave
        if prefixo_nome and chave_normalizada.startswith(f"{prefixo_nome}_"):
            chave_normalizada = chave_normalizada[len(prefixo_nome) + 1 :]
        if chave_normalizada.startswith("cosine_"):
            chave_normalizada = chave_normalizada[len("cosine_") :]
        metricas_limpas[chave_normalizada] = float(valor)

    # Chaves prioritarias para exibicao
    chaves_destaque = ["ndcg@10", "mrr@10", "map@100", "accuracy@1", "accuracy@5", "accuracy@10"]
    resultado = {}
    for k in chaves_destaque:
        if k in metricas_limpas:
            resultado[k] = metricas_limpas[k]
    # Adiciona restantes
    for k, v in sorted(metricas_limpas.items()):
        if k not in resultado:
            resultado[k] = v

    return resultado


def executar_avaliacao_completa(
    model: Union[SentenceTransformer, str, Path],
    caminho_processos: Path | str,
    caminho_chunks: Path | str,
    output_dir: Optional[Path | str] = None,
    nome_avaliacao: str = "test_in_domain",
    query_instruction: str = "Instruct: Dado um trecho de um documento jurídico brasileiro, recupere o trecho mais relevante semanticamente.\nQuery: ",
    excluir_chunks_contaminados: bool = True,
    batch_size: int = 64,
    avaliar_por_tribunal: bool = True,
) -> dict[str, Any]:
    """
    Executa a avaliacao de IR de ponta a ponta:
    1. Avaliacao global no conjunto completo.
    2. Avaliacoes segmentadas por tribunal (se habilitado).
    3. Gravacao de relatorio em texto formatado e arquivo JSON com as metricas.
    """
    if isinstance(model, (str, Path)):
        log.info("Carregando modelo SentenceTransformer a partir de: %s", model)
        modelo_instanciado = SentenceTransformer(str(model), trust_remote_code=True)
    else:
        modelo_instanciado = model

    dir_saida = Path(output_dir) if output_dir else None
    if dir_saida:
        dir_saida.mkdir(parents=True, exist_ok=True)

    relatorio_geral: dict[str, Any] = {
        "nome_avaliacao": nome_avaliacao,
        "descontaminacao_chunk0": excluir_chunks_contaminados,
        "metricas_global": {},
        "metricas_por_tribunal": {},
    }

    # 1. Avaliacao Global
    print(f"\nIniciando avaliacao global ({nome_avaliacao})...")
    evaluator_global = criar_ir_evaluator(
        caminho_processos=caminho_processos,
        caminho_chunks=caminho_chunks,
        nome=f"{nome_avaliacao}_global",
        query_instruction=query_instruction,
        excluir_chunks_contaminados=excluir_chunks_contaminados,
        batch_size=batch_size,
    )

    metricas_brutas_global = evaluator_global(
        modelo_instanciado,
        output_path=str(dir_saida) if dir_saida else None,
    )
    metricas_global = formatar_metricas_ir(metricas_brutas_global, prefixo_nome=f"{nome_avaliacao}_global")
    relatorio_geral["metricas_global"] = metricas_global

    # 2. Avaliacao Segmentada por Tribunal
    if avaliar_por_tribunal:
        processos_todos = carregar_processos_jsonl(caminho_processos)
        tribunais_disponiveis = sorted(list({p.tribunal.strip() for p in processos_todos if p.tribunal.strip()}))

        if len(tribunais_disponiveis) > 1:
            print(f"\nExecutando avaliacoes segmentadas por tribunal: {', '.join(tribunais_disponiveis)} ...")
            for trib in tribunais_disponiveis:
                try:
                    evaluator_trib = criar_ir_evaluator(
                        caminho_processos=caminho_processos,
                        caminho_chunks=caminho_chunks,
                        nome=f"{nome_avaliacao}_{trib}",
                        query_instruction=query_instruction,
                        excluir_chunks_contaminados=excluir_chunks_contaminados,
                        filtro_tribunal=trib,
                        batch_size=batch_size,
                    )
                    metricas_brutas_trib = evaluator_trib(modelo_instanciado)
                    metricas_trib = formatar_metricas_ir(metricas_brutas_trib, prefixo_nome=f"{nome_avaliacao}_{trib}")
                    relatorio_geral["metricas_por_tribunal"][trib] = metricas_trib
                except ValueError as e:
                    log.warning("Nao foi possivel avaliar isoladamente tribunal %s: %s", trib, e)

    # 3. Salvar relatorios
    if dir_saida:
        # JSON
        caminho_json = dir_saida / f"metricas_{nome_avaliacao}.json"
        with caminho_json.open("w", encoding="utf-8") as f:
            json.dump(relatorio_geral, f, indent=2, ensure_ascii=False)

        # TXT Relatorio Tabular
        caminho_txt = dir_saida / f"relatorio_{nome_avaliacao}.txt"
        with caminho_txt.open("w", encoding="utf-8") as f:
            f.write("=" * 64 + "\n")
            f.write(f"RELATORIO DE AVALIACAO IN-DOMAIN - {nome_avaliacao.upper()}\n")
            f.write("=" * 64 + "\n\n")
            f.write(f"Descontaminacao de Chunk 0 : {excluir_chunks_contaminados}\n")
            f.write(f"Instrucao de Query         : {query_instruction.strip()}\n\n")

            f.write("--- RESULTADOS GLOBAIS ---\n")
            for met, val in metricas_global.items():
                f.write(f"  {met:<20}: {val:.4f}\n")
            f.write("\n")

            if relatorio_geral["metricas_por_tribunal"]:
                f.write("--- RESULTADOS POR TRIBUNAL ---\n")
                for trib, m_trib in relatorio_geral["metricas_por_tribunal"].items():
                    f.write(f"\n[Tribunal: {trib}]\n")
                    for met, val in m_trib.items():
                        f.write(f"  {met:<20}: {val:.4f}\n")
            f.write("\n" + "=" * 64 + "\n")

        print(f"\nRelatorio salvo em: {caminho_txt}")
        print(f"Metricas em JSON salvas em: {caminho_json}")

    return relatorio_geral
