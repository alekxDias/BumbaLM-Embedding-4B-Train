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
import math
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
    limite_processos: Optional[int] = None,
    seed: int = 42,
    relevancia_por_grupo: bool = True,
    estatisticas: Optional[dict[str, Any]] = None,
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

    relevancia_por_grupo: os chunks limpos dos processos do mesmo grupo de quase duplicatas
    (ementas acima do limiar de Jaccard, mesmo split) tambem contam como relevantes. Sem isso, achar o
    acordao "gemeo" contaria como erro.

    estatisticas: se for um dict, recebe as contagens de exclusao para o relatorio.

    limite_processos sorteia processos com a semente e mantem no corpus so os chunks deles.
    Serve para a validacao durante o treino. O corpus fica menor, entao as metricas sobem
    em relacao ao split completo: compare variantes sempre com a mesma amostra.
    """
    # 1. Filtra tribunal se especificado
    if filtro_tribunal:
        tribunal_alvo = filtro_tribunal.upper().strip()
        processos = [p for p in processos if p.tribunal.upper().strip() == tribunal_alvo]
        chunks = [c for c in chunks if c.tribunal.upper().strip() == tribunal_alvo]

    # 1b. Amostra fixa de processos. Ordena antes para o sorteio nao depender da ordem do arquivo.
    if limite_processos and len(processos) > limite_processos:
        import random

        ordenados = sorted(processos, key=lambda p: p.id)
        processos = random.Random(seed).sample(ordenados, int(limite_processos))
        ids_amostra = {p.id for p in processos}
        chunks = [c for c in chunks if c.processo_id in ids_amostra]

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

    # Chunks limpos de cada grupo de quase duplicatas presente neste split
    chunks_por_grupo: dict[str, set[str]] = defaultdict(set)
    for proc in processos:
        chunks_por_grupo[proc.grupo_duplicata] |= processo_para_chunk_ids.get(proc.id, set())

    # 3. Construtor de queries e mapa de relevancias
    queries: dict[str, str] = {}
    relevant_docs: dict[str, set[str]] = {}
    processos_sem_chunk_limpo = 0
    queries_com_gemeos = 0

    for proc in processos:
        pid = proc.id
        proprios = processo_para_chunk_ids.get(pid, set())

        # Sem chunk limpo proprio a consulta sai, mesmo que o gemeo tenha: a ementa
        # copiada no cabecalho era a unica "resposta" desse acordao.
        if not proprios:
            processos_sem_chunk_limpo += 1
            continue

        chunks_relevantes = proprios
        if relevancia_por_grupo:
            chunks_relevantes = chunks_por_grupo[proc.grupo_duplicata]
            if len(chunks_relevantes) > len(proprios):
                queries_com_gemeos += 1

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
    if estatisticas is not None:
        estatisticas.update({
            "processos_avaliados": len(processos),
            "queries": len(queries),
            "chunks_no_corpus": len(corpus),
            "chunks_descartados_vazamento": chunks_descartados_vazamento,
            "processos_sem_chunk_limpo": processos_sem_chunk_limpo,
            "queries_com_quase_duplicatas_relevantes": queries_com_gemeos,
            "relevancia_por_grupo": relevancia_por_grupo,
        })

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
    limite_processos: Optional[int] = None,
    seed: int = 42,
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

    estatisticas: dict[str, Any] = {}
    queries, corpus, relevant_docs = montar_estruturas_ir(
        processos=processos,
        chunks=chunks,
        query_instruction=query_instruction,
        excluir_chunks_contaminados=excluir_chunks_contaminados,
        limite_queries=limite_queries,
        filtro_tribunal=filtro_tribunal,
        limite_processos=limite_processos,
        seed=seed,
        estatisticas=estatisticas,
    )
    print(
        f"Avaliador '{nome}': {len(queries):,} queries | {len(corpus):,} chunks no corpus"
        + (f" (amostra de {limite_processos:,} processos)" if limite_processos else "")
    )
    print(
        f"  - Fora da avaliacao: {estatisticas['chunks_descartados_vazamento']:,} chunks com vazamento | "
        f"{estatisticas['processos_sem_chunk_limpo']:,} processos sem chunk limpo | "
        f"{estatisticas['queries_com_quase_duplicatas_relevantes']:,} consultas com quase duplicatas relevantes"
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
    # Lido por executar_avaliacao_completa para o relatorio.
    evaluator.estatisticas_bumbalm = estatisticas

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


def _capturar_resultados_por_consulta(evaluator: Any) -> list[list[dict[str, Any]]]:
    """
    Faz o evaluator guardar a lista de resultados (top-k de cada consulta) que ele ja calcula
    internamente antes de tirar as medias. Nao recalcula nada: so copia o que passa por
    compute_metrics. A lista devolvida e preenchida quando o evaluator roda.
    """
    capturados: list[list[dict[str, Any]]] = []
    original = evaluator.compute_metrics

    def compute_metrics_com_captura(queries_result_list, *args, **kwargs):
        capturados.append(queries_result_list)
        return original(queries_result_list, *args, **kwargs)

    evaluator.compute_metrics = compute_metrics_com_captura
    return capturados


def _dcg(relevancias: list[int], k: int) -> float:
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(relevancias[:k]))


def calcular_metricas_por_consulta(evaluator: Any, queries_result_list: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """
    Metricas de cada consulta com as mesmas formulas do InformationRetrievalEvaluator
    (a media destas linhas reproduz o resultado global). Servem para testes pareados
    (bootstrap, Wilcoxon) entre modelos: o query_id e o mesmo em todos os modelos.
    """
    linhas: list[dict[str, Any]] = []
    for indice, resultados in enumerate(queries_result_list):
        query_id = evaluator.queries_ids[indice]
        relevantes = evaluator.relevant_docs[query_id]
        ordenados = sorted(resultados, key=lambda x: x["score"], reverse=True)
        acertos = [1 if hit["corpus_id"] in relevantes else 0 for hit in ordenados]

        posicao = next((i + 1 for i, a in enumerate(acertos) if a), None)
        linha: dict[str, Any] = {
            "query_id": query_id,
            "n_relevantes": len(relevantes),
            # 1 = primeiro lugar. None = nenhum relevante entre os resultados guardados (top 100).
            "posicao_primeiro_relevante": posicao,
        }
        for k in (1, 5, 10):
            linha[f"accuracy@{k}"] = 1.0 if any(acertos[:k]) else 0.0
        linha["mrr@10"] = 1.0 / posicao if posicao is not None and posicao <= 10 else 0.0
        linha["ndcg@10"] = _dcg(acertos, 10) / _dcg([1] * len(relevantes), 10)
        linha["precision@10"] = sum(acertos[:10]) / 10
        linha["recall@10"] = sum(acertos[:10]) / len(relevantes)

        soma_precisoes = 0.0
        corretos = 0
        for i, a in enumerate(acertos[:100]):
            if a:
                corretos += 1
                soma_precisoes += corretos / (i + 1)
        linha["map@100"] = soma_precisoes / min(100, len(relevantes))

        # Os 10 primeiros, para analise de erros sem precisar rodar o modelo de novo
        linha["top10"] = [[hit["corpus_id"], round(float(hit["score"]), 5)] for hit in ordenados[:10]]
        linhas.append(linha)
    return linhas


def gravar_resultados_por_consulta(
    linhas: list[dict[str, Any]],
    metricas_global: dict[str, float],
    caminho: Path,
) -> None:
    """Grava uma linha JSON por consulta e confere que as medias batem com as do evaluator."""
    with caminho.open("w", encoding="utf-8") as f:
        for linha in linhas:
            f.write(json.dumps(linha, ensure_ascii=False) + "\n")

    maior_diferenca = 0.0
    for metrica in ("mrr@10", "ndcg@10", "accuracy@1", "accuracy@10", "map@100"):
        if metrica in metricas_global and linhas:
            media = sum(l[metrica] for l in linhas) / len(linhas)
            maior_diferenca = max(maior_diferenca, abs(media - metricas_global[metrica]))
    print(f"Resultados por consulta salvos em: {caminho} ({len(linhas):,} consultas)")
    if maior_diferenca > 1e-4:
        print(
            f"  - AVISO: as medias por consulta diferem do resultado global em ate {maior_diferenca:.5f}. "
            "Confira a versao do sentence-transformers antes de usar este arquivo."
        )
    else:
        print(f"  - Conferencia: medias por consulta iguais as globais (diferenca maxima {maior_diferenca:.1e})")


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
    gravar_por_consulta: bool = True,
) -> dict[str, Any]:
    """
    Executa a avaliacao de IR de ponta a ponta:
    1. Avaliacao global no conjunto completo. Com gravar_por_consulta, grava tambem
       por_consulta_<nome_avaliacao>.jsonl (uma linha por ementa) para testes de significancia.
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
    capturados = _capturar_resultados_por_consulta(evaluator_global) if gravar_por_consulta and dir_saida else None

    metricas_brutas_global = evaluator_global(
        modelo_instanciado,
        output_path=str(dir_saida) if dir_saida else None,
    )
    metricas_global = formatar_metricas_ir(metricas_brutas_global, prefixo_nome=f"{nome_avaliacao}_global")
    relatorio_geral["metricas_global"] = metricas_global
    relatorio_geral["protocolo"] = getattr(evaluator_global, "estatisticas_bumbalm", {})

    # 1b. Resultado de cada consulta. Uma falha aqui nao derruba a avaliacao: as medias ja estao prontas.
    if capturados is not None:
        try:
            if not capturados:
                raise RuntimeError("o evaluator nao passou por compute_metrics")
            linhas = calcular_metricas_por_consulta(evaluator_global, capturados[-1])
            caminho_consultas = dir_saida / f"por_consulta_{nome_avaliacao}.jsonl"
            gravar_resultados_por_consulta(linhas, metricas_global, caminho_consultas)
            relatorio_geral["arquivo_por_consulta"] = str(caminho_consultas)
        except Exception as exc:  # noqa: BLE001
            log.warning("Nao foi possivel gravar os resultados por consulta: %s", exc)
            print(f"  - AVISO: resultados por consulta NAO gravados ({exc}). As medias continuam validas.")

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
            for chave, valor in relatorio_geral["protocolo"].items():
                f.write(f"{chave:<40}: {valor}\n")
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
