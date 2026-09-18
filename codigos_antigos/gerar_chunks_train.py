#!/usr/bin/env python3
"""
Filtragem de Pares e Chunking por Tokens.

Este script faz parte do pipeline original do BumbaLM-Embedding-4B
(artigo: "Mining Hard Negatives for Contrastive Learning of Brazilian Legal Embeddings").

Objetivo:
1. Ler os registros brutos de acórdãos do TJMA (pares ementa + inteiro teor).
2. Aplicar filtros de qualidade e unicidade para descartar ruídos e duplicatas.
3. Fragmentar apenas o 'str_inteiro_teor' em janelas contíguas de 768 tokens
   (com sobreposição de 128 tokens) utilizando o tokenizador do Qwen3-Embedding-4B.
4. Manter a 'str_ementa' intacta como query/âncora para recuperação semântica.
5. Salvar a coleção de chunks identificados de forma única por processo.

Entradas esperadas:
- pares_60k.jsonl (ou pares_completo.jsonl) contendo:
  - str_numero_unico: identificador CNJ do processo
  - str_ementa: texto da ementa (resumo jurídico do caso)
  - str_inteiro_teor: decisão judicial na íntegra

Saídas geradas:
- pares_train_filtrado.jsonl: pares limpos e deduplicados por número do processo.
- corpus_chunks.jsonl: todos os chunks gerados a partir do inteiro teor.

Uso:
    python gerar_chunks_train.py
"""

from __future__ import annotations

import json
import re
import statistics
from collections import Counter
from pathlib import Path

from tqdm import tqdm
from transformers import AutoTokenizer


# Configuração de caminhos e parâmetros
BASE_DIR = Path(__file__).resolve().parent

ARQUIVO_TRAIN = BASE_DIR / "pares_60k.jsonl"
ARQUIVO_FILTRADO = BASE_DIR / "pares_train_filtrado.jsonl"
ARQUIVO_CHUNKS = BASE_DIR / "corpus_chunks.jsonl"

# Modelo base de referência - apenas o tokenizer é carregado (sem pesos do modelo)
MODEL_ID = "Qwen/Qwen3-Embedding-4B"

# Parâmetros de janelamento deslizante (em tokens do vocabulário Qwen)
CHUNK_SIZE_TOKENS = 768       # Tamanho máximo de cada trecho do documento
CHUNK_OVERLAP_TOKENS = 128    # Tokens compartilhados entre chunks adjacentes para preservar contexto

# Limiares de corte de qualidade dos textos brutos
MIN_CHARS_EMENTA = 50         # Ementas menores que 50 caracteres são descartadas (ruído/metadado)
MIN_CHARS_TEOR = 200          # Inteiro teor com menos de 200 caracteres não possui fundamentação válida


def ler_jsonl(caminho: Path) -> list[dict]:
    """
    Lê um arquivo no formato JSONL (JSON Lines) e retorna uma lista de dicionários.
    
    Args:
        caminho: Objeto Path indicando o arquivo a ser lido.
        
    Returns:
        Lista contendo cada registro decodificado como dict.
    """
    registros = []
    with caminho.open(encoding="utf-8") as f:
        for num_linha, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSON invalido na linha {num_linha} de {caminho}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"Linha {num_linha} nao e um objeto JSON valido.")
            registros.append(obj)
    return registros


def escrever_jsonl(caminho: Path, registros: list[dict]) -> None:
    """
    Escreve uma lista de dicionários em arquivo JSONL, garantindo UTF-8 sem escape ASCII.
    
    Args:
        caminho: Objeto Path indicando o arquivo de destino.
        registros: Lista de dicts a serem persistidos.
    """
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with caminho.open("w", encoding="utf-8") as f:
        for reg in registros:
            f.write(json.dumps(reg, ensure_ascii=False) + "\n")


def normalizar_numero(numero: str) -> str:
    """
    Remove pontuação e caracteres não numéricos do identificador CNJ do processo.
    Exemplo: '0800123-45.2020.8.10.0001' -> '08001234520208100001'.
    
    Args:
        numero: String contendo o número do processo formatado.
        
    Returns:
        Apenas os dígitos correspondentes.
    """
    return re.sub(r"\D+", "", str(numero or "").strip())


def filtrar_pares(registros: list[dict]) -> tuple[list[dict], dict]:
    """
    Filtra os pares brutos garantindo integridade e ausência de duplicatas de processo.

    Critérios de descarte:
      - sem_numero_unico: registro sem campo identificador
      - numero_unico_duplicado: processo já visto anteriormente na base
      - campo_vazio: ausência de ementa ou de inteiro teor
      - ementa_curta: ementa com comprimento inferior a MIN_CHARS_EMENTA
      - teor_curto: inteiro teor com comprimento inferior a MIN_CHARS_TEOR
      - teor_menor_que_ementa: inconsistência em que o resumo é mais longo que o acórdão

    Args:
        registros: Lista de registros brutos lidos do JSONL.

    Returns:
        Tupla contendo (lista_de_pares_validos, dicionario_de_estatisticas).
    """
    validos = []
    motivos = Counter()
    vistos = set()

    for reg in registros:
        numero = str(reg.get("str_numero_unico", "")).strip()
        ementa = str(reg.get("str_ementa", "")).strip()
        teor = str(reg.get("str_inteiro_teor", "")).strip()
        numero_norm = normalizar_numero(numero)

        if not numero_norm:
            motivos["sem_numero_unico"] += 1
            continue
        if numero_norm in vistos:
            motivos["numero_unico_duplicado"] += 1
            continue
        if not ementa or not teor:
            motivos["campo_vazio"] += 1
            continue
        if len(ementa) < MIN_CHARS_EMENTA:
            motivos["ementa_curta"] += 1
            continue
        if len(teor) < MIN_CHARS_TEOR:
            motivos["teor_curto"] += 1
            continue
        if len(teor) < len(ementa):
            motivos["teor_menor_que_ementa"] += 1
            continue

        vistos.add(numero_norm)
        validos.append({
            "str_numero_unico": numero,
            "str_ementa": ementa,
            "str_inteiro_teor": teor,
        })

    stats = {
        "total_lido": len(registros),
        "validos": len(validos),
        "descartados": sum(motivos.values()),
        "motivos_descarte": dict(motivos),
    }
    return validos, stats


def chunkar_texto(
    texto: str,
    tokenizer: AutoTokenizer,
    chunk_size: int,
    overlap: int,
) -> list[dict]:
    """
    Divide o texto do inteiro teor em janelas contíguas de tamanho fixo em tokens,
    utilizando sobreposição (stride = chunk_size - overlap).

    Nota sobre o Chunk 0:
    O primeiro trecho (chunk_idx = 0) frequentemente contém o cabeçalho e a própria
    ementa transcrita no texto original do acórdão. Essa característica requer
    auditoria específica na etapa de avaliação para evitar vazamento de dados.

    Args:
        texto: Inteiro teor a ser segmentado.
        tokenizer: Tokenizador do modelo base (Qwen3).
        chunk_size: Quantidade máxima de tokens por fragmento.
        overlap: Quantidade de tokens compartilhados entre trechos consecutivos.

    Returns:
        Lista de dicionários com metadados de cada chunk gerado.
    """
    ids = tokenizer.encode(texto, add_special_tokens=False)
    if not ids:
        return []

    if overlap >= chunk_size:
        raise ValueError("O overlap deve ser estritamente menor que chunk_size.")

    stride = chunk_size - overlap
    chunks = []
    start = 0
    idx = 0

    while start < len(ids):
        end = min(start + chunk_size, len(ids))
        piece_ids = ids[start:end]
        piece_text = tokenizer.decode(piece_ids, skip_special_tokens=True).strip()
        
        if piece_text:
            chunks.append({
                "chunk_idx": idx,
                "token_start": start,
                "token_end": end,
                "n_tokens": len(piece_ids),
                "texto": piece_text,
            })
            idx += 1
            
        if end >= len(ids):
            break
        start += stride

    return chunks


def gerar_corpus_chunks(
    pares_validos: list[dict],
    tokenizer: AutoTokenizer,
) -> tuple[list[dict], dict]:
    """
    Itera sobre os pares validados, aplica a fragmentação do inteiro teor
    e monta a coleção completa de chunks do corpus.

    Cada chunk produzido recebe a chave identificadora:
    chunk_id = "{numero_processo_normalizado}__{chunk_idx}"

    Args:
        pares_validos: Lista de pares validados (ementa e inteiro teor).
        tokenizer: Tokenizador do modelo base.

    Returns:
        Tupla contendo (lista_de_chunks_do_corpus, estatisticas_do_processamento).
    """
    corpus_chunks = []
    stats = {
        "processos": 0,
        "total_chunks": 0,
        "chunks_por_processo": [],
        "chunk_unico": 0,
    }

    for par in tqdm(pares_validos, desc="Processando Chunks"):
        numero = normalizar_numero(par["str_numero_unico"])
        teor = par["str_inteiro_teor"].strip()

        chunks = chunkar_texto(teor, tokenizer, CHUNK_SIZE_TOKENS, CHUNK_OVERLAP_TOKENS)
        if not chunks:
            continue

        stats["processos"] += 1
        stats["total_chunks"] += len(chunks)
        stats["chunks_por_processo"].append(len(chunks))
        if len(chunks) == 1:
            stats["chunk_unico"] += 1

        for ch in chunks:
            corpus_chunks.append({
                "chunk_id": f"{numero}__{ch['chunk_idx']}",
                "str_numero_unico": par["str_numero_unico"],
                "chunk_idx": ch["chunk_idx"],
                "n_tokens": ch["n_tokens"],
                "texto": ch["texto"],
            })

    return corpus_chunks, stats


def main() -> None:
    """Execução principal do processo de filtragem e chunking."""
    print("=" * 64)
    print("BumbaLM - Etapa 1: Filtragem e Chunking de Documentos")
    print("=" * 64)
    print(f"Arquivo de Entrada : {ARQUIVO_TRAIN}")
    print(f"Arquivo Filtrado   : {ARQUIVO_FILTRADO}")
    print(f"Corpus de Chunks   : {ARQUIVO_CHUNKS}")
    print(f"Janela de Tokens   : {CHUNK_SIZE_TOKENS} tokens | Sobreposicao: {CHUNK_OVERLAP_TOKENS}")
    print("-" * 64)

    if not ARQUIVO_TRAIN.is_file():
        raise FileNotFoundError(f"Arquivo de entrada nao encontrado: {ARQUIVO_TRAIN}")

    # 1. Leitura e filtragem de qualidade
    registros_brutos = ler_jsonl(ARQUIVO_TRAIN)
    pares_validos, stats_filtro = filtrar_pares(registros_brutos)
    escrever_jsonl(ARQUIVO_FILTRADO, pares_validos)

    print(f"Total de registros lidos : {stats_filtro['total_lido']:,}")
    print(f"Pares validos aprovados  : {stats_filtro['validos']:,}")
    print(f"Registros descartados    : {stats_filtro['descartados']:,}")
    for motivo, qtd in sorted(stats_filtro["motivos_descarte"].items()):
        print(f"  - {motivo:24s}: {qtd:,}")
    print(f"Pares salvos em          : {ARQUIVO_FILTRADO}")

    if not pares_validos:
        raise ValueError("Nenhum par valido restou apos os filtros de qualidade.")

    # 2. Carregamento do tokenizador do modelo base
    print("-" * 64)
    print(f"Carregando tokenizador: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    print(f"Tamanho do vocabulario : {tokenizer.vocab_size:,}")

    # 3. Fragmentação do inteiro teor em chunks
    print("-" * 64)
    corpus_chunks, stats_chunks = gerar_corpus_chunks(pares_validos, tokenizer)
    escrever_jsonl(ARQUIVO_CHUNKS, corpus_chunks)

    print(f"Processos fragmentados   : {stats_chunks['processos']:,}")
    print(f"Total de chunks gerados  : {stats_chunks['total_chunks']:,}")
    if stats_chunks["chunks_por_processo"]:
        med = statistics.median(stats_chunks["chunks_por_processo"])
        print(f"Mediana chunks/processo  : {med:.1f}")
    print(f"Processos de chunk unico : {stats_chunks['chunk_unico']:,}")

    print("=" * 64)
    print("Processamento concluido com sucesso.")
    print(f"1. {ARQUIVO_FILTRADO}")
    print(f"2. {ARQUIVO_CHUNKS}")
    print("=" * 64)


if __name__ == "__main__":
    main()