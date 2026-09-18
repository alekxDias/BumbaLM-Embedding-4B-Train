#!/usr/bin/env python3
"""
Auditoria e Descontaminacao de Validacao (Vazamento de Ementa).

Problema Abordado:
Em acórdãos judiciais brasileiros, o início do documento de inteiro teor frequentemente
reproduz na íntegra a ementa do julgado (Chunk 0).
Se a validação do modelo avaliar a recuperação desse trecho idêntico à consulta,
ocorre um vazamento espúrio de dados (data leakage), inflando artificialmente as
métricas de recuperação de informação (nDCG@10, MRR@10, Accuracy@1).

Objetivo:
1. Auditar conjuntos de validação (ou coleções de chunks) identificando instâncias
   onde o texto do documento relevante é uma cópia literal ou parcial da ementa.
2. Identificar os três modos de contaminação:
   - "inteira": a ementa inteira está contida no chunk.
   - "trecho": uma janela longa (>= 200 caracteres normalizados) da ementa está no chunk.
   - "chunk_na_ementa": o chunk é curto (>= 100 caracteres) e está inteiramente contido na ementa.
3. Emitir relatório detalhado com distribuição de contaminações por índice de chunk.
4. Opcionalmente (com --gravar), salvar o arquivo descontaminado, garantindo que
   apenas chunks de efetiva fundamentação jurídica permaneçam como positivos.

Esquemas suportados:
  a. Formato de Tripletos (ex.: tripletos_val.jsonl):
     Campos: query, positive, str_numero_unico, positive_chunk_idx
  b. Formato de Corpus de Chunks (ex.: corpus_chunks_val.jsonl):
     Campos: str_ementa, str_inteiro_teor, chunk_idx, str_numero_unico

Uso:
    # Apenas emitir relatório estatístico (sem alterar arquivos):
    python limpar_validacao.py --entrada tripletos_val.jsonl

    # Gerar versão descontaminada:
    python limpar_validacao.py --entrada tripletos_val.jsonl --gravar
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional


# Parâmetros de auditoria léxica
TRECHO_MINIMO = 200          # Comprimento mínimo de janela contígua para caracterizar cópia de trecho
MIN_CHUNK_NA_EMENTA = 100    # Comprimento mínimo para flag de chunk totalmente contido na ementa

# Prefixos formais recorrentes em decisões judiciais que são ignorados na normalização
_ROTULOS_INICIAIS = re.compile(
    r"^(?:ementa|acordao|relatorio|voto(?: do relator)?|decisao|"
    r"vistos?|relatados?|discutidos?|e acordam)\s+"
)


def parse_args() -> argparse.Namespace:
    """Configura e processa os argumentos de linha de comando."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--entrada",
        type=Path,
        required=True,
        help="Caminho do arquivo JSONL de entrada (tripletos ou corpus).",
    )
    parser.add_argument(
        "--saida",
        type=Path,
        default=None,
        help="Caminho do arquivo limpo. Padrao: <entrada>_limpo.jsonl",
    )
    parser.add_argument(
        "--relatorio",
        type=Path,
        default=None,
        help="Caminho do relatorio de auditoria. Padrao: <entrada>_relatorio.txt",
    )
    parser.add_argument(
        "--gravar",
        action="store_true",
        help="Persiste o arquivo descontaminado. Sem esta flag, apenas audita e emite relatorio.",
    )
    return parser.parse_args()


def text_or_empty(value) -> str:
    """Converte valor para string limpa de espaços."""
    return str(value or "").strip()


def normalizar(texto: str) -> str:
    """
    Remove acentos, pontuações e converte para minúsculas para comparação uniforme.
    """
    texto = unicodedata.normalize("NFKD", str(texto or "").lower())
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", " ", texto).strip()


def cortar_rotulos(texto: str) -> str:
    """Remove palavras de cabeçalho judicial no início do texto."""
    while True:
        novo = _ROTULOS_INICIAIS.sub("", texto)
        if novo == texto:
            return texto
        texto = novo


def numero_processo(raw) -> str:
    """Extrai somente os dígitos do número do processo."""
    base = str(raw or "").strip().split("__")[0]
    return "".join(c for c in base if c.isdigit()) or "(sem_numero)"


def chunk_idx_de(item: dict) -> Optional[int]:
    """Recupera o índice sequencial do chunk a partir dos campos disponíveis."""
    raw = item.get("positive_chunk_idx")
    if raw is None:
        raw = item.get("chunk_idx")
    if raw is None:
        sid = text_or_empty(item.get("str_numero_unico") or item.get("positive_chunk_id"))
        if "__" in sid:
            raw = sid.rsplit("__", 1)[-1]
    if raw is None or raw == "":
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def texto_consulta(item: dict) -> str:
    """Obtém a ementa/consulta do registro."""
    return text_or_empty(item.get("query") or item.get("str_ementa"))


def texto_documento(item: dict) -> str:
    """Obtém o texto do chunk (positivo). Nunca utiliza a ementa como documento."""
    return text_or_empty(
        item.get("positive")
        or item.get("texto")
        or item.get("str_inteiro_teor")
    )


def detectar_schema(registros: list[dict]) -> str:
    """Identifica automaticamente o formato do arquivo fornecido."""
    amostra = registros[0]
    if amostra.get("query") is not None or amostra.get("positive") is not None:
        return "tripletos"
    if amostra.get("str_ementa") is not None or amostra.get("str_inteiro_teor") is not None:
        return "corpus_chunks"
    return "desconhecido"


def classificar_contaminacao(query: str, positive: str) -> Optional[str]:
    """
    Avalia a sobreposição textual entre consulta e documento.
    
    Returns:
        None se não houver contaminação; caso contrário, 'inteira', 'trecho' ou 'chunk_na_ementa'.
    """
    consulta = cortar_rotulos(normalizar(query))
    chunk = cortar_rotulos(normalizar(positive))
    if not consulta or not chunk:
        return None
    if consulta in chunk:
        return "inteira"
    if len(chunk) >= MIN_CHUNK_NA_EMENTA and chunk in consulta:
        return "chunk_na_ementa"
    if len(consulta) < TRECHO_MINIMO or len(chunk) < TRECHO_MINIMO:
        return None

    # Verificação por janela deslizante de 200 caracteres
    janelas = {
        chunk[i : i + TRECHO_MINIMO]
        for i in range(len(chunk) - TRECHO_MINIMO + 1)
    }
    for i in range(len(consulta) - TRECHO_MINIMO + 1):
        if consulta[i : i + TRECHO_MINIMO] in janelas:
            return "trecho"
    return None


def ler_registros(caminho: Path) -> list[dict]:
    """Lê registros suportando formato JSON array ou JSONL linha a linha."""
    texto = caminho.read_text(encoding="utf-8").lstrip()
    if texto.startswith("["):
        dados = json.loads(texto)
        if not isinstance(dados, list):
            raise ValueError("O JSON raiz do arquivo nao e uma lista.")
        return dados

    registros = []
    with caminho.open(encoding="utf-8") as arquivo:
        for numero, linha in enumerate(arquivo, start=1):
            linha = linha.strip()
            if not linha:
                continue
            try:
                registros.append(json.loads(linha))
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSON invalido na linha {numero}") from exc
    return registros


def chave_chunk(idx: Optional[int]) -> str:
    """Formata o índice do chunk para agrupamento estatístico."""
    if idx is None:
        return "(sem_idx)"
    return str(idx)


def main() -> None:
    """Execução principal da auditoria e limpeza de validação."""
    args = parse_args()
    if not args.entrada.is_file():
        raise SystemExit(f"Arquivo nao encontrado: {args.entrada}")

    saida = args.saida or args.entrada.with_name(args.entrada.stem + "_limpo.jsonl")
    relatorio = args.relatorio or args.entrada.with_name(args.entrada.stem + "_relatorio.txt")

    registros = ler_registros(args.entrada)
    if not registros:
        raise SystemExit("Nenhum registro encontrado no arquivo informado.")

    schema = detectar_schema(registros)
    por_chunk: Counter[str] = Counter()
    por_motivo: Counter[str] = Counter()
    por_processo: dict[str, list[tuple[dict, Optional[str]]]] = defaultdict(list)
    rotulos_motivo = {
        "inteira": "Ementa inteira no chunk",
        "trecho": "Trecho longo da ementa (>=200 chars)",
        "chunk_na_ementa": "Chunk contido na ementa",
    }
    sem_consulta = 0
    sem_documento = 0

    for item in registros:
        consulta = texto_consulta(item)
        documento = texto_documento(item)
        if not consulta:
            sem_consulta += 1
        if not documento:
            sem_documento += 1
        motivo = classificar_contaminacao(consulta, documento)
        numero = numero_processo(item.get("str_numero_unico"))
        if motivo is not None:
            por_motivo[motivo] += 1
            por_chunk[chave_chunk(chunk_idx_de(item))] += 1
        por_processo[numero].append((item, motivo))

    limpos: list[dict] = []
    contaminados = 0
    consultas_sem_relevante = 0
    chunks_por_processo = [len(pares) for pares in por_processo.values()]
    media_chunks = (
        sum(chunks_por_processo) / len(chunks_por_processo) if chunks_por_processo else 0.0
    )

    for pares in por_processo.values():
        sobreviventes = [item for item, motivo in pares if motivo is None]
        contaminados += sum(1 for _, motivo in pares if motivo is not None)
        if not sobreviventes:
            consultas_sem_relevante += 1
            continue
        limpos.extend(sobreviventes)

    consultas_restantes = len(por_processo) - consultas_sem_relevante
    linhas = [
        "=" * 64,
        f"Relatorio de Auditoria de Vazamento: {args.entrada.name}",
        "=" * 64,
        f"Schema detectado                 : {schema}",
        f"Campo de consulta                : query / str_ementa",
        f"Campo de documento               : positive / texto / str_inteiro_teor",
        f"Agrupamento por processo         : digitos de str_numero_unico",
        f"Instancias analisadas            : {len(registros):,}",
        f"Linhas sem consulta              : {sem_consulta:,}",
        f"Linhas sem documento             : {sem_documento:,}",
        f"Total de processos unicos        : {len(por_processo):,}",
        f"Media de chunks por processo     : {media_chunks:.2f}",
        f"Casos com ementa no documento    : {contaminados:,} "
        f"({contaminados / len(registros):.1%})",
    ]
    for chave, rotulo in rotulos_motivo.items():
        linhas.append(f"  - {rotulo:<35}: {por_motivo[chave]:,}")
    linhas.extend(
        [
            f"Processos sem positivo limpo     : {consultas_sem_relevante:,}",
            f"Instancias validas restantes     : {len(limpos):,}",
            f"Processos avaliaveis restantes   : {consultas_restantes:,}",
            "",
            "Distribuicao de contaminacoes por indice de chunk:",
        ]
    )
    if por_chunk:
        def ordem_chunk(item: tuple[str, int]) -> tuple:
            chave, _ = item
            if chave.lstrip("-").isdigit():
                return (0, int(chave))
            return (1, chave)

        for idx, qtd in sorted(por_chunk.items(), key=ordem_chunk):
            linhas.append(f"  Chunk {idx}: {qtd:,}")
    else:
        linhas.append("  Nenhum chunk contaminado identificado.")

    if schema == "corpus_chunks":
        linhas.extend(
            [
                "",
                "Nota Metodologica:",
                "No schema corpus_chunks, o Chunk 0 contaminado pode atuar como",
                "distrator desafiador na avaliacao. A flag --gravar remove esses",
                "trechos da colecao. Para manter distratores dificeis, utilize apenas",
                "o relatorio sem aplicar --gravar no corpus.",
            ]
        )
    linhas.append("=" * 64)

    texto = "\n".join(linhas)
    print(texto)
    relatorio.write_text(texto + "\n", encoding="utf-8")
    print(f"\nRelatorio salvo em: {relatorio}")

    if args.gravar:
        with saida.open("w", encoding="utf-8") as arquivo:
            for item in limpos:
                arquivo.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"Arquivo descontaminado salvo com sucesso em: {saida}")
    else:
        print("Auditoria concluida em modo somente-leitura. Use a flag --gravar para persistir os dados limpos.")


if __name__ == "__main__":
    main()