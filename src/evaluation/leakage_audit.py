"""
Auditoria de vazamento de ementa no inteiro teor (Problema do Chunk 0).
"""

from __future__ import annotations

import difflib
import re
import unicodedata
from collections import Counter
from typing import Optional

from src.data.schema import ChunkProcesso, ProcessoJudicial


_ROTULOS_INICIAIS = re.compile(
    r"^(?:ementa|acordao|relatorio|voto(?: do relator)?|decisao|"
    r"vistos?|relatados?|discutidos?|e acordam)\s+"
)

# Faixas do histograma de cobertura gravado no relatorio da Etapa 2.
_FAIXAS_COBERTURA = [0.0, 0.05, 0.10, 0.20, 0.30, 0.50, 0.75, 1.01]


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


def _shingles(palavras: list[str], n: int) -> set[tuple[str, ...]]:
    return {tuple(palavras[i : i + n]) for i in range(len(palavras) - n + 1)}


def cobertura_ementa(
    palavras_ementa: list[str],
    palavras_chunk: list[str],
    ngram: int = 6,
) -> tuple[float, int]:
    """
    Fracao das palavras da ementa que aparecem copiadas no chunk, em blocos de `ngram` palavras.

    Pega copias parciais que a janela de 200 caracteres perde: a ementa cortada na borda do
    chunk ou interrompida por rodape e numeracao. Paráfrase nao conta, e frase-padrao curta
    (menos de `ngram` palavras, ex. "recurso conhecido e desprovido") tambem nao.

    Returns:
        (cobertura entre 0 e 1, numero de palavras da ementa cobertas)
    """
    if len(palavras_ementa) < ngram or len(palavras_chunk) < ngram:
        return 0.0, 0
    blocos_chunk = _shingles(palavras_chunk, ngram)
    coberta = [False] * len(palavras_ementa)
    for i in range(len(palavras_ementa) - ngram + 1):
        if tuple(palavras_ementa[i : i + ngram]) in blocos_chunk:
            for j in range(i, i + ngram):
                coberta[j] = True
    n_cobertas = sum(coberta)
    return n_cobertas / len(palavras_ementa), n_cobertas


def classificar_contaminacao(
    ementa: str,
    texto_chunk: str,
    trecho_minimo: int = 200,
    min_chunk_na_ementa: int = 100,
    limiar_cobertura: Optional[float] = 0.30,
    min_palavras_cobertura: int = 12,
    ngram_cobertura: int = 6,
) -> Optional[str]:
    """
    Classifica se o chunk contem vazamento espurio do texto da ementa.

    Retornos possiveis:
      - None: O chunk e genuina fundamentacao sem vazamento.
      - "inteira": A ementa completa esta reproduzida dentro do chunk.
      - "chunk_na_ementa": O chunk e curto e esta contido dentro da ementa.
      - "trecho": Uma janela contigua longa (>= trecho_minimo caracteres) da ementa
        esta presente de forma identica dentro do chunk.
      - "cobertura": Pelo menos `limiar_cobertura` das palavras da ementa (e no minimo
        `min_palavras_cobertura`) aparecem copiadas no chunk, mesmo que em pedacos.
        limiar_cobertura=None desliga este criterio.
    """
    motivo, _ = _classificar(
        remover_rotulos_iniciais(normalizar_texto_auditoria(ementa)),
        texto_chunk,
        trecho_minimo,
        min_chunk_na_ementa,
        limiar_cobertura,
        min_palavras_cobertura,
        ngram_cobertura,
    )
    return motivo


def _classificar(
    consulta: str,
    texto_chunk: str,
    trecho_minimo: int,
    min_chunk_na_ementa: int,
    limiar_cobertura: Optional[float],
    min_palavras_cobertura: int,
    ngram_cobertura: int,
) -> tuple[Optional[str], float]:
    """Recebe a ementa ja normalizada. Devolve (motivo, cobertura)."""
    chunk = remover_rotulos_iniciais(normalizar_texto_auditoria(texto_chunk))

    if not consulta or not chunk:
        return None, 0.0

    cobertura, n_cobertas = cobertura_ementa(consulta.split(), chunk.split(), ngram_cobertura)

    if consulta in chunk:
        return "inteira", cobertura

    if len(chunk) >= min_chunk_na_ementa and chunk in consulta:
        return "chunk_na_ementa", cobertura

    if len(consulta) >= trecho_minimo and len(chunk) >= trecho_minimo:
        # Verificacao por janela deslizante de caracteres
        janelas = {
            chunk[i : i + trecho_minimo]
            for i in range(len(chunk) - trecho_minimo + 1)
        }
        for i in range(len(consulta) - trecho_minimo + 1):
            if consulta[i : i + trecho_minimo] in janelas:
                return "trecho", cobertura

    if (
        limiar_cobertura is not None
        and cobertura >= limiar_cobertura
        and n_cobertas >= min_palavras_cobertura
    ):
        return "cobertura", cobertura

    return None, cobertura


def remover_bloco_ementa(
    teor: str,
    ementa: str,
    min_cobertura: float = 0.5,
    max_lacuna: int = 8,
    max_blocos: int = 3,
) -> tuple[str, int, int]:
    """
    Recorta do inteiro teor o bloco onde a ementa foi reproduzida (no TJRS, em geral no fim
    ou no meio do acordao), em vez de descartar o chunk inteiro e perder o voto que esta junto.

    A comparacao e palavra a palavra sobre o texto normalizado, entao tolera diferencas de
    acento, pontuacao e quebra de linha. Trechos casados separados por ate `max_lacuna`
    palavras contam como o mesmo bloco. So recorta quando o bloco cobre pelo menos
    `min_cobertura` das palavras da ementa: frases da ementa espalhadas pelo voto ficam.
    Repete ate `max_blocos` vezes, para o caso de a ementa aparecer mais de uma vez.

    Returns:
        (teor sem o bloco, blocos removidos, palavras removidas)
    """
    palavras_ementa = remover_rotulos_iniciais(normalizar_texto_auditoria(ementa)).split()
    if len(palavras_ementa) < 8:
        return teor, 0, 0

    blocos_removidos = 0
    palavras_removidas = 0
    for _ in range(max_blocos):
        # Cada palavra normalizada aponta para o token original, para recortar o texto bruto
        tokens = list(re.finditer(r"\S+", teor))
        palavras_teor: list[str] = []
        token_da_palavra: list[int] = []
        for indice, token in enumerate(tokens):
            for palavra in normalizar_texto_auditoria(token.group()).split():
                palavras_teor.append(palavra)
                token_da_palavra.append(indice)
        if not palavras_teor:
            break

        # autojunk=False: com o padrao, palavras frequentes ("de", "que") quebram os blocos
        casados = [
            b for b in difflib.SequenceMatcher(None, palavras_ementa, palavras_teor, autojunk=False).get_matching_blocks()
            if b.size >= 3
        ]
        if not casados:
            break
        casados.sort(key=lambda b: b.b)

        melhor: tuple[int, int, int] = (0, 0, 0)  # (palavras da ementa cobertas, inicio, fim) no teor
        inicio, fim, cobertas = casados[0].b, casados[0].b + casados[0].size, casados[0].size
        for bloco in casados[1:]:
            if bloco.b - fim <= max_lacuna:
                fim, cobertas = bloco.b + bloco.size, cobertas + bloco.size
            else:
                melhor = max(melhor, (cobertas, inicio, fim))
                inicio, fim, cobertas = bloco.b, bloco.b + bloco.size, bloco.size
        melhor = max(melhor, (cobertas, inicio, fim))

        cobertas, inicio, fim = melhor
        if cobertas / len(palavras_ementa) < min_cobertura:
            break
        # Leva junto o rotulo "EMENTA" logo antes do bloco
        if inicio > 0 and palavras_teor[inicio - 1] == "ementa":
            inicio -= 1
        corte_ini = tokens[token_da_palavra[inicio]].start()
        corte_fim = tokens[token_da_palavra[fim - 1]].end()
        palavras_removidas += fim - inicio
        blocos_removidos += 1
        teor = (teor[:corte_ini].rstrip() + "\n\n" + teor[corte_fim:].lstrip()).strip()

    return teor, blocos_removidos, palavras_removidas


def auditar_e_marcar_chunks(
    processo: ProcessoJudicial,
    chunks: list[ChunkProcesso],
    trecho_minimo: int = 200,
    limiar_cobertura: Optional[float] = 0.30,
    min_palavras_cobertura: int = 12,
    ngram_cobertura: int = 6,
) -> list[ChunkProcesso]:
    """
    Analisa todos os chunks de um determinado processo judicial, marcando
    os campos 'contem_vazamento_ementa', 'motivo_vazamento' e 'cobertura_ementa'.
    """
    # A ementa e normalizada uma vez por processo, nao uma vez por chunk.
    consulta = remover_rotulos_iniciais(normalizar_texto_auditoria(processo.ementa))
    for ch in chunks:
        motivo, cobertura = _classificar(
            consulta,
            ch.texto,
            trecho_minimo,
            100,
            limiar_cobertura,
            min_palavras_cobertura,
            ngram_cobertura,
        )
        ch.cobertura_ementa = round(cobertura, 4)
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

    # Processos sem nenhum chunk limpo saem da avaliacao: o numero entra no texto do TCC.
    limpos_por_processo: Counter = Counter()
    for c in chunks:
        limpos_por_processo[c.id_global_processo] += 0 if c.contem_vazamento_ementa else 1
    processos_sem_chunk_limpo = sum(1 for n in limpos_por_processo.values() if n == 0)

    # Histograma da cobertura, para justificar o limiar escolhido.
    histograma: dict[str, int] = {}
    for inicio, fim in zip(_FAIXAS_COBERTURA, _FAIXAS_COBERTURA[1:]):
        rotulo = f"[{inicio:.2f}, {min(fim, 1.0):.2f}{']' if fim > 1 else ')'}"
        histograma[rotulo] = sum(1 for c in chunks if inicio <= c.cobertura_ementa < fim)

    return {
        "total_chunks": total,
        "limpos": limpos,
        "contaminados": contaminados,
        "taxa_contaminacao": round(contaminados / max(1, total), 4),
        "motivos": dict(por_motivo),
        "distribuicao_por_indice_chunk": {f"chunk_{k}": v for k, v in sorted(por_chunk_idx.items())},
        "total_processos": len(limpos_por_processo),
        "processos_sem_chunk_limpo": processos_sem_chunk_limpo,
        "histograma_cobertura": histograma,
    }
