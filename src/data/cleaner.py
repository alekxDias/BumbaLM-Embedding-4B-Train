"""
Limpeza, normalizacao de texto e filtros de qualidade para processos judiciais.
"""

from __future__ import annotations

import html
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

from src.data.schema import ProcessoJudicial


# 1. Regex de Tags HTML e Scripts
_RE_TAGS_HTML = re.compile(r"<[^>]+>")

# 2. Regex de Caracteres de Controle e Invisiveis (BOM, Zero-Width, Soft-Hyphen)
_RE_CARACTERES_INVISIVEIS = re.compile(
    r"[\ufeff\u200b\u200c\u200d\u200e\u200f\xad\x00-\x08\x0b\x0c\x0e-\x1f\x7f]"
)

# 3. Regex de Linhas Divisorias Repetitivas (ex.: '----------------', '================')
_RE_DIVISORES_REPETITIVOS = re.compile(r"([-=_.*~#]){4,}")

# 4. Regex de Palavras Hifenizadas em Quebra de Linha (ex.: 'responsa- \nbilidade' -> 'responsabilidade')
_RE_HIFEN_QUEBRA_LINHA = re.compile(
    r"(\b[a-zA-Zá-úÁ-Úà-ùÀ-Ùâ-ûÂ-Ûã-õÃ-ÕçÇ]+)-\s*\n\s*([a-zA-Zá-úÁ-Úà-ùÀ-Ùâ-ûÂ-Ûã-õÃ-ÕçÇ]+\b)"
)

# 5. Regex de Metadados de Assinatura Eletronica e Carimbos de Pagina
_RE_CARIMBOS_ASSINATURA = re.compile(
    r"(?i)(?:documento\s+assinado\s+digitalmente[^\n]*|"
    r"assinado\s+eletronicamente\s+por[^\n]*|"
    r"num\.\s*\d+\s*-\s*p[aá]g\.\s*\d+|"
    r"p[aá]gina\s+\d+\s+de\s+\d+)"
)

# 6. Regex de Espacamentos e Quebras Artificiais
_RE_ESPACOS_MULTIPLOS = re.compile(r"[ \t]+")
_RE_QUEBRAS_EXCESSIVAS = re.compile(r"\n\s*\n+")
_RE_ESPACOS_AO_REDOR_QUEBRAS = re.compile(r"[ \t]*\n[ \t]*")

# Caracteres, parenteses e operadores isolados
_RE_PONTUACAO_ISOLADA_LINHA = re.compile(r"\n+([:;/\\–—\-])\n+")
_RE_PARENTESES_ITENS = re.compile(r"\(\s*([a-zA-Z0-9ivxldcmIVXLDCM§º°ª\.\-]+)\s*\)")
_RE_SIGLA_BARRA_QUEBRADA = re.compile(r"(?i)\b([a-z0-9])\s*/\s*([a-z0-9])\b")

# Line Reflow / Recomposicao de linhas quebradas
_RE_QUEBRA_CONTINUACAO = re.compile(r"([a-zA-Zá-úÁ-Úà-ùÀ-Ùâ-ûÂ-Ûã-õÃ-ÕçÇ0-9,])\n+([a-zá-úà-ùâ-ûã-õç])")
_RE_QUEBRA_PONTUACAO_SEGUIDA = re.compile(r"([,;\-–—])\n+([A-ZÁ-ÚÀ-ÙÂ-ÛÃ-Õa-zá-úà-ùâ-ûã-õçÇ0-9§])")
_RE_QUEBRA_ABREVIACOES = re.compile(
    r"(?i)\b(e\.|d\.|art\.|arts\.|inc\.|incs\.|fls?\.|lei|n[º°ª]|par[aá]grafo|c/c|c\.c\.|d[oe]|d[ao]s?)\n+([A-ZÁ-ÚÀ-ÙÂ-ÛÃ-Õa-zá-úà-ùâ-ûã-õçÇ0-9§])"
)
_RE_QUEBRA_CONECTIVOS = re.compile(
    r"(?i)\b(de|do|da|dos|das|em|no|na|nos|nas|com|por|para|ao|aos|sob|sobre|pelo|pela|pelos|pelas|c/c)\n+([A-ZÁ-ÚÀ-ÙÂ-ÛÃ-Õ0-9§][a-zá-úà-ùâ-ûã-õç0-9]*)"
)
_RE_QUEBRA_NUMEROS_LEGISLATIVOS = re.compile(
    r"(\b\d+[a-zA-Zº°ª]*|\b\d+X)\n+([A-ZÁ-ÚÀ-ÙÂ-ÛÃ-Õa-zá-úà-ùâ-ûã-õçÇ0-9§])"
)
_RE_QUEBRA_NOMES_PROPRIOS = re.compile(
    r"([A-ZÁ-ÚÀ-ÙÂ-ÛÃ-Õ][a-zá-úà-ùâ-ûã-õç]+)\n+([A-ZÁ-ÚÀ-ÙÂ-ÛÃ-Õ][a-zá-úà-ùâ-ûã-õç]+)"
)
_RE_QUEBRA_DOIS_PONTOS = re.compile(r"(:)\n+([A-ZÁ-ÚÀ-ÙÂ-ÛÃ-Õ0-9])")

# Correcao de Kerning / Letras e Silabas Descoladas por OCR de PDF
_RE_KERNING_CONSOANTE_MAIUSCULA = re.compile(r"\b([B-HJ-NP-TW-Z])\s+([a-zá-úà-ùâ-ûã-õç]{3,})\b")
_RE_KERNING_CONSOANTE_MINUSCULA = re.compile(r"\b([b-df-hj-np-tv-z])\s+([a-zá-úà-ùâ-ûã-õç]{3,})\b")
_RE_KERNING_PREP_D = re.compile(r"\b(d)\s+([oas]{1,2})\b")
_RE_KERNING_PLURAL_S = re.compile(r"\b([a-zA-Zá-úÁ-Úà-ùÀ-Ùâ-ûÂ-Ûã-õÃ-ÕçÇ]{3,}[a-zá-úà-ùâ-ûã-õç])\s+([sS])\b(?=[^\w]|$)")
_RE_KERNING_VOGAL_FINAL = re.compile(r"\b([a-zA-Zá-úÁ-Úà-ùÀ-Ùâ-ûÂ-Ûã-õÃ-ÕçÇ]{3,}(?:post|pel|ativ|itiv|ad|ent))\s+([oOaA])\b")
_RE_KERNING_SUFIXO_MENTO = re.compile(r"\b([a-zA-Zá-úÁ-Úà-ùÀ-Ùâ-ûÂ-Ûã-õÃ-ÕçÇ]{4,}amen)\s+to\b")
_RE_KERNING_SUFIXO_MENTE = re.compile(r"\b([a-zA-Zá-úÁ-Úà-ùÀ-Ùâ-ûÂ-Ûã-õÃ-ÕçÇ]{4,}(?:tiv|tiva|ria|iva|ica|al|el|il))\s+mente\b")
_RE_KERNING_TERMOS_COMUNS = re.compile(r"\b(insufici[eê]n)\s+ci\s+a\b")
_RE_REDIMENSIONA = re.compile(r"\b(redimensio)\s+na\s+(mento)\b")

# Deteccao de Corrupcao de OCR e Fontes de PDF (glifos CID, /0/1, codigos brutos)
_RE_LETRAS = re.compile(r"[a-zA-Zá-úÁ-Úà-ùÀ-Ùâ-ûÂ-Ûã-õÃ-ÕçÇ]")
_RE_PALAVRAS_REAIS = re.compile(r"[a-zA-Zá-úÁ-Úà-ùÀ-Ùâ-ûÂ-Ûã-õÃ-ÕçÇ]{2,}")


def eh_texto_valido(texto: str, min_taxa_letras: float = 0.40, min_taxa_palavras_reais: float = 0.50) -> bool:
    """Verifica se o texto e legivel em portugues ou se e lixo de extracao de PDF (fontes CID, glifos, OCR quebrado)."""
    if not texto or not isinstance(texto, str):
        return False
    chars_total = len(texto)
    if chars_total == 0:
        return False
    letras = len(_RE_LETRAS.findall(texto))
    if (letras / chars_total) < min_taxa_letras:
        return False
    palavras = texto.split()
    if not palavras:
        return False
    palavras_reais = sum(1 for p in palavras if _RE_PALAVRAS_REAIS.search(p))
    if (palavras_reais / len(palavras)) < min_taxa_palavras_reais:
        return False
    # Anomalia de barras de glifos de PDF (ex: /0/1 /2/0 quebrado)
    if (texto.count("/") / chars_total) > 0.08:
        return False
    return True


def limpar_texto(
    texto: str,
    remover_html: bool = True,
    desescapar_html: bool = True,
    normalizar_unicode: bool = True,
    remover_invisiveis: bool = True,
    remover_carimbos_assinatura: bool = True,
    juntar_hifens_quebra: bool = True,
    reduzir_divisores: bool = True,
) -> str:
    """
    Limpeza profunda de texto juridico para treinamento de embeddings.

    Etapas executadas:
    1. Conversao de entidades HTML (&nbsp;, &quot;, &atilde;, etc.) via html.unescape.
    2. Remocao de tags HTML (<p>, <div>, etc.).
    3. Normalizacao Unicode NFKC (uniformiza aspas, tracos e ligaduras tipograficas).
    4. Remocao de caracteres invisiveis (BOM, zero-width, soft hyphens e caracteres de controle).
    5. Conversao de espacos rigidos (\xa0) para espacos normais.
    6. Remocao de carimbos de assinatura eletronica e contadores de pagina ("Num. X - Pag. Y").
    7. Juncao de palavras partidas por hifen no final de linha ("juris- \nprudencia" -> "jurisprudencia").
    8. Reducao de linhas de separadores repetitivos (ex.: "-------" reduzido para "---").
    9. Padronizacao de espacos horizontais e quebras de linha duplas.
    """
    if not texto:
        return ""

    limpo = str(texto)

    # 1. Desescapa entidades HTML (&nbsp;, &atilde;, etc.)
    if desescapar_html:
        limpo = html.unescape(limpo)

    # 2. Converte tags de bloco HTML em quebras de linha reais antes de remover tags
    if remover_html:
        limpo = re.sub(r"(?i)<\s*(?:br|p|div|tr|li|\/p|\/div|\/tr|\/li)[^>]*>", "\n", limpo)
        limpo = _RE_TAGS_HTML.sub(" ", limpo)

    # 3. Substitui espacos nao quebraveis (\xa0) por espaco comum
    limpo = limpo.replace("\xa0", " ")

    # 4. Remove caracteres invisiveis e de controle
    if remover_invisiveis:
        limpo = _RE_CARACTERES_INVISIVEIS.sub("", limpo)

    # 5. Normalizacao Unicode NFKC
    if normalizar_unicode:
        limpo = unicodedata.normalize("NFKC", limpo)

    # 6. Remove carimbos de assinatura eletronica e paginadores repetitivos
    if remover_carimbos_assinatura:
        limpo = _RE_CARIMBOS_ASSINATURA.sub("", limpo)

    # 7. Corrige palavras partidas por quebra de linha
    if juntar_hifens_quebra:
        limpo = _RE_HIFEN_QUEBRA_LINHA.sub(r"\1\2", limpo)

    # 8. Recomposicao de linhas quebradas artificialmente (Line Reflow / Unwrap)
    limpo = _RE_ESPACOS_AO_REDOR_QUEBRAS.sub("\n", limpo)
    limpo = _RE_PONTUACAO_ISOLADA_LINHA.sub(r"\1 ", limpo)
    limpo = _RE_PARENTESES_ITENS.sub(r"(\1)", limpo)
    limpo = _RE_SIGLA_BARRA_QUEBRADA.sub(r"\1/\2", limpo)
    limpo = _RE_QUEBRA_CONTINUACAO.sub(r"\1 \2", limpo)
    limpo = _RE_QUEBRA_PONTUACAO_SEGUIDA.sub(r"\1 \2", limpo)
    limpo = _RE_QUEBRA_ABREVIACOES.sub(r"\1 \2", limpo)
    limpo = _RE_QUEBRA_CONECTIVOS.sub(r"\1 \2", limpo)
    limpo = _RE_QUEBRA_NUMEROS_LEGISLATIVOS.sub(r"\1 \2", limpo)
    limpo = _RE_QUEBRA_NOMES_PROPRIOS.sub(r"\1 \2", limpo)
    limpo = _RE_QUEBRA_DOIS_PONTOS.sub(r"\1 \2", limpo)

    # 9. Correcao de kerning e letras soltas de extratores PDF
    limpo = _RE_KERNING_CONSOANTE_MAIUSCULA.sub(r"\1\2", limpo)
    limpo = _RE_KERNING_CONSOANTE_MINUSCULA.sub(r"\1\2", limpo)
    limpo = _RE_KERNING_PREP_D.sub(r"\1\2", limpo)
    limpo = _RE_KERNING_PLURAL_S.sub(r"\1s", limpo)
    limpo = _RE_KERNING_VOGAL_FINAL.sub(r"\1\2", limpo)
    limpo = _RE_KERNING_SUFIXO_MENTO.sub(r"\1to", limpo)
    limpo = _RE_KERNING_SUFIXO_MENTE.sub(r"\1mente", limpo)
    limpo = _RE_KERNING_TERMOS_COMUNS.sub(r"\1cia", limpo)
    limpo = _RE_KERNING_REDIMENSIONA.sub(r"\1na\2", limpo)

    # 10. Reduz linhas divisorias decorativas que gastam tokens a toa
    if reduzir_divisores:
        limpo = _RE_DIVISORES_REPETITIVOS.sub("---", limpo)

    # 11. Padroniza espacos horizontais e quebras de linha
    limpo = _RE_ESPACOS_MULTIPLOS.sub(" ", limpo)
    limpo = _RE_QUEBRAS_EXCESSIVAS.sub("\n\n", limpo)

    return limpo.strip()


def carregar_e_filtrar_processos(
    caminho_arquivo: Path | str | list[Path | str],
    mapeamento_colunas: dict[str, str],
    min_caracteres_ementa: int = 50,
    min_caracteres_teor: int = 200,
    descartar_teor_menor_que_ementa: bool = True,
    remover_html: bool = True,
    max_por_tribunal: int = 0,
) -> tuple[list[ProcessoJudicial], dict[str, Any]]:
    """
    Le os arquivos brutos JSONL (arquivo unico, lista de arquivos ou diretorio),
    mapeia as colunas e aplica a limpeza profunda junto aos filtros de qualidade.

    Parametros:
      - max_por_tribunal: Se > 0, limita a quantidade de processos validos por tribunal
                          (essencial para testes rapidos e balanceamento).
    """
    if isinstance(caminho_arquivo, (list, tuple)):
        lista_caminhos = [Path(p) for p in caminho_arquivo]
    elif Path(caminho_arquivo).is_dir():
        lista_caminhos = sorted(list(Path(caminho_arquivo).glob("pares_*.jsonl")))
    else:
        lista_caminhos = [Path(caminho_arquivo)]

    col_tribunal = mapeamento_colunas.get("tribunal", "str_tribunal")
    col_numero = mapeamento_colunas.get("numero_processo", "str_numero_unico")
    col_ementa = mapeamento_colunas.get("ementa", "str_ementa")
    col_teor = mapeamento_colunas.get("inteiro_teor", "str_inteiro_teor")

    processos_validos: list[ProcessoJudicial] = []
    motivos_descarte = Counter()
    vistos: set[str] = set()
    total_lido = 0
    contagem_tribunal = Counter()

    for caminho in lista_caminhos:
        if not caminho.is_file():
            print(f"Aviso: Arquivo nao encontrado ou ignorado: {caminho}")
            continue

        with caminho.open(encoding="utf-8", errors="replace") as f:
            for num_linha, linha in enumerate(f, start=1):
                linha = linha.strip()
                if not linha:
                    continue

                total_lido += 1
                try:
                    dado = json.loads(linha)
                except json.JSONDecodeError:
                    motivos_descarte["json_invalido"] += 1
                    continue

                if not isinstance(dado, dict):
                    motivos_descarte["registro_nao_dicionario"] += 1
                    continue

                tribunal = str(dado.get(col_tribunal) or dado.get("tribunal") or dado.get("str_tribunal") or "TCU" if "ANOACORDAO" in dado else "").strip().upper()
                if not tribunal:
                    partes = caminho.stem.replace("pares_", "").split("_")
                    tribunal = partes[0].upper()

                if max_por_tribunal > 0 and contagem_tribunal[tribunal] >= max_por_tribunal:
                    break

                proc_val = str(dado.get(col_numero) or dado.get("str_numero_unico") or dado.get("numero_processo") or dado.get("numero") or dado.get("PROC") or dado.get("KEY") or "").strip()
                numero = re.sub(r"<[^>]+>", "", proc_val).strip()
                if numero == "000.000/0000-0":
                    numero = str(dado.get("KEY") or dado.get("NUMACORDAO") or "").strip()

                ementa_raw = str(dado.get(col_ementa) or dado.get("str_ementa") or dado.get("ementa") or dado.get("SUMARIO") or dado.get("sumario") or "").strip()
                if ementa_raw.lower() in ["número de acórdão não utilizado", "numero de acordao nao utilizado"]:
                    ementa_raw = ""

                teor_raw = str(dado.get(col_teor) or dado.get("str_inteiro_teor") or dado.get("texto_inteiro_teor") or dado.get("inteiro_teor") or "").strip()

                if not numero:
                    motivos_descarte["sem_numero_processo"] += 1
                    continue

                ementa = limpar_texto(ementa_raw, remover_html=remover_html)
                teor = limpar_texto(teor_raw, remover_html=remover_html)

                if not ementa or not teor:
                    motivos_descarte["campo_vazio"] += 1
                    continue

                if len(ementa) < min_caracteres_ementa:
                    motivos_descarte["ementa_curta"] += 1
                    continue

                if len(teor) < min_caracteres_teor:
                    motivos_descarte["teor_curto"] += 1
                    continue

                if descartar_teor_menor_que_ementa and len(teor) < len(ementa):
                    motivos_descarte["teor_menor_que_ementa"] += 1
                    continue

                if not eh_texto_valido(ementa):
                    motivos_descarte["ementa_corrompida_ocr_pdf"] += 1
                    continue

                if not eh_texto_valido(teor):
                    motivos_descarte["teor_corrompido_ocr_pdf"] += 1
                    continue

                processo = ProcessoJudicial(
                    tribunal=tribunal,
                    numero_processo=numero,
                    ementa=ementa,
                    inteiro_teor=teor,
                    metadados={k: v for k, v in dado.items() if k not in (col_tribunal, col_numero, col_ementa, col_teor)},
                )

                if processo.id_global in vistos:
                    motivos_descarte["processo_duplicado"] += 1
                    continue

                vistos.add(processo.id_global)
                processos_validos.append(processo)
                contagem_tribunal[tribunal] += 1

                if max_por_tribunal > 0 and contagem_tribunal[tribunal] >= max_por_tribunal:
                    break

    estatisticas = {
        "total_lido": total_lido,
        "validos": len(processos_validos),
        "descartados": sum(motivos_descarte.values()),
        "motivos_descarte": dict(motivos_descarte),
        "distribuicao_tribunais": dict(Counter(p.tribunal for p in processos_validos)),
    }

    return processos_validos, estatisticas

