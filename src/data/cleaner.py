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

# 6. Regex de Espacamentos
_RE_ESPACOS_MULTIPLOS = re.compile(r"[ \t]+")
_RE_QUEBRAS_EXCESSIVAS = re.compile(r"\n\s*\n+")


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

    # 2. Remove tags HTML residuais
    if remover_html:
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

    # 8. Reduz linhas divisorias decorativas que gastam tokens a toa
    if reduzir_divisores:
        limpo = _RE_DIVISORES_REPETITIVOS.sub("---", limpo)

    # 9. Padroniza espacos horizontais e quebras de linha
    limpo = _RE_ESPACOS_MULTIPLOS.sub(" ", limpo)
    limpo = _RE_QUEBRAS_EXCESSIVAS.sub("\n\n", limpo)

    return limpo.strip()


def carregar_e_filtrar_processos(
    caminho_arquivo: Path | str,
    mapeamento_colunas: dict[str, str],
    min_caracteres_ementa: int = 50,
    min_caracteres_teor: int = 200,
    descartar_teor_menor_que_ementa: bool = True,
    remover_html: bool = True,
) -> tuple[list[ProcessoJudicial], dict[str, Any]]:
    """
    Le o arquivo bruto JSONL, mapeia as colunas e aplica a limpeza profunda
    junto aos filtros de qualidade e integridade.

    Filtros aplicados:
      - sem_numero_processo: registro sem identificador unico
      - processo_duplicado: processo ja visto com o mesmo id_global ({tribunal}_{numero})
      - campo_vazio: ementa ou inteiro teor ausentes
      - ementa_curta: ementa menor que min_caracteres_ementa apos a limpeza
      - teor_curto: inteiro teor menor que min_caracteres_teor apos a limpeza
      - teor_menor_que_ementa: inconsistencia em que a decisao e menor que o proprio resumo

    Returns:
        Tupla com (lista_de_ProcessoJudicial, relatorio_de_estatisticas).
    """
    caminho = Path(caminho_arquivo)
    if not caminho.is_file():
        raise FileNotFoundError(f"Arquivo de dados nao encontrado: {caminho}")

    col_tribunal = mapeamento_colunas.get("tribunal", "str_tribunal")
    col_numero = mapeamento_colunas.get("numero_processo", "str_numero_unico")
    col_ementa = mapeamento_colunas.get("ementa", "str_ementa")
    col_teor = mapeamento_colunas.get("inteiro_teor", "str_inteiro_teor")

    processos_validos: list[ProcessoJudicial] = []
    motivos_descarte = Counter()
    vistos: set[str] = set()
    total_lido = 0

    with caminho.open(encoding="utf-8") as f:
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

            tribunal = str(dado.get(col_tribunal) or "").strip()
            numero = str(dado.get(col_numero) or "").strip()
            ementa_raw = str(dado.get(col_ementa) or "").strip()
            teor_raw = str(dado.get(col_teor) or "").strip()

            if not numero:
                motivos_descarte["sem_numero_processo"] += 1
                continue

            # Aplica a limpeza profunda
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

    estatisticas = {
        "total_lido": total_lido,
        "validos": len(processos_validos),
        "descartados": sum(motivos_descarte.values()),
        "motivos_descarte": dict(motivos_descarte),
        "distribuicao_tribunais": dict(Counter(p.tribunal for p in processos_validos)),
    }

    return processos_validos, estatisticas
