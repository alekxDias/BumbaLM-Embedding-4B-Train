"""
Agrupamento de processos com ementas quase identicas (litigancia em massa).

O numero do processo nao basta para deduplicar: o TJMA tem muitos acordaos quase iguais
em processos diferentes. Sem agrupar, um caso do teste tem um "gemeo" no treino, e o
chunk do gemeo vira negativo na mineracao mesmo sendo, na pratica, relevante.

MinHash + LSH acha os pares candidatos; o Jaccard exato confirma cada par.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict
from typing import Any

import numpy as np

from src.data.schema import ProcessoJudicial
from src.evaluation.leakage_audit import normalizar_texto_auditoria

_PRIMO = np.uint64((1 << 61) - 1)


def _hash_estavel(texto: str) -> int:
    # hash() do Python muda a cada execucao. blake2b deixa os grupos reproduziveis.
    return int.from_bytes(hashlib.blake2b(texto.encode("utf-8"), digest_size=8).digest(), "little")


def shingles_texto(texto: str, ngram: int = 5) -> set[int]:
    """
    Blocos de `ngram` palavras do texto normalizado, como inteiros de 61 bits.
    Numeros isolados viram "0": nos casos em massa a ementa muda o valor da condenacao, a data
    e o numero do processo, e isso nao deve separar acordaos de mesmo texto. Digitos dentro
    de uma palavra (codigos como "lc123") ficam, senao palavras diferentes virariam a mesma.
    """
    palavras = re.sub(r"\b\d+\b", "0", normalizar_texto_auditoria(texto)).split()
    if len(palavras) < ngram:
        return {_hash_estavel(" ".join(palavras)) % int(_PRIMO)} if palavras else set()
    return {
        _hash_estavel(" ".join(palavras[i : i + ngram])) % int(_PRIMO)
        for i in range(len(palavras) - ngram + 1)
    }


def _assinaturas_minhash(conjuntos: list[set[int]], n_perm: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = rng.integers(1, int(_PRIMO), size=n_perm, dtype=np.uint64)
    b = rng.integers(0, int(_PRIMO), size=n_perm, dtype=np.uint64)
    assinaturas = np.full((len(conjuntos), n_perm), np.iinfo(np.uint64).max, dtype=np.uint64)
    for i, conjunto in enumerate(conjuntos):
        if not conjunto:
            continue
        x = np.fromiter(conjunto, dtype=np.uint64, count=len(conjunto))
        # (a*x + b) mod p em uint64: o estouro e aceito, so precisa ser deterministico.
        valores = (np.outer(x, a) + b) % _PRIMO
        assinaturas[i] = valores.min(axis=0)
    return assinaturas


class _UniaoBusca:
    def __init__(self, n: int):
        self.pai = list(range(n))

    def achar(self, x: int) -> int:
        while self.pai[x] != x:
            self.pai[x] = self.pai[self.pai[x]]
            x = self.pai[x]
        return x

    def unir(self, x: int, y: int) -> None:
        rx, ry = self.achar(x), self.achar(y)
        if rx != ry:
            self.pai[max(rx, ry)] = min(rx, ry)


def agrupar_quase_duplicatas(
    processos: list[ProcessoJudicial],
    limiar_jaccard: float = 0.80,
    ngram: int = 5,
    n_perm: int = 128,
    n_bandas: int = 32,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Preenche processo.grupo_duplicata. Processos no mesmo grupo tem ementas com
    Jaccard >= limiar_jaccard (direto ou por encadeamento: A~B e B~C juntam A, B e C).

    32 bandas de 4 linhas acham pares com Jaccard acima de ~0,4 com alta probabilidade,
    bem abaixo do limiar padrao de 0,8; o Jaccard exato descarta os falsos candidatos.

    Returns:
        Estatisticas para o relatorio da Etapa 1.
    """
    n = len(processos)
    if n == 0:
        return {"processos": 0, "grupos": 0, "processos_em_grupos_multiplos": 0, "pares_confirmados": 0}

    conjuntos = [shingles_texto(p.ementa, ngram) for p in processos]
    linhas = n_perm // n_bandas
    assinaturas = _assinaturas_minhash(conjuntos, n_perm, seed)

    candidatos: set[tuple[int, int]] = set()
    for banda in range(n_bandas):
        baldes: dict[bytes, list[int]] = defaultdict(list)
        fatia = assinaturas[:, banda * linhas : (banda + 1) * linhas]
        for i in range(n):
            if conjuntos[i]:
                baldes[fatia[i].tobytes()].append(i)
        for membros in baldes.values():
            if len(membros) < 2:
                continue
            # Balde enorme (ementa-padrao repetida centenas de vezes): liga tudo ao primeiro
            # e confere so esses pares, para nao virar O(n^2).
            if len(membros) > 200:
                candidatos.update((membros[0], m) for m in membros[1:])
            else:
                for x in range(len(membros)):
                    for y in range(x + 1, len(membros)):
                        candidatos.add((membros[x], membros[y]))

    uniao = _UniaoBusca(n)
    confirmados = 0
    # Histograma do Jaccard dos candidatos: mostra se o limiar caiu num vale da distribuicao.
    faixas = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.01]
    hist = Counter()
    for i, j in candidatos:
        a, b = conjuntos[i], conjuntos[j]
        jaccard = len(a & b) / max(1, len(a | b))
        for inicio, fim in zip(faixas, faixas[1:]):
            if inicio <= jaccard < fim:
                hist[f"[{inicio:.2f}, {min(fim, 1.0):.2f}{']' if fim > 1 else ')'}"] += 1
                break
        if jaccard >= limiar_jaccard:
            uniao.unir(i, j)
            confirmados += 1

    # O grupo recebe o menor id_global entre os membros: estavel entre execucoes.
    membros_por_raiz: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        membros_por_raiz[uniao.achar(i)].append(i)
    for membros in membros_por_raiz.values():
        rotulo = min(processos[i].id_global for i in membros)
        for i in membros:
            processos[i].grupo_duplicata = rotulo

    tamanhos = Counter(len(m) for m in membros_por_raiz.values())
    multiplos = [m for m in membros_por_raiz.values() if len(m) > 1]
    return {
        "processos": n,
        "grupos": len(membros_por_raiz),
        "grupos_multiplos": len(multiplos),
        "processos_em_grupos_multiplos": sum(len(m) for m in multiplos),
        "maior_grupo": max(len(m) for m in membros_por_raiz.values()),
        "pares_candidatos": len(candidatos),
        "pares_confirmados": confirmados,
        "distribuicao_tamanho_grupo": {str(k): v for k, v in sorted(tamanhos.items())},
        "histograma_jaccard_candidatos": {k: hist[k] for k in sorted(hist)},
        "limiar_jaccard": limiar_jaccard,
        "ngram": ngram,
    }
