"""
Divisao estratificada por tribunal e agrupada por processo (Train / Val / Test).
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from src.data.schema import ProcessoJudicial


def dividir_treino_val_teste(
    processos: list[ProcessoJudicial],
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    seed: int = 42,
) -> tuple[list[ProcessoJudicial], list[ProcessoJudicial], list[ProcessoJudicial], dict[str, Any]]:
    """
    Divide a lista de processos em Treino, Validacao e Teste com garantia de:
    1. Zero vazamento de dados: O particionamento e feito a nivel de processo judicial
       (id_global unico). Nenhum processo estara em mais de uma particao.
    2. Estratificacao por tribunal: A distribuicao percentual (ex.: 80/10/10) e aplicada
       proporcionalmente dentro de cada tribunal existente na base.

    Args:
        processos: Lista de objetos ProcessoJudicial ja validados e deduplicados.
        train_ratio: Proporcao destinada ao treino (padrao: 0.80).
        val_ratio: Proporcao destinada a validacao intermediaria (padrao: 0.10).
        test_ratio: Proporcao destinada ao teste cego final (padrao: 0.10).
        seed: Semente pseudoaleatoria para reprodutibilidade estrita.

    Returns:
        Tupla com (treino, validacao, teste, dicionario_estatisticas).
    """
    total_ratios = round(train_ratio + val_ratio + test_ratio, 4)
    if total_ratios != 1.0:
        raise ValueError(
            f"O somatorio das proporcoes deve ser 1.0. Valor atual: {total_ratios} "
            f"(train={train_ratio}, val={val_ratio}, test={test_ratio})"
        )

    # Agrupamento dos processos por tribunal de origem
    por_tribunal: dict[str, list[ProcessoJudicial]] = defaultdict(list)
    for proc in processos:
        por_tribunal[proc.tribunal].append(proc)

    rng = random.Random(seed)

    treino: list[ProcessoJudicial] = []
    validacao: list[ProcessoJudicial] = []
    teste: list[ProcessoJudicial] = []

    for tribunal, lista_procs in sorted(por_tribunal.items()):
        # Embaralhamento deterministico por tribunal
        procs_embaralhados = list(lista_procs)
        rng.shuffle(procs_embaralhados)

        n = len(procs_embaralhados)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        # Se o tribunal tiver pouquissimos registros (ex.: menos de 3)
        if n_train == 0 and n > 0:
            n_train = 1

        t_split = procs_embaralhados[:n_train]
        v_split = procs_embaralhados[n_train : n_train + n_val]
        te_split = procs_embaralhados[n_train + n_val :]

        treino.extend(t_split)
        validacao.extend(v_split)
        teste.extend(te_split)

    # Verificacao formal de blindagem contra vazamento de dados
    ids_treino = {p.id_global for p in treino}
    ids_val = {p.id_global for p in validacao}
    ids_teste = {p.id_global for p in teste}

    inter_train_val = ids_treino.intersection(ids_val)
    inter_train_test = ids_treino.intersection(ids_teste)
    inter_val_test = ids_val.intersection(ids_teste)

    if inter_train_val or inter_train_test or inter_val_test:
        raise RuntimeError(
            f"Falha de integridade: Vazamento detectado entre conjuntos! "
            f"Treino x Val: {len(inter_train_val)} | Treino x Teste: {len(inter_train_test)} | Val x Teste: {len(inter_val_test)}"
        )

    estatisticas = {
        "total_processos": len(processos),
        "treino": {
            "total": len(treino),
            "proporcao_real": round(len(treino) / max(1, len(processos)), 4),
            "por_tribunal": dict(Counter(p.tribunal for p in treino)),
        },
        "validacao": {
            "total": len(validacao),
            "proporcao_real": round(len(validacao) / max(1, len(processos)), 4),
            "por_tribunal": dict(Counter(p.tribunal for p in validacao)),
        },
        "teste": {
            "total": len(teste),
            "proporcao_real": round(len(teste) / max(1, len(processos)), 4),
            "por_tribunal": dict(Counter(p.tribunal for p in teste)),
        },
        "vazamento_detectado": False,
    }

    return treino, validacao, teste, estatisticas


def salvar_splits_processos(
    diretorio_destino: Path | str,
    treino: list[ProcessoJudicial],
    validacao: list[ProcessoJudicial],
    teste: list[ProcessoJudicial],
) -> dict[str, Path]:
    """
    Persiste os processos particionados em arquivos JSONL independentes no disco.
    """
    pasta = Path(diretorio_destino)
    pasta.mkdir(parents=True, exist_ok=True)

    caminhos = {
        "treino": pasta / "train_processos.jsonl",
        "validacao": pasta / "val_processos.jsonl",
        "teste": pasta / "test_processos.jsonl",
    }

    for conjunto, lista_procs in [("treino", treino), ("validacao", validacao), ("teste", teste)]:
        destino = caminhos[conjunto]
        with destino.open("w", encoding="utf-8") as f:
            for p in lista_procs:
                f.write(json.dumps(p.to_dict(), ensure_ascii=False) + "\n")

    return caminhos
