"""
Configuracao de dataset, funcao de perda (CachedMNRL) e SentenceTransformerTrainer.
"""

from __future__ import annotations

import inspect
import json
import logging
import os
from pathlib import Path
from typing import Any, Optional

import torch
from datasets import Dataset
from sentence_transformers import SentenceTransformer
from sentence_transformers.losses import CachedMultipleNegativesRankingLoss
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import BatchSamplers, SentenceTransformerTrainingArguments

from src.training.telemetry import ScientificMetricsCallback


log = logging.getLogger("bumbalm_trainer")


def _processo_principal() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def _imprimir(mensagem: str) -> None:
    if _processo_principal():
        print(mensagem)


def _num_processos() -> int:
    return max(1, int(os.environ.get("WORLD_SIZE", "1")))


def chave_tupla(item: dict[str, Any]) -> str:
    """Identifica o par (processo, trecho positivo). E a mesma nos arquivos de todas as variantes."""
    positivo = item.get("positive_chunk_id")
    if positivo is None:
        positivo = item.get("positive_chunk_idx", "")
    return f"{item.get('id_global', '')}|{positivo}"


def _tupla_valida(item: dict[str, Any], n_negativos: int) -> bool:
    query = str(item.get("query", "")).strip()
    positive = str(item.get("positive", "")).strip()
    negatives = [n for n in item.get("negatives", []) or [] if str(n).strip()]
    return bool(query and positive and len(negatives) >= n_negativos)


def _assinatura_arquivos(caminhos: list[Path]) -> str:
    return ";".join(f"{p.name}:{p.stat().st_size}" for p in caminhos)


def sortear_chaves_tuplas(
    arquivos_referencia: list[Path | str],
    limite_tuplas: int,
    seed: int = 42,
    n_negativos: int = 5,
    arquivo_cache: Optional[Path | str] = None,
) -> list[str]:
    """
    Sorteia limite_tuplas pares (processo, trecho positivo) validos em TODOS os arquivos de referencia
    (tuplas dificeis e aleatorias). As tres variantes treinam com exatamente estes pares, mesmo que algum
    arquivo tenha perdido uma tupla ou mudado a ordem das linhas. O sorteio parte da lista ordenada de
    chaves, entao nao depende da ordem das linhas nos arquivos.

    arquivo_cache: grava a lista (uma chave por linha, com cabecalho) e a reaproveita nas outras
    variantes e nas retomadas, desde que os arquivos de referencia nao tenham mudado de tamanho.
    """
    import random

    caminhos = [Path(c) for c in arquivos_referencia]
    for caminho in caminhos:
        if not caminho.is_file():
            raise FileNotFoundError(f"Arquivo de tuplas nao encontrado: {caminho}")
    cabecalho = f"# limite={int(limite_tuplas)} seed={seed} n_negativos={n_negativos} arquivos={_assinatura_arquivos(caminhos)}"

    if arquivo_cache is not None and Path(arquivo_cache).is_file():
        linhas = Path(arquivo_cache).read_text(encoding="utf-8").splitlines()
        if linhas and linhas[0] == cabecalho:
            chaves = [l for l in linhas[1:] if l.strip()]
            _imprimir(f"Tuplas sorteadas lidas de {arquivo_cache} ({len(chaves):,} pares)")
            return chaves
        _imprimir(f"Aviso: {arquivo_cache} foi gerado com outros parametros ou arquivos. Sorteando de novo.")

    comuns: Optional[set[str]] = None
    for caminho in caminhos:
        validas: set[str] = set()
        with caminho.open(encoding="utf-8") as f:
            for linha in f:
                linha = linha.strip()
                if not linha:
                    continue
                item = json.loads(linha)
                if _tupla_valida(item, n_negativos):
                    validas.add(chave_tupla(item))
        _imprimir(f"  - {caminho.name}: {len(validas):,} pares validos")
        comuns = validas if comuns is None else comuns & validas

    disponiveis = sorted(comuns or set())
    if not disponiveis:
        raise ValueError("Nenhum par valido em comum entre os arquivos de tuplas.")
    n = min(int(limite_tuplas), len(disponiveis))
    chaves = random.Random(seed).sample(disponiveis, n)
    _imprimir(f"Sorteio das tuplas: {n:,} de {len(disponiveis):,} pares em comum (semente {seed})")

    if arquivo_cache is not None and _processo_principal():
        destino = Path(arquivo_cache)
        destino.parent.mkdir(parents=True, exist_ok=True)
        temporario = destino.with_name(destino.name + ".tmp")
        temporario.write_text(cabecalho + "\n" + "\n".join(chaves) + "\n", encoding="utf-8")
        temporario.replace(destino)
    return chaves


def hash_chaves_tuplas(chaves: list[str]) -> str:
    """SHA-256 da lista de pares sorteados. Tem que ser igual nas tres variantes."""
    import hashlib

    return hashlib.sha256("\n".join(chaves).encode("utf-8")).hexdigest()


def carregar_dataset_tripletos(
    caminho_arquivo: Path | str,
    n_negativos: int = 5,
    query_instruction: str = "Instruct: Dado um trecho de um documento jurídico brasileiro, recupere o trecho mais relevante semanticamente.\nQuery: ",
    limite_tuplas: Optional[int] = None,
    seed: int = 42,
    chaves: Optional[list[str]] = None,
) -> Dataset:
    """
    Le o arquivo de tripletos minerados e monta o Dataset HuggingFace estruturado
    nas colunas exigidas pela CachedMultipleNegativesRankingLoss:
    ['anchor', 'positive', 'negative_0', ..., 'negative_4']

    Com n_negativos=0 restam ['anchor', 'positive']: a loss usa so os negativos do proprio lote.
    chaves: lista de pares de sortear_chaves_tuplas. Carrega so esses pares, na ordem da lista,
    o que garante os mesmos pares e a mesma ordem nas tres variantes.
    limite_tuplas (sem chaves): sorteio antigo pela posicao das linhas, que so pega os mesmos pares
    se os arquivos das variantes tiverem as mesmas linhas na mesma ordem.
    """
    caminho = Path(caminho_arquivo)
    if not caminho.is_file():
        raise FileNotFoundError(f"Arquivo de tripletos nao encontrado: {caminho}")

    colunas = ["anchor", "positive"] + [f"negative_{i}" for i in range(n_negativos)]
    dados: dict[str, list[str]] = {col: [] for col in colunas}

    lidos = 0
    validos = 0
    descartados = 0
    posicao_chave = {c: i for i, c in enumerate(chaves)} if chaves is not None else None
    selecionadas: dict[int, tuple[str, str, list[str]]] = {}

    with caminho.open(encoding="utf-8") as f:
        for linha in f:
            linha = linha.strip()
            if not linha:
                continue
            lidos += 1
            item = json.loads(linha)

            query = str(item.get("query", "")).strip()
            positive = str(item.get("positive", "")).strip()
            negatives = [str(n).strip() for n in item.get("negatives", []) if str(n).strip()]

            if not query or not positive or len(negatives) < n_negativos:
                descartados += 1
                continue

            # Aplica a instrucao formal na query se necessario
            anchor_texto = query if query.startswith("Instruct:") else f"{query_instruction}{query}"

            if posicao_chave is not None:
                posicao = posicao_chave.get(chave_tupla(item))
                if posicao is not None and posicao not in selecionadas:
                    selecionadas[posicao] = (anchor_texto, positive, negatives[:n_negativos])
                validos += 1
                continue

            dados["anchor"].append(anchor_texto)
            dados["positive"].append(positive)
            for i in range(n_negativos):
                dados[f"negative_{i}"].append(negatives[i])

            validos += 1

    if validos == 0:
        raise ValueError(f"Nenhum tripleto valido com {n_negativos} negativos foi encontrado em {caminho}")

    if posicao_chave is not None:
        faltando = len(posicao_chave) - len(selecionadas)
        if faltando:
            raise ValueError(f"{faltando:,} pares sorteados nao estao em {caminho}. Refaca o sorteio.")
        for posicao in range(len(posicao_chave)):
            anchor_texto, positive, negatives = selecionadas[posicao]
            dados["anchor"].append(anchor_texto)
            dados["positive"].append(positive)
            for i in range(n_negativos):
                dados[f"negative_{i}"].append(negatives[i])

    import pyarrow as pa
    schema_large = pa.schema([pa.field(col, pa.large_string()) for col in colunas])
    tabela_arrow = pa.Table.from_pydict(dados, schema=schema_large)
    dataset = Dataset(tabela_arrow)

    _imprimir(f"Dataset de Treino Carregado: {validos:,} tripletos validos (descartados: {descartados:,})")
    if posicao_chave is not None:
        _imprimir(f"  - Pares sorteados         : {len(dataset):,} (mesma lista nas tres variantes)")
    elif limite_tuplas and len(dataset) > limite_tuplas:
        dataset = dataset.shuffle(seed=seed).select(range(int(limite_tuplas)))
        _imprimir(f"  - Subconjunto sorteado    : {len(dataset):,} tuplas (semente {seed})")
    _imprimir(f"  - Colunas                 : {dataset.column_names}")
    return dataset


def configurar_treinador(
    model: SentenceTransformer,
    train_dataset: Dataset,
    output_dir: Path | str,
    batch_size: int = 96,
    mini_batch_size: int = 96,
    gradient_accumulation_steps: int = 1,
    learning_rate: float = 2.0e-5,
    weight_decay: float = 0.01,
    num_epochs: int = 1,
    warmup_ratio: float = 0.05,
    eval_steps: int = 250,
    save_steps: int = 250,
    precisao_mista: str = "bf16",
    dataloader_num_workers: int = 4,
    dataloader_pin_memory: bool = False,
    scale: float = 50.0,
    evaluator: Optional[Any] = None,
    metric_for_best_model: str = "val_cosine_ndcg@10",
    greater_is_better: bool = True,
    save_total_limit: int = 2,
    carregar_melhor_no_fim: bool = True,
    logging_steps: int = 10,
    salvar_checkpoints: bool = True,
    seed: int = 42,
) -> tuple[SentenceTransformerTrainer, ScientificMetricsCallback]:
    """
    Instancia a CachedMultipleNegativesRankingLoss e configura o SentenceTransformerTrainer.

    precisao_mista: "bf16" (A100/H200), "fp16" (V100) ou "no".
    carregar_melhor_no_fim: com LoRA fica False. O melhor checkpoint e lido depois pelo caminho
    gravado em trainer.state.best_model_checkpoint.
    salvar_checkpoints: False nao grava checkpoint nenhum (diagnostico: cada um do full fine-tuning tem 24 GB).
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    num_processos = _num_processos()
    juntar_gpus = num_processos > 1

    # 1. Funcao de Perda Contrastiva (CachedMNRL)
    kwargs_loss: dict[str, Any] = {
        "model": model,
        "mini_batch_size": mini_batch_size,
        "scale": scale,
    }
    if juntar_gpus:
        parametros_loss = inspect.signature(CachedMultipleNegativesRankingLoss.__init__).parameters
        if "gather_across_devices" in parametros_loss:
            kwargs_loss["gather_across_devices"] = True
        else:
            juntar_gpus = False
            _imprimir(
                "Aviso: esta versao do Sentence Transformers nao aceita gather_across_devices. "
                "Cada GPU usara apenas os negativos do proprio lote."
            )
    loss = CachedMultipleNegativesRankingLoss(**kwargs_loss)

    # 2. Calculo de steps e warmup
    lote_passo = batch_size * gradient_accumulation_steps * num_processos
    passos_por_epoca = max(1, len(train_dataset) // lote_passo)
    total_steps = passos_por_epoca * num_epochs
    # Pelo menos 1 passo de warmup: no smoke, int(5 * 0.05) dava 0 e o 1o passo ja usava o LR cheio
    warmup_steps = max(1, int(total_steps * warmup_ratio)) if warmup_ratio > 0 else 0

    tem_avaliador = evaluator is not None

    # Ajuste dinamico de steps se o total_steps for menor que eval_steps
    eval_steps_efetivo = eval_steps
    save_steps_efetivo = save_steps
    if tem_avaliador and total_steps < eval_steps:
        eval_steps_efetivo = max(5, total_steps // 2)
        save_steps_efetivo = eval_steps_efetivo

    _imprimir("Configuracao do Treinamento:")
    _imprimir(f"  - Batch por GPU            : {batch_size}")
    _imprimir(f"  - Processos (GPUs)         : {num_processos}")
    _imprimir(f"  - Mini Batch Size (Loss)   : {mini_batch_size}")
    _imprimir(f"  - Acumulacao de Gradientes : {gradient_accumulation_steps} (Batch do passo: {lote_passo})")
    _imprimir(f"  - Negativos entre GPUs     : {juntar_gpus}")
    _imprimir(f"  - Total Steps Estimados    : {total_steps:,} | Warmup: {warmup_steps:,}")
    _imprimir(f"  - Taxa de Aprendizado (LR) : {learning_rate}")
    _imprimir(f"  - Precisao mista           : {precisao_mista}")
    _imprimir(f"  - Dataloader Workers       : {dataloader_num_workers} | Pin Memory: {dataloader_pin_memory}")
    if tem_avaliador:
        _imprimir(f"  - Avaliador de Validacao   : Ativado (a cada {eval_steps_efetivo} steps)")
        _imprimir(f"  - Metrica de Selecao       : {metric_for_best_model} (load_best_model_at_end=True)")
    else:
        _imprimir("  - Avaliador de Validacao   : Desativado (load_best_model_at_end=False)")

    # 3. Argumentos do Trainer
    training_args = SentenceTransformerTrainingArguments(
        output_dir=str(output_path),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        lr_scheduler_type="cosine",
        bf16=precisao_mista == "bf16" and torch.cuda.is_available(),
        fp16=precisao_mista == "fp16" and torch.cuda.is_available(),
        eval_strategy="steps" if tem_avaliador else "no",
        eval_steps=eval_steps_efetivo if tem_avaliador else None,
        save_strategy="steps" if salvar_checkpoints else "no",
        save_steps=save_steps_efetivo,
        save_total_limit=save_total_limit,
        load_best_model_at_end=tem_avaliador and carregar_melhor_no_fim and salvar_checkpoints,
        metric_for_best_model=metric_for_best_model if tem_avaliador else None,
        greater_is_better=greater_is_better if tem_avaliador else None,
        logging_steps=logging_steps,
        # Loss e grad_norm do 1o passo, ainda calculados com os pesos do modelo base
        logging_first_step=True,
        report_to="none",
        dataloader_num_workers=dataloader_num_workers,
        dataloader_pin_memory=dataloader_pin_memory,
        remove_unused_columns=False,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        seed=seed,
    )

    # 4. Callback de Telemetria Cientifica
    telemetry_callback = ScientificMetricsCallback(
        output_dir=output_path,
        nome_modelo=model[0].auto_model.config.name_or_path if hasattr(model[0], "auto_model") else "Qwen3-4B",
    )

    # 5. Trainer. Telemetria e gravacao de metricas ficam no processo 0.
    callbacks = [telemetry_callback] if _processo_principal() else []
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        loss=loss,
        evaluator=evaluator,
        callbacks=callbacks,
    )

    return trainer, telemetry_callback
