"""
Configuracao de dataset, funcao de perda (CachedMNRL) e SentenceTransformerTrainer.
"""

from __future__ import annotations

import json
import logging
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


def carregar_dataset_tripletos(
    caminho_arquivo: Path | str,
    n_negativos: int = 5,
    query_instruction: str = "Instruct: Dado um trecho de um documento jurídico brasileiro, recupere o trecho mais relevante semanticamente.\nQuery: ",
) -> Dataset:
    """
    Le o arquivo de tripletos minerados e monta o Dataset HuggingFace estruturado
    nas colunas exigidas pela CachedMultipleNegativesRankingLoss:
    ['anchor', 'positive', 'negative_0', ..., 'negative_4']
    """
    caminho = Path(caminho_arquivo)
    if not caminho.is_file():
        raise FileNotFoundError(f"Arquivo de tripletos nao encontrado: {caminho}")

    colunas = ["anchor", "positive"] + [f"negative_{i}" for i in range(n_negativos)]
    dados: dict[str, list[str]] = {col: [] for col in colunas}

    lidos = 0
    validos = 0
    descartados = 0

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

            dados["anchor"].append(anchor_texto)
            dados["positive"].append(positive)
            for i in range(n_negativos):
                dados[f"negative_{i}"].append(negatives[i])

            validos += 1

    if validos == 0:
        raise ValueError(f"Nenhum tripleto valido com {n_negativos} negativos foi encontrado em {caminho}")

    dataset = Dataset.from_dict(dados)
    dataset = dataset.select_columns(colunas)

    print(f"Dataset de Treino Carregado: {validos:,} tripletos validos (descartados: {descartados:,})")
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
    bf16: bool = True,
    dataloader_num_workers: int = 8,
    dataloader_pin_memory: bool = True,
    scale: float = 50.0,
    evaluator: Optional[Any] = None,
    metric_for_best_model: str = "val_cosine_ndcg@10",
    greater_is_better: bool = True,
) -> tuple[SentenceTransformerTrainer, ScientificMetricsCallback]:
    """
    Instancia a CachedMultipleNegativesRankingLoss e configura o SentenceTransformerTrainer
    com argumentos de alto desempenho otimizados para GPU H200 (141 GB).
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. Funcao de Perda Contrastiva (CachedMNRL)
    loss = CachedMultipleNegativesRankingLoss(
        model=model,
        mini_batch_size=mini_batch_size,
        scale=scale,
    )

    # 2. Calculo de steps e warmup
    passos_por_epoca = max(1, len(train_dataset) // (batch_size * gradient_accumulation_steps))
    total_steps = passos_por_epoca * num_epochs
    warmup_steps = int(total_steps * warmup_ratio)

    tem_avaliador = evaluator is not None

    # Ajuste dinamico de steps se o total_steps for menor que eval_steps
    eval_steps_efetivo = eval_steps
    save_steps_efetivo = save_steps
    if tem_avaliador and total_steps < eval_steps:
        eval_steps_efetivo = max(5, total_steps // 2)
        save_steps_efetivo = eval_steps_efetivo

    print("Configuracao do Treinamento:")
    print(f"  - Batch Size Real          : {batch_size}")
    print(f"  - Mini Batch Size (Loss)   : {mini_batch_size}")
    print(f"  - Acumulacao de Gradientes : {gradient_accumulation_steps} (Batch efetivo: {batch_size * gradient_accumulation_steps})")
    print(f"  - Total Steps Estimados    : {total_steps:,} | Warmup: {warmup_steps:,}")
    print(f"  - Taxa de Aprendizado (LR) : {learning_rate}")
    print(f"  - Precisao bfloat16        : {bf16}")
    print(f"  - Dataloader Workers       : {dataloader_num_workers} | Pin Memory: {dataloader_pin_memory}")
    if tem_avaliador:
        print(f"  - Avaliador de Validacao   : Ativado (a cada {eval_steps_efetivo} steps)")
        print(f"  - Metrica de Selecao       : {metric_for_best_model} (load_best_model_at_end=True)")
    else:
        print(f"  - Avaliador de Validacao   : Desativado (load_best_model_at_end=False)")

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
        bf16=bf16 and torch.cuda.is_available(),
        fp16=False,
        eval_strategy="steps" if tem_avaliador else "no",
        eval_steps=eval_steps_efetivo if tem_avaliador else None,
        save_strategy="steps",
        save_steps=save_steps_efetivo,
        save_total_limit=2,
        load_best_model_at_end=tem_avaliador,
        metric_for_best_model=metric_for_best_model if tem_avaliador else None,
        greater_is_better=greater_is_better if tem_avaliador else None,
        logging_steps=10,
        report_to="none",
        dataloader_num_workers=dataloader_num_workers,
        dataloader_pin_memory=dataloader_pin_memory,
        remove_unused_columns=False,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
    )

    # 4. Callback de Telemetria Cientifica
    telemetry_callback = ScientificMetricsCallback(
        output_dir=output_path,
        nome_modelo=model[0].auto_model.config.name_or_path if hasattr(model[0], "auto_model") else "Qwen3-4B",
    )

    # 5. Trainer
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        loss=loss,
        evaluator=evaluator,
        callbacks=[telemetry_callback],
    )

    return trainer, telemetry_callback
