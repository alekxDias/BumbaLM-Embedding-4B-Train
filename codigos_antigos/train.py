#!/usr/bin/env python3
"""
Experimento Exploratorio: Fine-Tuning do Qwen3-Embedding-8B.

Este script faz parte do acervo de experimentos iniciais do projeto BumbaLM,
sendo a versao exploratoria desenvolvida para o modelo de 8 bilhoes de parametros
(Qwen/Qwen3-Embedding-8B), anteriormente a padronizacao final no modelo de 4B
(Qwen/Qwen3-Embedding-4B) apresentada no artigo.

Objetivo:
1. Avaliar a viabilidade e consumo de memoria do Full Fine-Tuning do modelo 8B
   utilizando a GPU NVIDIA H200 (141 GB).
2. Treinamento contrastivo com a funcao CachedMultipleNegativesRankingLoss.
3. Telemetria completa de hardware (VRAM, RAM) e metricas de IR (nDCG@10, MRR@10).

Uso:
    python train.py
"""

from __future__ import annotations

import csv
import gc
import json
import logging
import os
import random
import sys
import time
from collections import defaultdict

import torch
from datasets import Dataset
from sentence_transformers import SentenceTransformer, evaluation, models
from sentence_transformers.losses import CachedMultipleNegativesRankingLoss
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import BatchSamplers, SentenceTransformerTrainingArguments
from sentence_transformers.util import cos_sim
from transformers import TrainerCallback


# 1. Configuracao do Diretorio Base e Logging
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(LOG_DIR, "train_8b_gpu1.log")),
    ],
    force=True,
)
log = logging.getLogger("bumbalm_train_8b")

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HOME"] = os.path.join(BASE_DIR, "hf_cache")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(BASE_DIR, "hf_cache")

torch.backends.cuda.matmul.allow_tf32 = True

log.info(f"PyTorch: {torch.__version__} | CUDA: {torch.version.cuda}")
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    log.info(f"GPU detectada: {torch.cuda.get_device_name(0)} | VRAM Total: {props.total_memory / 1e9:.1f} GB")
else:
    log.warning("CUDA nao disponivel no ambiente.")


# 2. Hiperparametros e Configuracao do Modelo 8B
CFG = {
    # Datasets
    "TRAIN_TRIPLETS_FILE": os.getenv(
        "BUMBA_TRAIN_TRIPLETS",
        os.path.join(BASE_DIR, "tripletos_train.jsonl"),
    ),
    "VAL_PAIRS_FILE": os.getenv(
        "BUMBA_VAL_PAIRS",
        os.path.join(BASE_DIR, "tripletos_val.jsonl"),
    ),
    "OUTPUT_PATH": os.getenv(
        "BUMBA_OUTPUT_PATH",
        os.path.join(BASE_DIR, "modelo_treinado_8b"),
    ),

    # Modelo Base (8B)
    "MODEL_ID": "Qwen/Qwen3-Embedding-8B",
    "MAX_SEQ_LEN": 768,
    "QUERY_INSTRUCTION": (
        "Instruct: Dado um trecho de um documento juridico brasileiro, "
        "recupere o trecho mais relevante semanticamente.\nQuery: "
    ),

    # Tripletos
    "N_HARD_NEG": 5,
    "PRINT_BATCH_DEBUG": False,
    "DEBUG_CHARS": 300,

    # Parametros de Treinamento
    "BATCH_SIZE_TRAIN": 16,
    "MINI_BATCH_SIZE": 16,
    "GRAD_ACCUMULATION": 4,      # Batch efetivo = 16 x 4 = 64 amostras/step
    "EPOCHS": 1,
    "WARMUP_RATIO": 0.05,
    "LR": 2e-5,
    "WEIGHT_DECAY": 0.01,
    "EVAL_STEPS": 250,
    "SAVE_STEPS": 250,
    "BF16": True,
    "GRADIENT_CHECKPOINTING": True,
    "SEED": 42,
}

os.makedirs(CFG["OUTPUT_PATH"], exist_ok=True)
os.makedirs(os.environ["HF_HOME"], exist_ok=True)
os.makedirs(os.environ["TRANSFORMERS_CACHE"], exist_ok=True)
log.info(f"Diretorio de saida do modelo 8B: {CFG['OUTPUT_PATH']}")


def read_jsonl(path: str) -> list[dict]:
    """Le um arquivo JSONL."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Arquivo nao encontrado: {path}")

    rows = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSON invalido em {path}, linha {line_no}") from exc
    return rows


def text_or_empty(value) -> str:
    """Retorna string limpa de espacos."""
    return str(value or "").strip()


def load_train_triplets(path: str, n_neg: int) -> list[dict]:
    """Carrega os tripletos de treino para o modelo 8B."""
    rows = read_jsonl(path)
    triplets = []
    skipped = 0

    for i, item in enumerate(rows):
        query = text_or_empty(item.get("query"))
        positive = text_or_empty(item.get("positive"))
        negatives = [text_or_empty(n) for n in item.get("negatives", [])[:n_neg]]
        negatives = [n for n in negatives if n]

        if not query or not positive or len(negatives) < n_neg:
            skipped += 1
            continue

        triplets.append(
            {
                "doc_id": text_or_empty(item.get("str_numero_unico")) or f"train_{i}",
                "query": query,
                "positive": positive,
                "negatives": negatives[:n_neg],
            }
        )

    if not triplets:
        raise ValueError("Nenhum tripleto valido encontrado no arquivo de treino.")

    log.info(f"Tripletos lidos: {len(rows):,} | validos: {len(triplets):,} | descartados: {skipped:,}")
    return triplets


def build_train_dataset(triplets: list[dict], n_neg: int) -> Dataset:
    """Monta o dataset HuggingFace para a CachedMultipleNegativesRankingLoss."""
    columns = ["anchor", "positive"] + [f"negative_{i}" for i in range(n_neg)]
    data = {col: [] for col in columns}

    for item in triplets:
        anchor_text = item["query"]
        if not anchor_text.startswith("Instruct:"):
            anchor_text = CFG["QUERY_INSTRUCTION"] + anchor_text
        data["anchor"].append(anchor_text)
        data["positive"].append(item["positive"])
        for i in range(n_neg):
            data[f"negative_{i}"].append(item["negatives"][i])

    dataset = Dataset.from_dict(data)
    dataset = dataset.select_columns(columns)
    log.info(f"Dataset de treino 8B: {len(dataset):,} amostras")
    return dataset


def build_evaluator() -> evaluation.InformationRetrievalEvaluator:
    """Monta o avaliador IR sobre o conjunto isolado de validacao."""
    val_rows = read_jsonl(CFG["VAL_PAIRS_FILE"])
    if not val_rows:
        raise ValueError("Nenhum registro de validacao encontrado.")

    query_map = defaultdict(list)
    for item in val_rows:
        q = text_or_empty(item.get("query"))
        pos = text_or_empty(item.get("positive"))
        negs = [text_or_empty(n) for n in item.get("negatives", []) if text_or_empty(n)]
        doc_id = text_or_empty(item.get("str_numero_unico"))
        if q and pos:
            query_map[q].append({"positive": pos, "negatives": negs, "doc_id": doc_id})

    queries_eval = {}
    relevant_docs = defaultdict(set)
    corpus_eval = {}

    for q_idx, (q_text, items) in enumerate(query_map.items()):
        qid = f"val_q_{q_idx}"
        queries_eval[qid] = q_text if q_text.startswith("Instruct:") else CFG["QUERY_INSTRUCTION"] + q_text

        for item_idx, it in enumerate(items):
            pos_id = f"pos_{q_idx}_{item_idx}"
            corpus_eval[pos_id] = it["positive"]
            relevant_docs[qid].add(pos_id)

            for neg_idx, neg_text in enumerate(it["negatives"]):
                neg_id = f"neg_{q_idx}_{item_idx}_{neg_idx}"
                corpus_eval[neg_id] = neg_text

    log.info(
        f"Avaliador IR 8B configurado: {len(queries_eval):,} queries | {len(corpus_eval):,} docs no corpus"
    )

    return evaluation.InformationRetrievalEvaluator(
        queries=queries_eval,
        corpus=corpus_eval,
        relevant_docs=dict(relevant_docs),
        name="tripletos_val_eval_8b",
        show_progress_bar=True,
        batch_size=32,
        score_functions={"cosine": cos_sim},
        main_score_function="cosine",
        mrr_at_k=[10],
        ndcg_at_k=[10],
        accuracy_at_k=[1, 5, 10],
        map_at_k=[100],
    )


class ScientificMetricsCallback(TrainerCallback):
    """Callback para registro de metricas e telemetria de VRAM/RAM para o modelo 8B."""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.start_time = time.time()
        self.metrics_history = []
        self.json_summary_path = os.path.join(output_dir, "relatorio_treinamento_8b.json")
        self.csv_path = os.path.join(output_dir, "metricas_treino_val_8b.csv")

    def _get_ram_gb(self) -> float:
        try:
            import psutil
            return round(psutil.Process(os.getpid()).memory_info().rss / (1024 ** 3), 2)
        except Exception:
            return 0.0

    def _get_vram_gb(self) -> tuple[float, float]:
        if torch.cuda.is_available():
            allocated = round(torch.cuda.max_memory_allocated() / (1024 ** 3), 2)
            reserved = round(torch.cuda.max_memory_reserved() / (1024 ** 3), 2)
            return allocated, reserved
        return 0.0, 0.0

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        vram_alloc, vram_res = self._get_vram_gb()
        ram_gb = self._get_ram_gb()
        elapsed = round(time.time() - self.start_time, 2)

        entry = {
            "step": state.global_step,
            "epoch": round(state.epoch, 4) if state.epoch else 0.0,
            "elapsed_sec": elapsed,
            "vram_allocated_gb": vram_alloc,
            "vram_reserved_gb": vram_res,
            "ram_gb": ram_gb,
        }
        for k, v in logs.items():
            if isinstance(v, (int, float)):
                entry[k] = round(v, 6)

        self.metrics_history.append(entry)

    def on_train_end(self, args, state, control, **kwargs):
        total_time_sec = round(time.time() - self.start_time, 2)
        vram_alloc, vram_res = self._get_vram_gb()
        ram_gb = self._get_ram_gb()

        resumo = {
            "modelo": CFG["MODEL_ID"],
            "total_steps": state.global_step,
            "epochs": CFG["EPOCHS"],
            "tempo_total_segundos": total_time_sec,
            "tempo_total_formatado": f"{int(total_time_sec // 3600)}h {int((total_time_sec % 3600) // 60)}m {int(total_time_sec % 60)}s",
            "vram_pico_alocada_gb": vram_alloc,
            "vram_pico_reservada_gb": vram_res,
            "ram_pico_gb": ram_gb,
            "configuracao": CFG,
            "historico_metricas": self.metrics_history,
        }

        with open(self.json_summary_path, "w", encoding="utf-8") as f:
            json.dump(resumo, f, indent=2, ensure_ascii=False)

        try:
            if self.metrics_history:
                all_keys = set()
                for h in self.metrics_history:
                    all_keys.update(h.keys())
                fieldnames = sorted(list(all_keys))

                with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(self.metrics_history)
                log.info(f"Historico de metricas exportado em CSV tabular: {self.csv_path}")
        except Exception as e:
            log.warning(f"Aviso ao salvar CSV de metricas: {e}")

        log.info(f"Relatorio salvo em: {self.json_summary_path}")
        log.info(f"Tempo total: {resumo['tempo_total_formatado']} | Pico VRAM: {vram_alloc} GB")


def main() -> None:
    """Execucao principal do treinamento exploratorio 8B."""
    random.seed(CFG["SEED"])
    torch.manual_seed(CFG["SEED"])

    # 1. Carregar Tripletos de Treino
    log.info(f"Carregando dados de treino de: {CFG['TRAIN_TRIPLETS_FILE']}")
    train_triplets = load_train_triplets(CFG["TRAIN_TRIPLETS_FILE"], CFG["N_HARD_NEG"])
    hf_train_dataset = build_train_dataset(train_triplets, CFG["N_HARD_NEG"])

    # 2. Construir Modelo de 8B
    log.info(f"Construindo modelo base: {CFG['MODEL_ID']} (Full Fine-Tuning)...")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    word_embedding_model = models.Transformer(
        model_name_or_path=CFG["MODEL_ID"],
        max_seq_length=CFG["MAX_SEQ_LEN"],
        tokenizer_args={"padding_side": "left"},
        model_args={
            "torch_dtype": torch.bfloat16,
            "trust_remote_code": True,
        },
    )
    word_embedding_model.auto_model.config.use_cache = False

    if CFG["GRADIENT_CHECKPOINTING"]:
        word_embedding_model.auto_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        log.info("Gradient Checkpointing ativado.")

    for param in word_embedding_model.auto_model.parameters():
        param.requires_grad = True

    pooling_model = models.Pooling(
        word_embedding_model.get_word_embedding_dimension(),
        pooling_mode="lasttoken",
    )
    normalize_model = models.Normalize()

    model = SentenceTransformer(
        modules=[word_embedding_model, pooling_model, normalize_model],
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    total_params = sum(p.numel() for p in word_embedding_model.auto_model.parameters())
    trainable_params = sum(p.numel() for p in word_embedding_model.auto_model.parameters() if p.requires_grad)
    log.info(
        f"Modelo 8B montado: {trainable_params:,} parametros treinaveis de {total_params:,}"
    )

    # 3. Configurar Avaliador IR
    log.info("Configurando avaliador IR de validacao...")
    evaluator = build_evaluator()

    log.info("Avaliando modelo base 8B antes do fine-tuning...")
    base_score = evaluator(model, output_path=CFG["OUTPUT_PATH"], epoch=0, steps=0)
    log.info(f"Baseline do modelo 8B: {base_score}")

    num_steps = max(1, len(hf_train_dataset) // (CFG["BATCH_SIZE_TRAIN"] * CFG["GRAD_ACCUMULATION"]))
    warmup_steps = int(num_steps * CFG["WARMUP_RATIO"])
    log.info(f"Steps estimados: {num_steps:,} | Warmup: {warmup_steps:,}")

    # 4. Funcao de Perda Contrastiva
    train_loss = CachedMultipleNegativesRankingLoss(
        model=model,
        mini_batch_size=CFG["MINI_BATCH_SIZE"],
        scale=50.0,
    )

    # 5. Argumentos de Treinamento
    training_args = SentenceTransformerTrainingArguments(
        output_dir=CFG["OUTPUT_PATH"],
        num_train_epochs=CFG["EPOCHS"],
        per_device_train_batch_size=CFG["BATCH_SIZE_TRAIN"],
        gradient_accumulation_steps=CFG["GRAD_ACCUMULATION"],
        learning_rate=CFG["LR"],
        weight_decay=CFG["WEIGHT_DECAY"],
        warmup_steps=warmup_steps,
        lr_scheduler_type="cosine",
        bf16=CFG["BF16"] and torch.cuda.is_available(),
        fp16=False,
        eval_strategy="steps",
        eval_steps=CFG["EVAL_STEPS"],
        save_strategy="steps",
        save_steps=CFG["SAVE_STEPS"],
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="tripletos_val_eval_8b_cosine_ndcg@10",
        greater_is_better=True,
        logging_steps=10,
        report_to="none",
        remove_unused_columns=False,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
    )

    metrics_callback = ScientificMetricsCallback(CFG["OUTPUT_PATH"])

    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=hf_train_dataset,
        loss=train_loss,
        evaluator=evaluator,
        callbacks=[metrics_callback],
    )

    # 6. Execucao
    log.info("Iniciando treinamento do modelo 8B...")
    trainer.train()

    best_path = os.path.join(CFG["OUTPUT_PATH"], "best_model_8b")
    model.save(best_path)
    log.info(f"Treinamento 8B finalizado. Modelo salvo em: {best_path}")


if __name__ == "__main__":
    main()