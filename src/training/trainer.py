"""
Configuracao de dataset, funcao de perda (CachedMNRL) e SentenceTransformerTrainer.
"""

from __future__ import annotations

import inspect
import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any, Iterator, Optional

import torch
from datasets import Dataset, Features, Value
from datasets.arrow_writer import ArrowWriter
from sentence_transformers import SentenceTransformer
from sentence_transformers.losses import CachedMultipleNegativesRankingLoss
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import BatchSamplers, SentenceTransformerTrainingArguments
from tqdm import tqdm

from src.training.telemetry import ScientificMetricsCallback


log = logging.getLogger("bumbalm_trainer")

# Lotes pequenos na conversao.
_LOTE_CONVERSAO_ARROW = 2048
_ESPERA_CACHE_SEGUNDOS = 5.0


def _processo_principal() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def _imprimir(mensagem: str) -> None:
    if _processo_principal():
        print(mensagem)


def _num_processos() -> int:
    return max(1, int(os.environ.get("WORLD_SIZE", "1")))


def _garantir_grupo_processo() -> None:
    """
    O torchrun define RANK/WORLD_SIZE, mas o process group so nasce no Trainer.
    Inicia aqui para o barrier funcionar antes da conversao do JSONL.
    """
    if _num_processos() <= 1:
        return
    if not torch.distributed.is_available():
        return
    if torch.distributed.is_initialized():
        return
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    torch.distributed.init_process_group(backend=backend)


def _sincronizar_ranks() -> None:
    """Espera todos os ranks do DDP. No processo unico nao faz nada."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def _aguardar_cache_tripletos(
    caminho_jsonl: Path,
    arrow_path: Path,
    meta_path: Path,
    n_negativos: int,
) -> dict[str, Any]:
    """Ranks != 0: espera o meta aparecer."""
    while True:
        meta = _cache_tripletos_valido(caminho_jsonl, arrow_path, meta_path, n_negativos)
        if meta is not None:
            return meta
        time.sleep(_ESPERA_CACHE_SEGUNDOS)


def _assinatura_jsonl(caminho: Path) -> dict[str, int]:
    stat = caminho.stat()
    return {
        "jsonl_size": int(stat.st_size),
        "jsonl_mtime_ns": int(getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1e9))),
    }


def _caminhos_cache_tripletos(caminho_jsonl: Path) -> tuple[Path, Path, Path]:
    """Pasta Arrow + arquivo de dados + metadado ao lado do JSONL."""
    cache_dir = caminho_jsonl.with_name(f"{caminho_jsonl.stem}_dataset")
    arrow_path = cache_dir / "data.arrow"
    meta_path = caminho_jsonl.with_name(f"{caminho_jsonl.stem}_dataset.meta.json")
    return cache_dir, arrow_path, meta_path


def _ler_meta_cache(meta_path: Path) -> dict[str, Any] | None:
    if not meta_path.is_file():
        return None
    try:
        with meta_path.open(encoding="utf-8") as f:
            dado = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    return dado if isinstance(dado, dict) else None


def _cache_tripletos_valido(
    caminho_jsonl: Path,
    arrow_path: Path,
    meta_path: Path,
    n_negativos: int,
) -> dict[str, Any] | None:
    """Devolve o meta se o cache Arrow ainda bate com o JSONL atual."""
    meta = _ler_meta_cache(meta_path)
    if meta is None or not arrow_path.is_file() or arrow_path.stat().st_size <= 0:
        return None
    atual = _assinatura_jsonl(caminho_jsonl)
    if int(meta.get("jsonl_size", -1)) != atual["jsonl_size"]:
        return None
    if int(meta.get("jsonl_mtime_ns", -1)) != atual["jsonl_mtime_ns"]:
        return None
    if int(meta.get("n_negativos", -1)) != int(n_negativos):
        return None
    if int(meta.get("validos", 0)) <= 0:
        return None
    return meta


def _gravar_meta_cache(meta_path: Path, meta: dict[str, Any]) -> None:
    tmp = meta_path.with_suffix(meta_path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
        f.write("\n")
    tmp.replace(meta_path)


def _iterar_tripletos_treino(
    caminho: Path,
    n_negativos: int,
    query_instruction: str,
    colunas: list[str],
    stats: dict[str, int],
) -> Iterator[dict[str, list[str]]]:
    """
    Le o JSONL em streaming e entrega lotes ja no formato da loss.
    Nao acumula o corpus inteiro em listas Python.
    """
    lote: dict[str, list[str]] = {col: [] for col in colunas}
    tamanho = caminho.stat().st_size

    with caminho.open(encoding="utf-8") as f, tqdm(
        total=tamanho,
        desc="Convertendo tripletos para Arrow",
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        disable=not _processo_principal(),
    ) as barra:
        for linha in f:
            barra.update(len(linha.encode("utf-8")))
            linha = linha.strip()
            if not linha:
                continue
            stats["lidos"] += 1
            try:
                item = json.loads(linha)
            except json.JSONDecodeError:
                stats["descartados"] += 1
                continue

            query = str(item.get("query", "")).strip()
            positive = str(item.get("positive", "")).strip()
            negatives = [str(n).strip() for n in item.get("negatives", []) if str(n).strip()]

            # n_negativos=0: apenas pares (anchor, positive); negativos vêm do batch.
            if not query or not positive:
                stats["descartados"] += 1
                continue
            if n_negativos > 0 and len(negatives) < n_negativos:
                stats["descartados"] += 1
                continue

            anchor_texto = query if query.startswith("Instruct:") else f"{query_instruction}{query}"
            lote["anchor"].append(anchor_texto)
            lote["positive"].append(positive)
            for i in range(n_negativos):
                lote[f"negative_{i}"].append(negatives[i])
            stats["validos"] += 1

            if len(lote["anchor"]) >= _LOTE_CONVERSAO_ARROW:
                yield lote
                lote = {col: [] for col in colunas}

        if lote["anchor"]:
            yield lote


def _converter_jsonl_para_arrow(
    caminho: Path,
    cache_dir: Path,
    arrow_path: Path,
    meta_path: Path,
    n_negativos: int,
    query_instruction: str,
) -> dict[str, Any]:
    """
    Rank 0: grava data.arrow em lotes e o .meta.json.
    Usa pasta temporaria e renomeia ao terminar.
    """
    colunas = ["anchor", "positive"] + [f"negative_{i}" for i in range(n_negativos)]
    features = Features({col: Value("large_string") for col in colunas})
    stats = {"lidos": 0, "validos": 0, "descartados": 0}

    tmp_dir = cache_dir.with_name(cache_dir.name + ".building")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_arrow = tmp_dir / "data.arrow"

    _imprimir(f"Convertendo JSONL para cache Arrow (streaming): {caminho}")
    _imprimir(f"Destino temporario: {tmp_dir}")

    try:
        with ArrowWriter(path=str(tmp_arrow), features=features) as writer:
            for lote in _iterar_tripletos_treino(
                caminho=caminho,
                n_negativos=n_negativos,
                query_instruction=query_instruction,
                colunas=colunas,
                stats=stats,
            ):
                writer.write_batch(lote)
            writer.finalize()
    except Exception:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        raise

    if stats["validos"] == 0:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        if n_negativos > 0:
            raise ValueError(
                f"Nenhum tripleto valido com {n_negativos} negativos foi encontrado em {caminho}"
            )
        raise ValueError(
            f"Nenhum par valido (anchor, positive) foi encontrado em {caminho}"
        )

    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    tmp_dir.rename(cache_dir)

    meta = {
        **_assinatura_jsonl(caminho),
        "n_negativos": int(n_negativos),
        "lidos": int(stats["lidos"]),
        "validos": int(stats["validos"]),
        "descartados": int(stats["descartados"]),
        "arrow_relpath": "data.arrow",
    }
    _gravar_meta_cache(meta_path, meta)

    if not arrow_path.is_file():
        raise FileNotFoundError(f"Cache Arrow nao foi criado: {arrow_path}")

    _imprimir(
        f"Cache Arrow gravado: {stats['validos']:,} validos "
        f"(lidos: {stats['lidos']:,}, descartados: {stats['descartados']:,})"
    )
    return meta


def _abrir_dataset_arrow(arrow_path: Path) -> Dataset:
    # from_file usa memory-map: os ranks compartilham o mesmo arquivo no disco.
    return Dataset.from_file(str(arrow_path))


def carregar_dataset_tripletos(
    caminho_arquivo: Path | str,
    n_negativos: int = 5,
    query_instruction: str = "Instruct: Dado um trecho de um documento jurídico brasileiro, recupere o trecho mais relevante semanticamente.\nQuery: ",
) -> Dataset:
    """
    Abre o JSONL no formato da CachedMultipleNegativesRankingLoss.

    - n_negativos > 0: colunas ['anchor', 'positive', 'negative_0', ..., 'negative_{n-1}']
    - n_negativos = 0: apenas ['anchor', 'positive'] (negativos in-batch)

    Na primeira vez o rank 0 converte o JSONL para Arrow em disco (streaming).
    """
    if n_negativos < 0:
        raise ValueError(f"n_negativos deve ser >= 0; recebido: {n_negativos}")

    caminho = Path(caminho_arquivo)
    if not caminho.is_file():
        raise FileNotFoundError(f"Arquivo de tripletos/pares nao encontrado: {caminho}")

    cache_dir, arrow_path, meta_path = _caminhos_cache_tripletos(caminho)

    # Garante barrier real no DDP (o Trainer ainda nao inicializou o grupo).
    _garantir_grupo_processo()
    _sincronizar_ranks()
    meta = _cache_tripletos_valido(caminho, arrow_path, meta_path, n_negativos)

    if meta is None:
        if _processo_principal():
            meta = _converter_jsonl_para_arrow(
                caminho=caminho,
                cache_dir=cache_dir,
                arrow_path=arrow_path,
                meta_path=meta_path,
                n_negativos=n_negativos,
                query_instruction=query_instruction,
            )
        else:
            # Sem process group, o barrier seria no-op: espera o arquivo no disco.
            meta = _aguardar_cache_tripletos(caminho, arrow_path, meta_path, n_negativos)
        _sincronizar_ranks()
        meta = _cache_tripletos_valido(caminho, arrow_path, meta_path, n_negativos)
        if meta is None:
            raise RuntimeError(
                f"Falha ao preparar o cache Arrow dos tripletos em {cache_dir}. "
            )
    else:
        _imprimir(
            f"Reutilizando cache Arrow: {arrow_path} "
            f"({int(meta['validos']):,} instancias)"
        )
        _sincronizar_ranks()

    dataset = _abrir_dataset_arrow(arrow_path)
    rotulo = "pares" if n_negativos == 0 else "tripletos"
    _imprimir(
        f"Dataset de Treino Carregado: {len(dataset):,} {rotulo} validos "
        f"(n_negativos={n_negativos}, descartados: {int(meta.get('descartados', 0)):,})"
    )
    return dataset


def configurar_treinador(
    model: SentenceTransformer,
    train_dataset: Dataset,
    output_dir: Path | str,
    batch_size: int = 16,
    mini_batch_size: int = 1,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 2.0e-5,
    weight_decay: float = 0.01,
    num_epochs: int = 1,
    warmup_ratio: float = 0.05,
    eval_steps: int = 500,
    save_steps: int = 500,
    bf16: bool = False,
    fp16: bool = False,
    dataloader_num_workers: int = 0,
    dataloader_pin_memory: bool = False,
    scale: float = 50.0,
    evaluator: Optional[Any] = None,
    metric_for_best_model: str = "val_cosine_ndcg@10",
    greater_is_better: bool = True,
    optim: str = "adamw_torch",
    logging_steps: int = 1,
) -> tuple[SentenceTransformerTrainer, ScientificMetricsCallback]:
    """
    Instancia a CachedMultipleNegativesRankingLoss e o SentenceTransformerTrainer.
    Nesta branch os flags bf16/fp16 do Trainer ficam desligados: a V100 recusa
    o kernel bfloat16, e o GradScaler do fp16 quebra com pesos ja em float16.
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
    warmup_steps = int(total_steps * warmup_ratio)

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
    _imprimir(f"  - Trainer bf16/fp16        : {bool(bf16)}/{bool(fp16)}")
    _imprimir(f"  - Otimizador               : {optim}")
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
        bf16=bool(bf16),
        fp16=bool(fp16),
        optim=optim,
        logging_steps=logging_steps,
        eval_strategy="steps" if tem_avaliador else "no",
        eval_steps=eval_steps_efetivo if tem_avaliador else None,
        save_strategy="steps",
        save_steps=save_steps_efetivo,
        save_total_limit=2,
        load_best_model_at_end=tem_avaliador,
        metric_for_best_model=metric_for_best_model if tem_avaliador else None,
        greater_is_better=greater_is_better if tem_avaliador else None,
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
    # O Trainer pode ligar o GradScaler ao ver pesos em meia precisao.
    # Na V100 o sqrt do Adam em float16 devolve "invalid argument".
    trainer.args.fp16 = False
    trainer.args.bf16 = False
    acelerador = getattr(trainer, "accelerator", None)
    if acelerador is not None and getattr(acelerador, "scaler", None) is not None:
        acelerador.scaler = None
    if optim == "adamw_torch":
        _usar_adamw_sqrt_fp32(trainer, learning_rate, weight_decay)

    return trainer, telemetry_callback


class AdamWSqrtFp32(torch.optim.Optimizer):
    """
    AdamW com momentos no mesmo dtype do peso e a raiz quadrada em float32.

    O driver 510 da V100 recusa exp_avg_sq.sqrt() quando o tensor e float16
    ("CUDA driver error: invalid argument"). A conta sobe para float32 e o
    resultado volta ao dtype do parametro. Os estados continuam em float16,
    senao os dois momentos do 4B nao cabem nos 32 GB.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ) -> None:
        super().__init__(params, {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay})

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for parametro in group["params"]:
                if parametro.grad is None:
                    continue
                gradiente = parametro.grad
                estado = self.state[parametro]
                if len(estado) == 0:
                    estado["step"] = 0
                    estado["exp_avg"] = torch.zeros_like(parametro)
                    estado["exp_avg_sq"] = torch.zeros_like(parametro)

                exp_avg = estado["exp_avg"]
                exp_avg_sq = estado["exp_avg_sq"]
                estado["step"] += 1
                passo = estado["step"]

                if weight_decay != 0:
                    parametro.mul_(1 - lr * weight_decay)

                exp_avg.mul_(beta1).add_(gradiente, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(gradiente, gradiente, value=1 - beta2)

                correcao_1 = 1 - beta1 ** passo
                correcao_2_sqrt = (1 - beta2 ** passo) ** 0.5
                denominador = exp_avg_sq.float().sqrt().div_(correcao_2_sqrt).add_(eps)
                atualizacao = exp_avg.float().div_(denominador).mul_(lr / correcao_1)
                parametro.add_(atualizacao.to(dtype=parametro.dtype), alpha=-1)

        return loss


def _usar_adamw_sqrt_fp32(trainer: SentenceTransformerTrainer, learning_rate: float, weight_decay: float) -> None:
    """Troca o AdamW do PyTorch pelo passo com sqrt em float32."""

    def create_optimizer():
        modelo = trainer.model
        try:
            nomes_decay = set(trainer.get_decay_parameter_names(modelo))
        except TypeError:
            nomes_decay = set(trainer.get_decay_parameter_names())
        grupos = [
            {
                "params": [p for n, p in modelo.named_parameters() if n in nomes_decay and p.requires_grad],
                "weight_decay": weight_decay,
            },
            {
                "params": [p for n, p in modelo.named_parameters() if n not in nomes_decay and p.requires_grad],
                "weight_decay": 0.0,
            },
        ]
        grupos = [grupo for grupo in grupos if grupo["params"]]
        trainer.optimizer = AdamWSqrtFp32(grupos, lr=learning_rate, weight_decay=weight_decay)
        return trainer.optimizer

    trainer.create_optimizer = create_optimizer
