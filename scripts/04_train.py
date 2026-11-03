#!/usr/bin/env python3
"""
Pipeline BumbaLM - Etapa 4: Treinamento Contrastivo do Modelo.

Executa:
1. Leitura dos hiperparametros de treinamento e hardware ('config_train.yaml').
2. Construcao da arquitetura do Qwen3-Embedding-4B (Full Fine-Tuning ou LoRA, conforme o YAML).
3. Carregamento das tuplas da variante escolhida em --negativos:
   - dificeis  : 'train_triplets.jsonl' (5 hard negatives minerados na Etapa 3)
   - aleatorios: 'train_tuplas_aleatorias.jsonl' (mesmos pares, 5 negativos sorteados na Etapa 3b)
   - nenhum    : mesmos pares de 'train_triplets.jsonl', sem negativos explicitos (so os do lote)
4. Configuracao da CachedMultipleNegativesRankingLoss e do SentenceTransformerTrainer.
5. Execucao do fine-tuning com telemetria cientifica de hardware em tempo real.
6. Salvamento do modelo em experiments/<nome_experimento>/<negativos>/ e do caminho do
   melhor checkpoint em 'melhor_checkpoint.json'.

Uso:
    python scripts/04_train.py --negativos dificeis
    python scripts/04_train.py --negativos nenhum
    python scripts/04_train.py --negativos aleatorios --limite_tuplas 50000
    python scripts/04_train.py --gpus 0,1
"""

from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

# Adiciona a raiz do repositorio ao PYTHONPATH
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import torch
import transformers.utils.import_utils as _tf_import_utils
import transformers.trainer as _tf_trainer

# Bypass do bloqueio de versao do torch.load para carregar checkpoints locais confiaveis (CVE-2025-32434)
_tf_import_utils.check_torch_load_is_safe = lambda: None
_tf_trainer.check_torch_load_is_safe = lambda: None

# Suporte para desserializacao segura de objetos NumPy (NumPy 2.0 / Python 3.13)
try:
    import numpy as np
    safe_list = [np.ndarray, np.dtype]
    if hasattr(np, "_core") and hasattr(np._core, "multiarray"):
        if hasattr(np._core.multiarray, "_reconstruct"):
            safe_list.append(np._core.multiarray._reconstruct)
        if hasattr(np._core.multiarray, "scalar"):
            safe_list.append(np._core.multiarray.scalar)
    if hasattr(torch.serialization, "add_safe_globals"):
        torch.serialization.add_safe_globals(safe_list)
except Exception:
    pass

# Tratamento defensivo para rng_state: evita abortar o treino caso o pickle da semente aleatoria falhe
if hasattr(_tf_trainer.Trainer, "_load_rng_state"):
    _orig_load_rng_state = _tf_trainer.Trainer._load_rng_state
    def _safe_load_rng_state(self, resume_from_checkpoint):
        try:
            _orig_load_rng_state(self, resume_from_checkpoint)
        except Exception as exc:
            print(f"[Aviso] Nao foi possivel restaurar rng_state ({exc}). O treinamento continuara com novo estado de RNG.")
    _tf_trainer.Trainer._load_rng_state = _safe_load_rng_state

from src.training.model_builder import construir_modelo_embedding
from src.training.trainer import carregar_dataset_tripletos, configurar_treinador
from src.evaluation.ir_evaluator import criar_ir_evaluator
from src.hardware import dtype_para_dispositivo


# Variante de negativos -> (arquivo de tuplas, negativos por tupla). None usa o n_negativos do YAML.
VARIANTES_NEGATIVOS = {
    "dificeis": ("train_triplets.jsonl", None),
    "aleatorios": ("train_tuplas_aleatorias.jsonl", None),
    "nenhum": ("train_triplets.jsonl", 0),
}


def carregar_yaml(caminho: Path) -> dict[str, Any]:
    """Le arquivo YAML com seguranca."""
    if not caminho.is_file():
        raise FileNotFoundError(f"Arquivo de configuracao nao encontrado: {caminho}")
    try:
        import yaml
        with caminho.open(encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        return {}


def _ids_gpus(texto: str) -> list[str]:
    return [parte.strip() for parte in str(texto or "").split(",") if parte.strip()]


def _ja_distribuido() -> bool:
    return os.environ.get("LOCAL_RANK") is not None


def _processo_principal() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def _imprimir(mensagem: str) -> None:
    if _processo_principal():
        print(mensagem)


def _resolver_gpus(args: argparse.Namespace, cfg_train: dict[str, Any]) -> str:
    if args.gpus is not None:
        return args.gpus.strip()
    bruto = cfg_train.get("hardware", {}).get("gpus")
    if bruto is None:
        return ""
    return str(bruto).strip()


def _preparar_gpus(gpus: str) -> None:
    """Fixa uma GPU ou relanca o script em DDP quando ha duas ou mais."""
    if _ja_distribuido():
        return

    ids = _ids_gpus(gpus)
    if not ids:
        return

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(ids)
    if len(ids) == 1:
        return

    comando = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node",
        str(len(ids)),
        str(Path(__file__).resolve()),
        *sys.argv[1:],
    ]
    print(f"Relancando treino em {len(ids)} GPUs ({','.join(ids)}) com torch.distributed.run ...")
    resultado = subprocess.run(comando)
    sys.exit(resultado.returncode)


def _dispositivo_local() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    return f"cuda:{local_rank}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config_base",
        type=Path,
        default=BASE_DIR / "configs" / "config_base.yaml",
        help="Caminho do config_base.yaml",
    )
    parser.add_argument(
        "--config_train",
        type=Path,
        default=BASE_DIR / "configs" / "config_train.yaml",
        help="Caminho do config_train.yaml",
    )
    parser.add_argument(
        "--config_data",
        type=Path,
        default=BASE_DIR / "configs" / "config_data.yaml",
        help="Caminho do config_data.yaml (n_negativos).",
    )
    parser.add_argument(
        "--negativos",
        type=str,
        choices=sorted(VARIANTES_NEGATIVOS),
        default="dificeis",
        help="Variante da ablacao: dificeis, aleatorios ou nenhum (so negativos do lote).",
    )
    parser.add_argument(
        "--limite_tuplas",
        type=int,
        default=None,
        help="Sorteia este numero de tuplas (mesma semente em todas as variantes). Sobrescreve treinamento.limite_tuplas.",
    )
    parser.add_argument(
        "--triplets_file",
        type=Path,
        default=None,
        help="Caminho do arquivo de tuplas. Padrao: o arquivo da variante em --negativos.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Diretorio de saida. Padrao: experiments/<nome_experimento>/<negativos>.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Sobrescreve o batch_size_train configurado no YAML.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Sobrescreve a quantidade de epocas configurada no YAML.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Sobrescreve a taxa de aprendizado (learning_rate).",
    )
    parser.add_argument(
        "--mini_batch_size",
        type=int,
        default=None,
        help="Sobrescreve o mini_batch_size para o GradCache (CachedMNRL).",
    )
    parser.add_argument(
        "--no_eval",
        action="store_true",
        help="Desativa o InformationRetrievalEvaluator durante o treinamento.",
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=None,
        help="Sobrescreve a frequencia de avaliacao intermediaria (eval_steps).",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Caminho do checkpoint ou 'auto' para retomar o treinamento a partir do mais recente.",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="GPUs a usar, separadas por virgula. Ex.: '1' ou '0,1'. Vazio mantem o processo unico atual.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg_base = carregar_yaml(args.config_base)
    cfg_train = carregar_yaml(args.config_train)
    cfg_data = carregar_yaml(args.config_data) if args.config_data.is_file() else {}
    _preparar_gpus(_resolver_gpus(args, cfg_train))

    seed = cfg_base.get("projeto", {}).get("seed", 42)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Localizacao do arquivo de dados de tripletos
    dir_processados = Path(cfg_base.get("dados", {}).get("diretorio_processados", "data/processed"))
    if not dir_processados.is_absolute():
        dir_processados = BASE_DIR / dir_processados

    arquivo_variante, n_neg_variante = VARIANTES_NEGATIVOS[args.negativos]
    n_negativos = (
        n_neg_variante
        if n_neg_variante is not None
        else int(cfg_data.get("mineracao_negativos", {}).get("n_negativos", 5))
    )
    caminho_tripletos = args.triplets_file or (dir_processados / arquivo_variante)
    if not Path(caminho_tripletos).is_absolute():
        caminho_tripletos = BASE_DIR / caminho_tripletos

    # Diretorio de saida: uma pasta por variante dentro do experimento
    exp_dir_base = Path(cfg_base.get("saidas", {}).get("diretorio_experimentos", "experiments"))
    exp_nome = cfg_base.get("saidas", {}).get("nome_experimento", "exp_001_h200_4b")
    dir_saida = args.output_dir or (exp_dir_base / exp_nome / args.negativos)
    if not Path(dir_saida).is_absolute():
        dir_saida = BASE_DIR / dir_saida
    dir_saida.mkdir(parents=True, exist_ok=True)

    # Parametros do modelo
    modelo_cfg = cfg_train.get("modelo", {})
    model_id = modelo_cfg.get("model_id", "Qwen/Qwen3-Embedding-4B")
    max_seq_len = modelo_cfg.get("max_seq_length", 768)
    query_instruction = modelo_cfg.get(
        "query_instruction",
        "Instruct: Dado um trecho de um documento jurídico brasileiro, recupere o trecho mais relevante semanticamente.\nQuery: ",
    )
    attn_impl = modelo_cfg.get("attn_implementation", "sdpa")

    # Parametros de treino
    train_cfg = cfg_train.get("treinamento", {})
    batch_size = args.batch_size or train_cfg.get("batch_size_train", 64)
    mini_batch_size = args.mini_batch_size or train_cfg.get("mini_batch_size", 32)
    grad_accum = train_cfg.get("gradient_accumulation_steps", 1)
    lr = args.lr or train_cfg.get("learning_rate", 2.0e-5)
    weight_decay = train_cfg.get("weight_decay", 0.01)
    epochs = args.epochs or train_cfg.get("num_epochs", 1)
    warmup_ratio = train_cfg.get("warmup_ratio", 0.05)
    precisao = str(train_cfg.get("precisao", "auto"))
    limite_tuplas = args.limite_tuplas or train_cfg.get("limite_tuplas")
    save_total_limit = int(train_cfg.get("save_total_limit", 2))
    cfg_lora = cfg_train.get("lora", {}) or {}
    usar_lora = bool(cfg_lora.get("ativo", False))
    grad_checkpointing = train_cfg.get("gradient_checkpointing", False)
    num_workers = train_cfg.get("dataloader_num_workers", 4)
    pin_memory = train_cfg.get("dataloader_pin_memory", False)
    eval_steps = train_cfg.get("eval_steps", 250)
    save_steps = train_cfg.get("save_steps", 250)
    loss_scale = cfg_train.get("loss", {}).get("scale", 50.0)

    dispositivo = _dispositivo_local()
    num_processos = max(1, int(os.environ.get("WORLD_SIZE", "1")))

    # bf16 na A100/H200; fp16 na V100. O Trainer usa a mesma escolha na precisao mista.
    dtype_alvo = dtype_para_dispositivo(dispositivo, precisao)
    precisao_mista = {torch.bfloat16: "bf16", torch.float16: "fp16"}.get(dtype_alvo, "no")

    _imprimir("=" * 64)
    _imprimir("BumbaLM - Etapa 4: Treinamento Contrastivo")
    _imprimir("=" * 64)
    _imprimir(f"Variante de Negativos: {args.negativos} ({n_negativos} por tupla)")
    _imprimir(f"Arquivo de Tripletos : {caminho_tripletos}")
    _imprimir(f"Diretorio de Saida   : {dir_saida}")
    _imprimir(f"Modelo de Base       : {model_id}")
    _imprimir(f"GPUs no passo        : {num_processos} | Dispositivo local: {dispositivo}")
    _imprimir(f"Batch por GPU        : {batch_size} (Acumulacao: {grad_accum})")
    _imprimir(f"Taxa de Aprendizado  : {lr} | Epocas: {epochs}")
    _imprimir(f"Precisao mista       : {precisao_mista} | Atencao: {attn_impl}")
    _imprimir(f"LoRA                 : {usar_lora} | Limite de tuplas: {limite_tuplas or 'todas'}")
    _imprimir(f"Dataloader Workers   : {num_workers} | Pin Memory: {pin_memory}")
    _imprimir("-" * 64)

    # 1. Construcao do Modelo de Embedding
    model = construir_modelo_embedding(
        model_id=model_id,
        max_seq_length=max_seq_len,
        attn_implementation=attn_impl,
        gradient_checkpointing=grad_checkpointing,
        device=dispositivo,
        precisao=precisao,
        lora=cfg_lora,
    )

    # 2. Carregamento do Dataset de Tripletos
    train_dataset = carregar_dataset_tripletos(
        caminho_arquivo=caminho_tripletos,
        n_negativos=n_negativos,
        query_instruction=query_instruction,
        limite_tuplas=limite_tuplas,
        seed=seed,
    )

    # 3. Construcao do Avaliador de Validacao (InformationRetrievalEvaluator)
    evaluator = None
    if not args.no_eval:
        arq_val_processos = dir_processados / "val_processos.jsonl"
        arq_val_chunks = dir_processados / "val_corpus_chunks.jsonl"
        if not arq_val_chunks.is_file():
            arq_val_chunks = dir_processados / "validacao_corpus_chunks.jsonl"

        if arq_val_processos.is_file() and arq_val_chunks.is_file():
            cfg_avaliacao = cfg_train.get("avaliacao", {})
            eval_batch_size = cfg_avaliacao.get("eval_batch_size", 128)
            max_processos_val = cfg_avaliacao.get("max_processos_val")
            _imprimir("Configurando InformationRetrievalEvaluator sobre o split de validacao...")
            evaluator = criar_ir_evaluator(
                caminho_processos=arq_val_processos,
                caminho_chunks=arq_val_chunks,
                nome="val",
                query_instruction=query_instruction,
                excluir_chunks_contaminados=True,
                batch_size=eval_batch_size,
                show_progress_bar=False,
                limite_processos=max_processos_val,
                seed=seed,
            )
            _imprimir("  - Avaliador de validacao pronto (chunks com vazamento de ementa excluidos).")
        else:
            _imprimir("Aviso: Arquivos de validacao nao encontrados. O treinamento continuara sem avaliacao intermediaria.")
    else:
        _imprimir("Flag --no_eval detectada: Avaliacao intermediaria desativada.")

    # 4. Configuracao do Trainer e Telemetria
    # As chaves ficam em treinamento: no YAML. A leitura antiga no topo sempre caia no padrao.
    metric_best = train_cfg.get("metric_for_best_model", cfg_train.get("metric_for_best_model", "val_cosine_ndcg@10"))
    greater_is_better = train_cfg.get("greater_is_better", cfg_train.get("greater_is_better", True))
    if args.eval_steps is not None:
        eval_steps = args.eval_steps

    trainer, telemetry = configurar_treinador(
        model=model,
        train_dataset=train_dataset,
        output_dir=dir_saida,
        batch_size=batch_size,
        mini_batch_size=mini_batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        weight_decay=weight_decay,
        num_epochs=epochs,
        warmup_ratio=warmup_ratio,
        eval_steps=eval_steps,
        save_steps=save_steps,
        precisao_mista=precisao_mista,
        dataloader_num_workers=num_workers,
        dataloader_pin_memory=pin_memory,
        scale=loss_scale,
        evaluator=evaluator,
        metric_for_best_model=metric_best,
        greater_is_better=greater_is_better,
        save_total_limit=save_total_limit,
        # Com LoRA o checkpoint guarda so o adaptador. O melhor e avaliado pelo caminho, na Etapa 5.
        carregar_melhor_no_fim=not usar_lora,
        seed=seed,
    )

    # 4. Execucao do Fine-Tuning
    resume = args.resume_from_checkpoint
    if resume is not None:
        if str(resume).strip().lower() in ("true", "auto", "1"):
            resume = True
        else:
            p_resume = Path(resume)
            if not p_resume.is_absolute():
                if (dir_saida / p_resume).exists():
                    p_resume = dir_saida / p_resume
                elif (BASE_DIR / p_resume).exists():
                    p_resume = BASE_DIR / p_resume
            resume = str(p_resume)

    _imprimir("-" * 64)
    if resume:
        _imprimir(f"Retomando treinamento a partir do checkpoint: {resume}")
    else:
        _imprimir("Disparando treinamento contrastivo com telemetria cientifica...")
    trainer.train(resume_from_checkpoint=resume)

    # 5. Salvamento do Modelo Final (somente o processo da primeira GPU)
    if _processo_principal():
        modelo_salvar = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
        # Sem LoRA, load_best_model_at_end ja trouxe o melhor. Com LoRA, este e o ultimo passo.
        nome_final = "modelo_final" if usar_lora else "best_model_4b"
        best_model_path = dir_saida / nome_final
        modelo_salvar.save(str(best_model_path))

        melhor_ckpt = trainer.state.best_model_checkpoint
        registro = {
            "variante_negativos": args.negativos,
            "n_negativos": n_negativos,
            "arquivo_tuplas": str(caminho_tripletos),
            "n_tuplas_treino": len(train_dataset),
            "lora": usar_lora,
            "precisao_mista": precisao_mista,
            "metrica_selecao": metric_best,
            "melhor_metrica": trainer.state.best_metric,
            "melhor_checkpoint": melhor_ckpt,
            "modelo_final": str(best_model_path),
            # A Etapa 5 avalia este caminho.
            "modelo_para_avaliar": str(melhor_ckpt) if (usar_lora and melhor_ckpt) else str(best_model_path),
        }
        (dir_saida / "melhor_checkpoint.json").write_text(
            json.dumps(registro, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        print("=" * 64)
        print("Treinamento concluido com sucesso!")
        print(f"Modelo final persistido em : {best_model_path}")
        print(f"Modelo para a Etapa 5      : {registro['modelo_para_avaliar']}")
        print(f"Relatorio executivo JSON   : {telemetry.json_path}")
        print(f"Metricas tabulares CSV     : {telemetry.csv_path}")
        print("=" * 64)

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


if __name__ == "__main__":
    main()
