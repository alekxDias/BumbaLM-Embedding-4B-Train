"""
Construcao e configuracao da arquitetura do modelo de embedding (SentenceTransformer).
"""

from __future__ import annotations

import gc
import logging
import os
from typing import Any, Optional

import torch
from sentence_transformers import SentenceTransformer

from src.hardware import dtype_para_dispositivo


log = logging.getLogger("bumbalm_model_builder")

# Projecoes de atencao e MLP do Qwen3.
LORA_ALVOS_PADRAO = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def _aplicar_lora(modelo: SentenceTransformer, cfg_lora: dict[str, Any], rank: str) -> None:
    """
    Adiciona adaptadores LoRA ao Transformer e congela o restante.
    Os pesos do LoRA ficam em float32: o GradScaler do fp16 nao aceita gradiente em fp16.
    """
    try:
        from peft import LoraConfig, TaskType
    except ImportError as exc:
        raise ImportError("LoRA ativado no config_train.yaml, mas o pacote 'peft' nao esta instalado.") from exc

    config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=int(cfg_lora.get("r", 16)),
        lora_alpha=int(cfg_lora.get("alpha", 32)),
        lora_dropout=float(cfg_lora.get("dropout", 0.05)),
        target_modules=list(cfg_lora.get("target_modules") or LORA_ALVOS_PADRAO),
        bias="none",
    )
    # add_adapter do SentenceTransformer: os checkpoints salvos ja sabem carregar o adaptador.
    modelo.add_adapter(config)

    for param in modelo.parameters():
        if param.requires_grad and param.dtype != torch.float32:
            param.data = param.data.float()

    if rank == "0":
        print(f"  - LoRA                 : r={config.r} | alpha={config.lora_alpha} | dropout={config.lora_dropout}")
        print(f"  - Modulos LoRA         : {', '.join(sorted(config.target_modules))}")


def construir_modelo_embedding(
    model_id: str = "Qwen/Qwen3-Embedding-4B",
    max_seq_length: int = 768,
    attn_implementation: str = "sdpa",
    gradient_checkpointing: bool = False,
    device: Optional[str] = None,
    precisao: str = "auto",
    lora: Optional[dict[str, Any]] = None,
    pesos_fp32: bool = False,
) -> SentenceTransformer:
    """
    Carrega o pipeline SentenceTransformer do Hub e o prepara para o fine-tuning:
    1. Camada Transformer: Base Qwen3 com precisao conforme a GPU e atencao acelerada.
    2. Camada Pooling: Modo 'lasttoken'.
    3. Camada Normalize: Projecao na esfera unitaria L2.

    Args:
        model_id: Identificador do modelo no Hugging Face Hub ou caminho local.
        max_seq_length: Comprimento maximo de tokens por texto (padrao: 768).
        attn_implementation: Mecanismo de atencao ('sdpa' ou 'flash_attention_2').
        gradient_checkpointing: Se True, economiza VRAM recalculando ativacoes no backward.
            Na V100 (32 GB) e obrigatorio. Na H200 (141 GB), manter False acelera o treino.
        device: Dispositivo de execucao ('cuda', 'cuda:0', 'cuda:1' ou 'cpu').
        precisao: "auto", "bf16", "fp16" ou "fp32". "auto" usa fp16 na V100 e bf16 da Ampere em diante.
        lora: Configuracao do LoRA ({"ativo": True, "r": 16, ...}). None ou ativo=False faz Full Fine-Tuning.
        pesos_fp32: No full fine-tuning, carrega os pesos em fp32 e deixa o bf16 so na autocast do Trainer.
            Sem isso, na H200 pesos, gradientes e Adam ficam todos em bf16 (bf16 "puro").

    Returns:
        Instancia de SentenceTransformer pronta para fine-tuning.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    em_cuda = str(device).startswith("cuda")
    if em_cuda and ":" in str(device):
        torch.cuda.set_device(int(str(device).split(":", 1)[1]))

    usar_lora = bool(lora and lora.get("ativo", False))
    dtype_modelo = dtype_para_dispositivo(device, precisao)
    # Full fine-tuning em fp16 precisa dos pesos mestres em fp32. So o LoRA mantem a base em fp16.
    if dtype_modelo == torch.float16 and not usar_lora:
        dtype_modelo = torch.float32
    # Em bf16 puro, atualizacoes de ~2e-5 somem no arredondamento de pesos da ordem de 1e-2
    if pesos_fp32 and not usar_lora and em_cuda:
        dtype_modelo = torch.float32

    rank = os.environ.get("RANK", "0")
    print(f"[rank {rank}] Construindo modelo em {device}", flush=True)
    if rank == "0":
        print(f"Construindo modelo base: {model_id} ...")
        print(f"  - Dispositivo          : {device}")
        print(f"  - Precisao dos pesos   : {dtype_modelo}")
        print(f"  - Comprimento Maximo   : {max_seq_length} tokens")
        print(f"  - Atencao Otimizada    : {attn_implementation}")
        print(f"  - Gradient Checkpoint  : {gradient_checkpointing}")
        print(f"  - Estrategia           : {'LoRA' if usar_lora else 'Full Fine-Tuning'}")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 1. Pipeline completo do Hub (Transformer, Pooling last-token e Normalize), do mesmo jeito que a
    # mineracao e a Etapa 5 carregam o modelo base. A montagem manual dos modulos
    # (models.Transformer + Pooling + Normalize) dava um modelo diferente no sentence-transformers do
    # node2: validacao no passo 0 com MRR 0,04 e loss 16 no primeiro passo, contra MRR 0,54 do base.
    # Confira com: bash node2/rodar.sh testar_modelo
    modelo = SentenceTransformer(
        model_id,
        device=device,
        trust_remote_code=True,
        model_kwargs={
            "attn_implementation": attn_implementation if em_cuda else "eager",
            "torch_dtype": dtype_modelo,
        },
        tokenizer_kwargs={"padding_side": "left"},
    )
    modelo.max_seq_length = max_seq_length
    word_embedding_model = modelo[0]
    word_embedding_model.auto_model.config.use_cache = False

    # 2. Ativacao ou desativacao do Gradient Checkpointing
    if gradient_checkpointing and em_cuda:
        word_embedding_model.auto_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        if rank == "0":
            print("  - Gradient Checkpointing ativado com sucesso.")
    elif rank == "0":
        print("  - Gradient Checkpointing desativado.")

    # 3. Full Fine-Tuning: Habilita gradientes em 100% dos parametros do Transformer.
    # Com LoRA, o add_adapter congela a base logo abaixo.
    for param in word_embedding_model.auto_model.parameters():
        param.requires_grad = True

    if rank == "0":
        tokenizador = getattr(modelo, "tokenizer", None)
        print(f"  - Padding do tokenizer : {getattr(tokenizador, 'padding_side', '?')}")
        print(f"  - Modulos              : {' -> '.join(type(m).__name__ for m in modelo)}")

    if usar_lora:
        _aplicar_lora(modelo, lora, rank)

    total_params = sum(p.numel() for p in word_embedding_model.auto_model.parameters())
    trainable_params = sum(p.numel() for p in word_embedding_model.auto_model.parameters() if p.requires_grad)
    pct_treinavel = 100 * trainable_params / max(1, total_params)

    if rank == "0":
        estrategia = "LoRA" if usar_lora else "Full Fine-Tuning"
        print(f"Modelo estruturado com sucesso: {trainable_params:,} parametros treinaveis ({pct_treinavel:.2f}% - {estrategia})")

    return modelo
