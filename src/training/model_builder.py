"""
Construcao e configuracao da arquitetura do modelo de embedding (SentenceTransformer).
"""

from __future__ import annotations

import gc
import logging
import os
from typing import Optional

import torch
from sentence_transformers import SentenceTransformer, models


log = logging.getLogger("bumbalm_model_builder")


def construir_modelo_embedding(
    model_id: str = "Qwen/Qwen3-Embedding-4B",
    max_seq_length: int = 768,
    attn_implementation: str = "sdpa",
    gradient_checkpointing: bool = False,
    device: Optional[str] = None,
) -> SentenceTransformer:
    """
    Instancia e configura o pipeline SentenceTransformer para Full Fine-Tuning:
    1. Camada Transformer: Base Qwen3 com precisao bfloat16 e atencao acelerada.
    2. Camada Pooling: Modo 'lasttoken'.
    3. Camada Normalize: Projecao na esfera unitaria L2.

    Args:
        model_id: Identificador do modelo no Hugging Face Hub ou caminho local.
        max_seq_length: Comprimento maximo de tokens por texto (padrao: 768).
        attn_implementation: Mecanismo de atencao ('sdpa' ou 'flash_attention_2').
        gradient_checkpointing: Se True, economiza VRAM recalculando ativacoes no backward.
            Na GPU H200 (141 GB), manter False acelera o treino em ate 35%.
        device: Dispositivo de execucao ('cuda', 'cuda:0', 'cuda:1' ou 'cpu').

    Returns:
        Instancia de SentenceTransformer pronta para fine-tuning.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    em_cuda = str(device).startswith("cuda")
    if em_cuda and ":" in str(device):
        torch.cuda.set_device(int(str(device).split(":", 1)[1]))

    dtype_modelo = torch.bfloat16 if em_cuda and torch.cuda.is_bf16_supported() else (torch.float16 if em_cuda else torch.float32)

    rank = os.environ.get("RANK", "0")
    print(f"[rank {rank}] Construindo modelo em {device}", flush=True)
    if rank == "0":
        print(f"Construindo modelo base: {model_id} ...")
        print(f"  - Dispositivo          : {device}")
        print(f"  - Precisao             : {dtype_modelo}")
        print(f"  - Comprimento Maximo   : {max_seq_length} tokens")
        print(f"  - Atencao Otimizada    : {attn_implementation}")
        print(f"  - Gradient Checkpoint  : {gradient_checkpointing}")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 1. Camada Transformer
    word_embedding_model = models.Transformer(
        model_name_or_path=model_id,
        max_seq_length=max_seq_length,
        tokenizer_args={"padding_side": "left"},
        model_args={
            "attn_implementation": attn_implementation if em_cuda else "eager",
            "torch_dtype": dtype_modelo,
            "trust_remote_code": True,
        },
    )
    word_embedding_model.auto_model.config.use_cache = False

    # 2. Ativacao ou desativacao do Gradient Checkpointing
    if gradient_checkpointing and em_cuda:
        word_embedding_model.auto_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        if rank == "0":
            print("  - Gradient Checkpointing ativado com sucesso.")
    elif rank == "0":
        print("  - Gradient Checkpointing desativado (priorizando velocidade na H200).")

    # 3. Full Fine-Tuning: Habilita gradientes em 100% dos parametros do Transformer
    for param in word_embedding_model.auto_model.parameters():
        param.requires_grad = True

    # 4. Camada de Pooling (Last Token)
    pooling_model = models.Pooling(
        word_embedding_model.get_word_embedding_dimension(),
        pooling_mode="lasttoken",
    )

    # 5. Camada de Normalizacao L2
    normalize_model = models.Normalize()

    # 6. Montagem do Pipeline Modular
    modelo = SentenceTransformer(
        modules=[word_embedding_model, pooling_model, normalize_model],
        device=device,
    )

    total_params = sum(p.numel() for p in word_embedding_model.auto_model.parameters())
    trainable_params = sum(p.numel() for p in word_embedding_model.auto_model.parameters() if p.requires_grad)
    pct_treinavel = 100 * trainable_params / max(1, total_params)

    if rank == "0":
        print(f"Modelo estruturado com sucesso: {trainable_params:,} parametros treinaveis ({pct_treinavel:.1f}% - Full Fine-Tuning)")

    return modelo
