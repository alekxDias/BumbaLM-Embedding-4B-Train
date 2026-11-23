"""
Escolha de precisao numerica conforme a GPU disponivel.
"""

from __future__ import annotations

import torch


def gpu_tem_bf16_nativo(device: int | str | torch.device | None = None) -> bool:
    """
    bfloat16 so tem Tensor Cores a partir da Ampere (capability 8.0).
    torch.cuda.is_bf16_supported() devolve True por emulacao na V100, o que deixa o treino lento.
    """
    if not torch.cuda.is_available():
        return False
    maior, _ = torch.cuda.get_device_capability(device)
    return maior >= 8


def dtype_para_dispositivo(device: int | str | torch.device | None = None, precisao: str = "auto") -> torch.dtype:
    """
    precisao: "auto", "bf16", "fp16", "fp32" ou "tf32".
    "auto" usa bf16 na A100/H100/H200, fp16 na V100 e fp32 na CPU.
    "tf32" guarda os pesos em fp32, como "fp32"; a diferenca esta nas multiplicacoes de matrizes,
    ligadas por aplicar_precisao_matmul.
    """
    escolha = str(precisao or "auto").strip().lower()
    em_cuda = torch.cuda.is_available() and not str(device or "cuda").startswith("cpu")
    if not em_cuda or escolha in ("fp32", "tf32"):
        return torch.float32
    if escolha == "bf16":
        return torch.bfloat16
    if escolha == "fp16":
        return torch.float16
    return torch.bfloat16 if gpu_tem_bf16_nativo(device) else torch.float16


def aplicar_precisao_matmul(precisao: str) -> str:
    """
    Com "tf32", as multiplicacoes de matrizes em fp32 usam os Tensor Cores TF32 (Ampere em diante):
    as entradas sao arredondadas para 10 bits de mantissa e a soma continua em fp32. Na A100 isso e
    ~8x mais rapido que o fp32 puro, sem arredondar os pesos. Nas outras precisoes fica o padrao do
    PyTorch ("highest", fp32 puro). Devolve o modo em vigor.
    """
    modo = "high" if str(precisao or "").strip().lower() == "tf32" else "highest"
    torch.set_float32_matmul_precision(modo)
    return torch.get_float32_matmul_precision()
