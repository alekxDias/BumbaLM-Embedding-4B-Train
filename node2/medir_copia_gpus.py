#!/usr/bin/env python3
"""
Mede as copias que a mineracao fazia por processo. Roda em ~1 min e pode rodar com a mineracao ligada.

    python node2/medir_copia_gpus.py

Na mineracao antiga, cada processo copiava ~10 + ~400 linhas de 2560 dimensoes da GPU 1 para a GPU 0.
Se a copia direta entre placas levar segundos aqui, era ela que deixava a mineracao a 7 s por processo.
"""

from __future__ import annotations

import time

import numpy as np
import torch


def medir(nome: str, funcao, repeticoes: int = 10) -> None:
    funcao()
    torch.cuda.synchronize("cuda:0")
    torch.cuda.synchronize("cuda:1")
    inicio = time.perf_counter()
    for _ in range(repeticoes):
        funcao()
    torch.cuda.synchronize("cuda:0")
    torch.cuda.synchronize("cuda:1")
    print(f"{nome:<42}: {(time.perf_counter() - inicio) / repeticoes * 1000:10.2f} ms")


def main() -> None:
    print(f"PyTorch {torch.__version__} | GPUs: {torch.cuda.device_count()}")
    print(f"Acesso direto 0->1: {torch.cuda.can_device_access_peer(0, 1)} | 1->0: {torch.cuda.can_device_access_peer(1, 0)}")

    fatia = torch.randn(250_000, 2560, device="cuda:1", dtype=torch.bfloat16)
    idx = torch.randint(0, 250_000, (400,), device="cuda:1")
    matriz_cpu = np.random.rand(500_000, 2560).astype(np.float16)
    idx_cpu = np.random.randint(0, 500_000, 400)

    medir("GPU 1 -> GPU 0, direto (400 linhas)", lambda: fatia.index_select(0, idx).to("cuda:0"))
    medir("GPU 1 -> CPU -> GPU 0 (400 linhas)", lambda: fatia.index_select(0, idx).cpu().to("cuda:0"))
    medir("Linhas do .npy na CPU (como agora)", lambda: np.asarray(matriz_cpu[idx_cpu], dtype=np.float32))


if __name__ == "__main__":
    main()
