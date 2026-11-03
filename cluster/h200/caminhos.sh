#!/bin/bash
# Ambiente do gpunode02 (2x H200). Os do tapajos/V100 ficam em cluster/caminhos.sh e nao sao usados aqui.

CONDA_BASE="${CONDA_BASE:-/home/fabio.lobato/miniconda3}"
# Mesmo ambiente que ja roda o codigo da main nesta maquina
AMBIENTE="${AMBIENTE:-bumbalm}"

# Cache do Hugging Face: o padrao (~/.cache/huggingface), o mesmo que a main usa.
# So mude exportando HF_HOME antes de rodar, se o modelo estiver em outra pasta.

# O gpunode02 nao tem internet: o modelo ja esta no cache. Offline evita esperar o timeout
# de rede a cada arquivo antes de cair no cache.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

source "$CONDA_BASE/etc/profile.d/conda.sh"
