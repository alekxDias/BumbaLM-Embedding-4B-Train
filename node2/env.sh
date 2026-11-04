#!/bin/bash
# Ambiente do node2 (gpunode02, 2x H200, sem internet). O node2/rodar.sh faz: source node2/env.sh
# Cada caminho pode ser trocado exportando a variavel antes de rodar.

CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
CONDA_ENV="${CONDA_ENV:-bumbalm}"
PARES_DIR="${PARES_DIR:-$HOME/bumbalm/Pares_Tribunais}"

source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}$CONDA_PREFIX/lib/"

# Sem internet: o Qwen3-Embedding-4B tem que estar no cache. Usa ~/bumbalm/hf_cache quando o
# modelo esta la (como na main); senao fica o HF_HOME atual ou o padrao ~/.cache/huggingface.
if [ -z "$HF_HOME" ] && [ -d "$HOME/bumbalm/hf_cache/hub/models--Qwen--Qwen3-Embedding-4B" ]; then
    export HF_HOME="$HOME/bumbalm/hf_cache"
fi
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

# Um no so: o rendezvous do torchrun, o NCCL e o Gloo ficam no loopback. Pelo hostname
# (gpunode02) o torchrun falhava com "errno 101 - Network is unreachable".
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-lo}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

mkdir -p logs
