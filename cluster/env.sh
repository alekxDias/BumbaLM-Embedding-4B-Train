#!/bin/bash
# Ambiente comum a todos os jobs. Os .pbs fazem: source cluster/env.sh
# Os caminhos ficam em cluster/caminhos.sh.

source "$(dirname "${BASH_SOURCE[0]}")/caminhos.sh"
conda activate "$AMBIENTE_DIR"

# Igual ao exemplo do orientador
module load cuda 2>/dev/null || true
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/

# O modelo e baixado antes, no no de login (cluster/baixar_modelo.sh).
# Offline evita que o job trave tentando acessar o Hugging Face sem internet.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
# Sem expandable_segments: no driver 510 (CUDA 11.6) da V100 ele falha com
# "CUDA driver error: invalid argument" assim que a memoria precisa crescer.
unset PYTORCH_CUDA_ALLOC_CONF
# Threads de CPU iguais aos nucleos pedidos no PBS (ppn)
export OMP_NUM_THREADS="${PBS_NP:-8}"

mkdir -p logs
