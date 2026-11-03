#!/bin/bash
# Ambiente comum a todos os jobs. Os .pbs fazem: source cluster/env.sh
# Ajuste os tres caminhos abaixo se mudar a pasta do projeto ou o nome do ambiente.

CONDA_BASE="/home/fabio.lobato/miniconda3"
AMBIENTE="bumbalm"
export HF_HOME="/projetos/ctcca/nlp/hf_cache"

source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$AMBIENTE"

# Igual ao exemplo do orientador
module load cuda 2>/dev/null || true
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/

# O modelo e baixado antes, no no de login (cluster/baixar_modelo.sh).
# Offline evita que o job trave tentando acessar o Hugging Face sem internet.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
# Threads de CPU iguais aos nucleos pedidos no PBS (ppn)
export OMP_NUM_THREADS="${PBS_NP:-8}"

mkdir -p logs
