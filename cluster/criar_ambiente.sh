#!/bin/bash
# Cria o ambiente conda "bumbalm". Rode uma vez, no no de login (precisa de internet):
#   bash cluster/criar_ambiente.sh
set -e

CONDA_BASE="/home/fabio.lobato/miniconda3"
AMBIENTE="bumbalm"

source "$CONDA_BASE/etc/profile.d/conda.sh"

if conda env list | grep -qE "^${AMBIENTE}[[:space:]]"; then
    echo "Ambiente '$AMBIENTE' ja existe. Para recriar: conda env remove -n $AMBIENTE"
else
    conda create -y -n "$AMBIENTE" python=3.11
fi
conda activate "$AMBIENTE"

# Driver do cluster: 510 (CUDA 11.6). As rodas cu118 funcionam nele (testado: torch 2.7.1+cu118 na V100S).
pip install --upgrade pip
pip install "torch==2.7.1" --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

python - <<'PY'
import torch, transformers, sentence_transformers, peft
print("torch                :", torch.__version__, "| CUDA:", torch.version.cuda)
print("transformers         :", transformers.__version__)
print("sentence-transformers:", sentence_transformers.__version__)
print("peft                 :", peft.__version__)
PY
echo "Ambiente '$AMBIENTE' pronto. A GPU so aparece dentro de um job: rode cluster/pbs/00_smoke.pbs."
mkdir -p logs
