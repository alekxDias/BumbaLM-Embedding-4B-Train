#!/bin/bash
# Cria o ambiente conda do projeto em /projetos (o /home tem cota pequena).
# Rode uma vez, no no de login (precisa de internet):
#   bash cluster/criar_ambiente.sh
set -e
source "$(dirname "$0")/caminhos.sh"

if [ -x "$AMBIENTE_DIR/bin/python" ]; then
    echo "Ambiente ja existe em $AMBIENTE_DIR. Para recriar: conda env remove -p $AMBIENTE_DIR"
else
    mkdir -p "$(dirname "$AMBIENTE_DIR")"
    conda create -y -p "$AMBIENTE_DIR" python=3.11
fi
conda activate "$AMBIENTE_DIR"

# --isolated ignora o pip.conf da conta, que aponta para pypi.ngc.nvidia.com
# (inacessivel daqui: cada pacote esperava 5 tentativas antes de seguir).
PIP="pip --isolated --cache-dir $PIP_CACHE_DIR"

# Driver do cluster: 510 (CUDA 11.6). As rodas cu118 funcionam nele (testado: torch 2.7.1+cu118 na V100S).
$PIP install --upgrade pip
$PIP install "torch==2.7.1" --index-url https://download.pytorch.org/whl/cu118
$PIP install -r requirements.txt

python - <<'PY'
import torch, transformers, sentence_transformers, peft
print("torch                :", torch.__version__, "| CUDA:", torch.version.cuda)
print("transformers         :", transformers.__version__)
print("sentence-transformers:", sentence_transformers.__version__)
print("peft                 :", peft.__version__)
PY
echo "Ambiente pronto em $AMBIENTE_DIR. A GPU so aparece dentro de um job: rode cluster/pbs/00_smoke.pbs."
mkdir -p logs
