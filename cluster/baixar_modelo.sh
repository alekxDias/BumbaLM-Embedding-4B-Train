#!/bin/bash
# Baixa o Qwen3-Embedding-4B para o cache em /projetos (~8 GB). Rode uma vez, no no de login:
#   bash cluster/baixar_modelo.sh
set -e
source cluster/env.sh
# env.sh liga o modo offline para os jobs. Aqui precisa de internet.
export HF_HUB_OFFLINE=0
export TRANSFORMERS_OFFLINE=0

python - <<'PY'
from huggingface_hub import snapshot_download
caminho = snapshot_download("Qwen/Qwen3-Embedding-4B")
print("Modelo no cache:", caminho)
PY
