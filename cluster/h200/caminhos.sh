#!/bin/bash
# Caminhos da maquina das 2x H200. Ajuste antes da primeira execucao.
# (Os do tapajos/V100 ficam em cluster/caminhos.sh e nao sao usados aqui.)

CONDA_BASE="${CONDA_BASE:-/home/fabio.lobato/miniconda3}"
# Nome (ou caminho) do ambiente conda que ja roda o pipeline nesta maquina
AMBIENTE="${AMBIENTE:-bumbalm}"
export HF_HOME="${HF_HOME:-/home/fabio.lobato/bumbalm/hf_cache}"

# O gpunode02 nao tem internet: o modelo precisa ja estar em $HF_HOME (ver cluster/h200/README.md).
# Sem estas variaveis o transformers tenta acessar o Hugging Face e falha ou fica esperando.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

source "$CONDA_BASE/etc/profile.d/conda.sh"
