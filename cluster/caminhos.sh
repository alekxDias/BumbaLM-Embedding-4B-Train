#!/bin/bash
# Caminhos usados por criar_ambiente.sh, baixar_modelo.sh e env.sh.
# O /home tem cota pequena: ambiente, caches e modelo ficam em /projetos.

CONDA_BASE="/home/fabio.lobato/miniconda3"
BASE_PROJETOS="/projetos/ctcca/nlp"

# Ambiente conda criado por caminho (-p), fora do /home
AMBIENTE_DIR="$BASE_PROJETOS/conda_envs/bumbalm"

# Caches que por padrao iriam para o /home
export CONDA_PKGS_DIRS="$BASE_PROJETOS/.cache/conda_pkgs"
export PIP_CACHE_DIR="$BASE_PROJETOS/.cache/pip"
export HF_HOME="$BASE_PROJETOS/hf_cache"
export TMPDIR="$BASE_PROJETOS/.cache/tmp"
mkdir -p "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR" "$HF_HOME" "$TMPDIR"

source "$CONDA_BASE/etc/profile.d/conda.sh"
