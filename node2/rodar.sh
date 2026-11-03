#!/bin/bash
# Ablacao de negativos do TCC no node2 (2x H200). Rode dentro do tmux, a partir da raiz do projeto:
#
#   bash node2/rodar.sh checar      # GPUs, modelo no cache, amostra e espaco em disco
#   bash node2/rodar.sh amostra     # sorteia os 60 mil acordaos de Camara do pares_tjrs.jsonl
#   bash node2/rodar.sh smoke       # ponta a ponta com 500 acordaos e 640 tuplas por variante
#   bash node2/rodar.sh preparar    # split, chunk, mineracao dos dificeis e sorteio dos aleatorios
#   bash node2/rodar.sh treinar     # as tres variantes, uma de cada vez
#   bash node2/rodar.sh avaliar     # teste completo das variantes e do modelo base
#   bash node2/rodar.sh completo    # preparar + treinar + avaliar
#
# Variaveis opcionais: VARIANTES="nenhum,aleatorios,dificeis", LIMITE_TUPLAS=100000,
# LIMPAR_CHECKPOINTS=0 (mantem os checkpoint-* depois de cada variante).
# Se o processo cair, rode o mesmo comando de novo: a mineracao retoma do ultimo bloco, o treino
# retoma do ultimo checkpoint e as variantes ja concluidas sao puladas.

set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source node2/env.sh

MODO="${1:-}"
VARIANTES="${VARIANTES:-nenhum,aleatorios,dificeis}"
LIMPAR_CHECKPOINTS="${LIMPAR_CHECKPOINTS:-1}"
AMOSTRA="$PARES_DIR/pares_tjrs_amostra60k_camaras.jsonl"

# Mesmos nomes de saidas.nome_experimento dos configs
CONFIG_BASE="configs/config_base.yaml"
EXPERIMENTO="experiments/tcc_tjrs_ablacao_negativos_h200"
if [ "$MODO" = "smoke" ]; then
    CONFIG_BASE="configs/config_base_smoke.yaml"
    EXPERIMENTO="experiments/smoke_tjrs_h200"
fi

LOG="logs/node2_${MODO:-ajuda}_$(date +%Y%m%d_%H%M%S).log"

pipeline() {
    python -u scripts/run_pipeline.py --config_base "$CONFIG_BASE" "$@" 2>&1 | tee -a "$LOG"
}

espaco_livre_gb() {
    df -Pk . | awk 'NR==2 {print int($4 / 1024 / 1024)}'
}

exigir_espaco() {
    local minimo="$1" livre
    livre="$(espaco_livre_gb)"
    if [ "$livre" -lt "$minimo" ]; then
        echo "ERRO: $livre GB livres em $(pwd). O modo '$MODO' precisa de ~$minimo GB." | tee -a "$LOG"
        exit 1
    fi
}

exigir_amostra() {
    if [ ! -f "$AMOSTRA" ]; then
        echo "ERRO: $AMOSTRA nao existe. Rode antes: bash node2/rodar.sh amostra" | tee -a "$LOG"
        exit 1
    fi
}

checar() {
    echo "Projeto : $(pwd) | branch $(git rev-parse --abbrev-ref HEAD 2>/dev/null) $(git rev-parse --short HEAD 2>/dev/null)"
    echo "Python  : $(command -v python)"
    nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader
    python - <<'EOF'
import torch
print(f"PyTorch {torch.__version__} | CUDA {torch.version.cuda} | GPUs visiveis: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"  cuda:{i} {torch.cuda.get_device_name(i)} | bf16 nativo: {torch.cuda.get_device_capability(i)[0] >= 8}")
import sentence_transformers, transformers
print(f"sentence-transformers {sentence_transformers.__version__} | transformers {transformers.__version__}")
from huggingface_hub import snapshot_download
caminho = snapshot_download("Qwen/Qwen3-Embedding-4B", local_files_only=True)
print(f"Modelo no cache: {caminho}")
EOF
    echo "HF_HOME : ${HF_HOME:-padrao (~/.cache/huggingface)}"
    if [ -f "$AMOSTRA" ]; then
        echo "Amostra : $AMOSTRA ($(du -h "$AMOSTRA" | cut -f1))"
    else
        echo "Amostra : ainda nao gerada ($AMOSTRA)"
    fi
    echo "Disco   : $(espaco_livre_gb) GB livres em $(pwd)"
    df -h . "$PARES_DIR" | sed 's/^/  /'
}

# Remove os checkpoint-* de uma variante concluida. No full fine-tuning cada um tem ~24 GB e a
# Etapa 5 avalia o best_model_4b, que ja e o melhor checkpoint. Com LoRA a Etapa 5 avalia o
# proprio checkpoint, entao nada e apagado.
limpar_checkpoints() {
    local pasta="$1"
    [ "$LIMPAR_CHECKPOINTS" = "1" ] || return 0
    [ -d "$pasta/best_model_4b" ] || return 0
    ls -d "$pasta"/checkpoint-* >/dev/null 2>&1 || return 0
    echo "Apagando checkpoints intermediarios de $pasta (o best_model_4b fica)." | tee -a "$LOG"
    rm -rf "$pasta"/checkpoint-*
}

treinar() {
    local limite_padrao="$1" variante pasta
    local limite="${LIMITE_TUPLAS:-$limite_padrao}"
    IFS=',' read -ra lista <<< "$VARIANTES"
    for variante in "${lista[@]}"; do
        pasta="$EXPERIMENTO/$variante"
        if [ -f "$pasta/melhor_checkpoint.json" ]; then
            echo "Variante '$variante' ja concluida ($pasta/melhor_checkpoint.json). Pulando." | tee -a "$LOG"
            limpar_checkpoints "$pasta"
            continue
        fi
        local extra=()
        if [ -n "$limite" ]; then
            extra+=(--limite_tuplas "$limite")
        fi
        if ls -d "$pasta"/checkpoint-* >/dev/null 2>&1; then
            echo "Checkpoint encontrado em $pasta: retomando." | tee -a "$LOG"
            extra+=(--resume_from_checkpoint auto)
        fi
        pipeline --steps train --negativos "$variante" --gpus 0,1 "${extra[@]}"
        limpar_checkpoints "$pasta"
    done
}

echo "node2/rodar.sh $MODO | $(hostname) | $(date)" | tee -a "$LOG"

case "$MODO" in
    checar)
        checar 2>&1 | tee -a "$LOG"
        ;;
    amostra)
        if [ -f "$AMOSTRA" ]; then
            echo "A amostra ja existe: $AMOSTRA" | tee -a "$LOG"
            exit 0
        fi
        exigir_espaco 5
        python -u scripts/amostrar_pares.py "$PARES_DIR/pares_tjrs.jsonl" --n 60000 --somente_camaras 2>&1 | tee -a "$LOG"
        ;;
    smoke)
        exigir_amostra
        exigir_espaco 60
        checar 2>&1 | tee -a "$LOG"
        pipeline --steps split,chunk,mine,random --max_por_tribunal 500 --gpus 0,1
        treinar 640
        pipeline --steps eval --compare_baseline
        ;;
    preparar)
        exigir_amostra
        exigir_espaco 50
        pipeline --steps split,chunk,mine,random --gpus 0,1
        ;;
    treinar)
        exigir_espaco 60
        treinar ""
        ;;
    avaliar)
        pipeline --steps eval --compare_baseline
        ;;
    completo)
        exigir_amostra
        exigir_espaco 130
        checar 2>&1 | tee -a "$LOG"
        pipeline --steps split,chunk,mine,random --gpus 0,1
        treinar ""
        pipeline --steps eval --compare_baseline
        ;;
    *)
        sed -n '2,16p' "$0"
        exit 1
        ;;
esac

echo "Concluido: modo '$MODO' | $(date) | log em $LOG" | tee -a "$LOG"
