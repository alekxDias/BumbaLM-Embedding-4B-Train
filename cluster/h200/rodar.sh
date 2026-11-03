#!/bin/bash
# Rodada da ablacao no TJRS nas 2x H200 (perfil configs/h200/).
# Rode da raiz do projeto. Para nao morrer ao fechar o terminal, use tmux/screen ou nohup:
#   nohup bash cluster/h200/rodar.sh tudo > /dev/null 2>&1 &
#
# Modos:
#   smoke                   teste rapido de ponta a ponta (500 processos, 640 tuplas por variante)
#   dados                   etapas 1 e 2 (split por grupo, limpeza e chunks)
#   minerar                 etapas 3 e 3b (hard negatives nas 2 GPUs e negativos aleatorios)
#   treinar <variante>      etapa 4: nenhum | aleatorios | dificeis (retoma do ultimo checkpoint)
#   avaliar                 etapa 5: baseline + variantes no teste
#   tudo                    dados -> minerar -> 3 treinos -> avaliar
set -eo pipefail
cd "$(dirname "$0")/../.."

source cluster/h200/caminhos.sh
conda activate "$AMBIENTE"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
# Ao contrario da V100 (driver 510), aqui o alocador expansivel funciona e reduz fragmentacao.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

BASE="${BASE:-configs/h200/config_base.yaml}"
DATA=configs/h200/config_data.yaml
TRAIN=configs/h200/config_train.yaml
CFG=(--config_base "$BASE" --config_data "$DATA" --config_train "$TRAIN")

mkdir -p logs
MODO="${1:-}"
LOG="logs/h200_${MODO}${2:+_$2}_$(date +%Y%m%d_%H%M%S).log"

info() {
    echo "== $(date '+%F %T') | $*" | tee -a "$LOG"
}

etapa() {
    info "run_pipeline $*"
    python -u scripts/run_pipeline.py "${CFG[@]}" "$@" 2>&1 | tee -a "$LOG"
}

treinar() {
    local variante="$1"
    case "$variante" in
        nenhum|aleatorios|dificeis) ;;
        *) echo "Variante invalida: '$variante' (nenhum, aleatorios ou dificeis)"; exit 1 ;;
    esac
    local experimento
    experimento=$(python -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['saidas']['nome_experimento'])" "$BASE")
    local extra=()
    # Retomar so quando ja existe checkpoint: com a pasta vazia o Trainer aborta.
    if ls -d "experiments/$experimento/$variante"/checkpoint-* >/dev/null 2>&1; then
        info "Checkpoint encontrado em experiments/$experimento/$variante: retomando."
        extra=(--resume_from_checkpoint auto)
    fi
    etapa --steps train --negativos "$variante" "${extra[@]}"
}

info "Modo: $MODO | GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader | paste -sd ',')"
case "$MODO" in
    smoke)
        BASE=configs/h200/config_base_smoke.yaml
        CFG=(--config_base "$BASE" --config_data "$DATA" --config_train "$TRAIN")
        etapa --all --max_por_tribunal 500 --negativos nenhum,aleatorios,dificeis \
              --limite_tuplas 640 --compare_baseline
        ;;
    dados)   etapa --steps split,chunk ;;
    minerar) etapa --steps mine,random ;;
    treinar) treinar "${2:-}" ;;
    avaliar) etapa --steps eval --compare_baseline ;;
    tudo)
        etapa --steps split,chunk
        etapa --steps mine,random
        for v in nenhum aleatorios dificeis; do treinar "$v"; done
        etapa --steps eval --compare_baseline
        ;;
    *)
        sed -n '2,13p' "$0"
        exit 1
        ;;
esac
info "Fim do modo $MODO. Log: $LOG"
