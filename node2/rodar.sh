#!/bin/bash
# Ablacao de negativos do TCC no node2 (2x H200). Rode dentro do tmux, a partir da raiz do projeto:
#
#   bash node2/rodar.sh checar      # GPUs, modelo no cache, amostra e espaco em disco
#   bash node2/rodar.sh testar_modelo # compara as formas de carregar o modelo (~5 min, uma GPU)
#   bash node2/rodar.sh amostra     # sorteia 100 mil acordaos do pares_tjrs.jsonl (sem filtro por orgao)
#   bash node2/rodar.sh smoke       # ponta a ponta com 500 acordaos e 640 tuplas por variante
#   bash node2/rodar.sh preparar    # split, chunk, mineracao dos dificeis e sorteio dos aleatorios
#   bash node2/rodar.sh minerar     # so mineracao e aleatorios (retoma o preparar interrompido na mineracao)
#   bash node2/rodar.sh diagnostico # depois do preparar: base no teste completo + 1 treino curto (~1,5 h)
#   bash node2/rodar.sh treinar     # as tres variantes, uma de cada vez
#   bash node2/rodar.sh avaliar     # teste completo das variantes e do modelo base
#   bash node2/rodar.sh completo    # preparar + treinar + avaliar
#
# Variaveis opcionais: VARIANTES="nenhum,aleatorios,dificeis", LIMITE_TUPLAS=100000 (padrao: todas),
# LIMPAR_CHECKPOINTS=0 (mantem os checkpoint-* depois de cada variante).
# Se o processo cair, rode o mesmo comando de novo: a mineracao retoma do ultimo bloco, o treino
# retoma do ultimo checkpoint e as variantes ja concluidas sao puladas.

set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source node2/env.sh

MODO="${1:-}"
VARIANTES="${VARIANTES:-nenhum,aleatorios,dificeis}"
LIMPAR_CHECKPOINTS="${LIMPAR_CHECKPOINTS:-1}"
# Precisa bater com dados.arquivos_pares de configs/config_base.yaml
AMOSTRA_N=100000
AMOSTRA="$PARES_DIR/pares_tjrs_amostra$((AMOSTRA_N / 1000))k.jsonl"
# Precisa bater com dados.diretorio_processados de configs/config_base.yaml
DADOS="data/processed_tjrs100k"

# Mesmos nomes de saidas.nome_experimento dos configs
CONFIG_BASE="configs/config_base.yaml"
EXPERIMENTO="experiments/tcc_tjrs100k_ablacao_negativos_h200"
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

# Remove os checkpoint-* de uma variante concluida. Com pesos em fp32 cada um tem ~48 GB e a
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

# Treino curto da variante "nenhum" (a mais barata): loss e grad_norm a cada passo e validacao
# a cada 5 passos, comecando no modelo base. Nao grava checkpoint nem modelo.
# A rodada de 03/11/2026 (60 mil, modelo montado errado) ficou em experiments/diagnostico_h200.
DIAGNOSTICO="experiments/diagnostico_h200_100k"
DIAG_TUPLAS="${DIAG_TUPLAS:-5120}"

diagnostico_treino() {
    local nome="$1"
    shift
    if [ -f "$DIAGNOSTICO/$nome/metricas_treino_val.csv" ]; then
        echo "Diagnostico '$nome' ja feito. Pulando." | tee -a "$LOG"
        return 0
    fi
    # Com avaliar_no_inicio e mrr_minimo_passo0, um modelo que ja comeca quebrado para no passo 0
    python -u scripts/04_train.py --negativos nenhum --gpus 0,1 --diagnostico \
        --limite_tuplas "$DIAG_TUPLAS" --eval_steps 5 --warmup_ratio 0.1 \
        --output_dir "$DIAGNOSTICO/$nome" "$@" 2>&1 | tee -a "$LOG"
}

# Tabela por passo: loss, grad_norm, LR e as metricas de validacao
resumo_diagnostico() {
    python - "$DIAGNOSTICO" <<'EOF'
import csv, math, sys
from pathlib import Path

raiz = Path(sys.argv[1])
print("Loss de quem chuta ao acaso na variante 'nenhum' (128 candidatos): ln(128) = %.2f" % math.log(128))
for arq in sorted(raiz.glob("*/metricas_treino_val.csv")):
    linhas = list(csv.DictReader(arq.open(encoding="utf-8")))
    colunas = ["step", "loss", "grad_norm", "learning_rate"]
    colunas += sorted({c for l in linhas for c in l if c.endswith(("ndcg@10", "mrr@10"))})
    print(f"\n=== {arq.parent.name} ===")
    print(" | ".join(c.replace("eval_val_cosine_", "val_") for c in colunas))
    for l in linhas:
        if any(l.get(c) for c in colunas[1:]):
            print(" | ".join((l.get(c) or "-")[:10] for c in colunas))
EOF
}

echo "node2/rodar.sh $MODO | $(hostname) | $(date)" | tee -a "$LOG"

case "$MODO" in
    checar)
        checar 2>&1 | tee -a "$LOG"
        ;;
    testar_modelo)
        # Usa a validacao ja preparada: a nova, se existir, ou a da rodada de 60 mil
        pasta_val="$DADOS"
        [ -f "$pasta_val/val_corpus_chunks.jsonl" ] || pasta_val="data/processed_tjrs"
        CUDA_VISIBLE_DEVICES=0 python -u scripts/testar_carregamento.py --dados "$pasta_val" --processos 100 2>&1 | tee -a "$LOG"
        ;;
    amostra)
        if [ -f "$AMOSTRA" ]; then
            echo "A amostra ja existe: $AMOSTRA" | tee -a "$LOG"
            exit 0
        fi
        exigir_espaco 5
        python -u scripts/amostrar_pares.py "$PARES_DIR/pares_tjrs.jsonl" --n "$AMOSTRA_N" 2>&1 | tee -a "$LOG"
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
    minerar)
        if [ ! -f "$DADOS/train_corpus_chunks.jsonl" ]; then
            echo "ERRO: faltam os chunks. Rode 'bash node2/rodar.sh preparar'." | tee -a "$LOG"
            exit 1
        fi
        exigir_espaco 40
        pipeline --steps mine,random --gpus 0,1
        ;;
    diagnostico)
        if [ ! -f "$DADOS/train_tuplas_aleatorias.jsonl" ]; then
            echo "ERRO: rode antes 'bash node2/rodar.sh preparar' (falta $DADOS/train_tuplas_aleatorias.jsonl)." | tee -a "$LOG"
            exit 1
        fi
        exigir_espaco 20
        # 1. Modelo base no teste completo: quanto sobra de margem para o ajuste mostrar ganho
        if [ ! -f "$DIAGNOSTICO/base_teste/avaliacao_consolidada.json" ]; then
            python -u scripts/05_evaluate.py --split test --compare_baseline \
                --experimento_dir "$DIAGNOSTICO/sem_variantes" \
                --output_dir "$DIAGNOSTICO/base_teste" 2>&1 | tee -a "$LOG"
        fi
        # 2. Treino curto na configuracao do experimento (pesos em fp32 + gradient checkpointing, do
        #    config_train.yaml). O bf16 puro ja foi descartado no diagnostico de 03/11/2026.
        #    Esperado: loss do passo 1 abaixo de ln(128) = 4,85 e validacao do passo 0 perto do base.
        diagnostico_treino config_final
        resumo_diagnostico 2>&1 | tee -a "$LOG" | tee "$DIAGNOSTICO/resumo.txt"
        ;;
    treinar)
        # Ate dois checkpoints de ~48 GB (o melhor e o ultimo) mais um sendo gravado
        exigir_espaco 110
        treinar ""
        ;;
    avaliar)
        pipeline --steps eval --compare_baseline
        ;;
    completo)
        exigir_amostra
        exigir_espaco 150
        checar 2>&1 | tee -a "$LOG"
        pipeline --steps split,chunk,mine,random --gpus 0,1
        treinar ""
        pipeline --steps eval --compare_baseline
        ;;
    *)
        sed -n '2,17p' "$0"
        exit 1
        ;;
esac

echo "Concluido: modo '$MODO' | $(date) | log em $LOG" | tee -a "$LOG"
