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
#   bash node2/rodar.sh avaliar     # teste completo: base do diagnostico + variantes, uma fila por placa (~8 h nas duas)
#   bash node2/rodar.sh completo    # preparar + treinar + avaliar
#
# Variaveis opcionais: VARIANTES="nenhum,aleatorios,dificeis", LIMITE_TUPLAS=100000 (padrao: todas),
# LIMPAR_CHECKPOINTS=0 (mantem os checkpoint-* depois de cada variante),
# GPUS="1" (placas usadas; padrao "0,1"). O lote do passo (lote_por_passo em configs/config_train.yaml)
# e o mesmo com uma ou duas placas: numa so, cada passo leva ~2x mais tempo.
# Se o processo cair, rode o mesmo comando de novo: a mineracao retoma do ultimo bloco, o treino
# retoma do ultimo checkpoint e as variantes ja concluidas sao puladas.

set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source node2/env.sh

MODO="${1:-}"
VARIANTES="${VARIANTES:-nenhum,aleatorios,dificeis}"
LIMPAR_CHECKPOINTS="${LIMPAR_CHECKPOINTS:-1}"
GPUS="${GPUS:-0,1}"
GPUS="${GPUS// /}"
IFS=',' read -ra PLACAS <<< "$GPUS"
# Todo processo lancado daqui so enxerga as placas de GPUS. Com uma placa visivel a mais, o Trainer
# ligava o DataParallel (lote em dobro e treino ~12x mais lento). A avaliacao troca por placa abaixo.
export CUDA_VISIBLE_DEVICES="$GPUS"
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
AVALIACOES="$EXPERIMENTO/avaliacoes"

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
        pipeline --steps train --negativos "$variante" --gpus "$GPUS" "${extra[@]}"
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
    python -u scripts/04_train.py --negativos nenhum --gpus "$GPUS" --diagnostico \
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

# O diagnostico ja mediu o modelo base no mesmo split de teste (~4 h em fp32). Medir de novo daria o
# mesmo numero. Copia o resultado e grava nele a configuracao lida do log do diagnostico (modelo,
# precisao, tokens, lote) e do relatorio (instrucao), conferindo que e a mesma de agora. A Etapa 5
# ainda confere as contagens do split e as 9.999 consultas antes de usar.
reaproveitar_base() {
    local origem="$DIAGNOSTICO/base_teste/test/baseline" destino="$AVALIACOES/test/baseline"
    [ -f "$destino/metricas_test_baseline.json" ] && return 0
    if [ ! -f "$origem/metricas_test_baseline.json" ]; then
        echo "Sem o base do diagnostico em $origem: o base sera medido junto com as variantes." | tee -a "$LOG"
        return 0
    fi
    mkdir -p "$destino"
    cp "$origem"/* "$destino"/
    if ! python - "$origem" "$destino/metricas_test_baseline.json" <<'EOF' 2>&1 | tee -a "$LOG"; then
import json, re, sys
from pathlib import Path

import yaml

origem, arq = Path(sys.argv[1]), Path(sys.argv[2])
saida = origem.parent.parent.as_posix()  # .../base_teste, o "Diretorio de Saida" do log do diagnostico
cfg = yaml.safe_load(open("configs/config_train.yaml", encoding="utf-8"))
agora = {
    "modelo": cfg["modelo"]["model_id"],
    "precisao": str(cfg.get("avaliacao", {}).get("precisao", "auto")),
    "max_seq_length": int(cfg["modelo"]["max_seq_length"]),
}

registro = None
for log in sorted(Path("logs").glob("node2_diagnostico_*.log")):
    texto = log.read_text(encoding="utf-8", errors="replace")
    linhas_saida = [l for l in texto.splitlines() if l.startswith("Diretorio de Saida")]
    if not any(l.split(":", 1)[1].strip().endswith(saida) for l in linhas_saida):
        continue
    modelo = re.search(r"^Modelo baseline\s*:\s*(\S+)", texto, re.M)
    lote = re.search(r"Batch Size Avaliacao : (\d+) \| Max tokens: (\d+) \| Precisao: (\w+)", texto)
    if modelo and lote:
        registro = {
            "modelo": modelo.group(1),
            "precisao": lote.group(3),
            "max_seq_length": int(lote.group(2)),
            "batch_size": int(lote.group(1)),
            "fonte": f"{log} (resultado copiado de {origem})",
        }
if registro is None:
    sys.exit(f"Base: nao achei em logs/ o log do diagnostico com a configuracao de {saida}.")

diferencas = [f"{k}: diagnostico {registro[k]!r}, agora {v!r}" for k, v in agora.items() if registro[k] != v]
relatorio = (origem / "relatorio_test_baseline.txt").read_text(encoding="utf-8")
instrucao = re.search(r"Instrucao de Query\s*: (.*?)\n\n", relatorio, re.S)
instrucao_agora = cfg["modelo"]["query_instruction"]
if not instrucao or instrucao.group(1).strip() != instrucao_agora.strip():
    diferencas.append("instrucao da consulta diferente da do config_train.yaml")
if diferencas:
    sys.exit("Base: o diagnostico usou outra configuracao (" + "; ".join(diferencas) + ").")

registro["query_instruction"] = instrucao_agora
dados = json.loads(arq.read_text(encoding="utf-8"))
dados["configuracao_avaliacao"] = registro
arq.write_text(json.dumps(dados, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"Base: reaproveitado o resultado do diagnostico ({registro['modelo']}, {registro['precisao']}, "
      f"{registro['max_seq_length']} tokens), conforme {registro['fonte']}.")
EOF
        rm -rf "$destino"
        echo "O base sera medido junto com as variantes." | tee -a "$LOG"
    fi
}

# Etapa 5 no teste completo. A fila (o base, se nao veio do diagnostico, e as variantes concluidas)
# e dividida entre as placas de GPUS, um modelo por vez em cada uma, em fp32 (~4 h por modelo).
# Com o base reaproveitado e duas placas: ~8 h; com uma placa: ~13 h. No fim, uma chamada junta
# tudo no comparativo sem carregar modelo. Se cair, rode de novo: o que ficou pronto e pulado.
avaliar_teste() {
    local fila=() todos=(baseline) por_placa=() pids=() v i p log_placa nomes falhou=0
    [ "$MODO" = "smoke" ] || reaproveitar_base
    [ -f "$AVALIACOES/test/baseline/metricas_test_baseline.json" ] || fila+=(baseline)
    IFS=',' read -ra lista <<< "$VARIANTES"
    for v in "${lista[@]}"; do
        if [ ! -f "$EXPERIMENTO/$v/melhor_checkpoint.json" ]; then
            echo "Variante '$v' sem $EXPERIMENTO/$v/melhor_checkpoint.json: fica fora da avaliacao." | tee -a "$LOG"
            continue
        fi
        todos+=("$v")
        [ -f "$AVALIACOES/test/$v/metricas_test_$v.json" ] || fila+=("$v")
    done
    for i in "${!fila[@]}"; do
        p=$((i % ${#PLACAS[@]}))
        por_placa[$p]="${por_placa[$p]:+${por_placa[$p]},}${fila[$i]}"
    done
    for p in "${!PLACAS[@]}"; do
        [ -n "${por_placa[$p]:-}" ] || continue
        log_placa="${LOG%.log}_placa${PLACAS[$p]}.log"
        echo "Placa ${PLACAS[$p]}: ${por_placa[$p]} | acompanhe com: tail -f $log_placa" | tee -a "$LOG"
        CUDA_VISIBLE_DEVICES="${PLACAS[$p]}" python -u scripts/05_evaluate.py --config_base "$CONFIG_BASE" --split test \
            --experimento_dir "$EXPERIMENTO" --output_dir "$AVALIACOES" --compare_baseline \
            --somente "${por_placa[$p]}" --pular_avaliados --sem_comparativo > "$log_placa" 2>&1 &
        pids+=($!)
    done
    for i in "${!pids[@]}"; do
        wait "${pids[$i]}" || falhou=1
    done
    if [ "$falhou" = 1 ]; then
        echo "ERRO: a avaliacao falhou em uma das placas (veja ${LOG%.log}_placa*.log)." | tee -a "$LOG"
        echo "Rode o mesmo comando de novo: os modelos ja avaliados sao pulados." | tee -a "$LOG"
        exit 1
    fi
    nomes="$(IFS=,; echo "${todos[*]}")"
    python -u scripts/05_evaluate.py --config_base "$CONFIG_BASE" --split test \
        --experimento_dir "$EXPERIMENTO" --output_dir "$AVALIACOES" --compare_baseline \
        --somente "$nomes" --pular_avaliados 2>&1 | tee -a "$LOG"
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
        CUDA_VISIBLE_DEVICES="${PLACAS[0]}" python -u scripts/testar_carregamento.py --dados "$pasta_val" --processos 100 2>&1 | tee -a "$LOG"
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
        pipeline --steps split,chunk,mine,random --max_por_tribunal 500 --gpus "$GPUS"
        treinar 640
        # Mesma avaliacao do full (uma fila por placa e o comparativo no fim), sem o base do diagnostico
        avaliar_teste
        ;;
    preparar)
        exigir_amostra
        exigir_espaco 50
        pipeline --steps split,chunk,mine,random --gpus "$GPUS"
        ;;
    minerar)
        if [ ! -f "$DADOS/train_corpus_chunks.jsonl" ]; then
            echo "ERRO: faltam os chunks. Rode 'bash node2/rodar.sh preparar'." | tee -a "$LOG"
            exit 1
        fi
        exigir_espaco 40
        pipeline --steps mine,random --gpus "$GPUS"
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
        # Ate dois checkpoints de ~48 GB (o melhor e o ultimo) e tres modelos finais de ~16 GB em fp32
        exigir_espaco 145
        treinar ""
        ;;
    avaliar)
        avaliar_teste
        ;;
    completo)
        exigir_amostra
        exigir_espaco 185
        checar 2>&1 | tee -a "$LOG"
        pipeline --steps split,chunk,mine,random --gpus "$GPUS"
        treinar ""
        avaliar_teste
        ;;
    *)
        sed -n '2,17p' "$0"
        exit 1
        ;;
esac

echo "Concluido: modo '$MODO' | $(date) | log em $LOG" | tee -a "$LOG"
