#!/bin/bash
# Empacota no node2 o que o Colab precisa para treinar e avaliar as tres variantes (colab/README.md).
# Rode a partir da raiz do projeto no node2:
#
#   bash node2/exportar_para_colab.sh
#
# Gera em ~/bumbalm/export_colab (troque com SAIDA=...):
#   processed_tjrs100k.tar.gz   tuplas (dificeis e aleatorias), validacao, teste, relatorios e SHA256SUMS
#   pip_freeze_node2.txt        versoes do ambiente do node2 (o Colab compara no 'checar')
#   modelo_revisao_node2.txt    hash do snapshot do Qwen3-Embedding-4B usado no node2
#   git_node2.txt               branch e commit do codigo no node2
#
# NAO vao: checkpoints, embeddings do corpus, consultas vetorizadas, train_processos e train_corpus_chunks
# (so servem para minerar de novo). Os checkpoints do node2 nao sao usados no Colab.

set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source node2/env.sh

DADOS="${DADOS:-data/processed_tjrs100k}"
SAIDA="${SAIDA:-$HOME/bumbalm/export_colab}"
NOME="$(basename "$DADOS")"
AMOSTRA_MANIFESTO="${AMOSTRA_MANIFESTO:-$PARES_DIR/pares_tjrs_amostra100k.manifest.json}"

ESSENCIAIS=(train_triplets.jsonl train_tuplas_aleatorias.jsonl val_processos.jsonl val_corpus_chunks.jsonl
            test_processos.jsonl test_corpus_chunks.jsonl)
DOCUMENTOS=(relatorio_preparacao_splits.txt relatorio_auditoria_chunks.txt relatorio_mineracao_tripletos.txt
            train_triplets.params.json)

for arq in "${ESSENCIAIS[@]}"; do
    if [ ! -f "$DADOS/$arq" ]; then
        echo "ERRO: falta $DADOS/$arq."
        if [ "$arq" = "train_tuplas_aleatorias.jsonl" ]; then
            echo "Gere os negativos aleatorios (CPU, ~1-2 min):"
            echo "  python -u scripts/run_pipeline.py --config_base configs/config_base.yaml --steps random"
        fi
        exit 1
    fi
done
# A mineracao tem que ter terminado: processos concluidos no .progress.json = processos de treino
if [ -f "$DADOS/train_triplets.progress.json" ] && [ -f "$DADOS/train_processos.jsonl" ]; then
    python - "$DADOS/train_triplets.progress.json" "$(wc -l < "$DADOS/train_processos.jsonl")" <<'EOF'
import json, sys
p = json.load(open(sys.argv[1], encoding="utf-8"))
feito, total = int(p.get("n_processos_concluidos", 0)), int(sys.argv[2])
print(f"Mineracao: {feito:,} de {total:,} processos de treino | {p.get('tripletos_salvos', 0):,} tuplas")
if feito < total:
    sys.exit("ERRO: a mineracao nao terminou. Rode 'bash node2/rodar.sh minerar' antes de exportar.")
EOF
fi

mkdir -p "$SAIDA"
livre_gb=$(df -Pk "$SAIDA" | awk 'NR==2 {print int($4 / 1024 / 1024)}')
if [ "$livre_gb" -lt 15 ]; then
    echo "ERRO: so $livre_gb GB livres em $SAIDA (precisa de ~15 GB)."
    exit 1
fi

# Arquivos do pacote (os documentos so se existirem) e o manifesto da amostra, copiado para dentro dos dados
LISTA=("${ESSENCIAIS[@]}")
for arq in "${DOCUMENTOS[@]}"; do
    [ -f "$DADOS/$arq" ] && LISTA+=("$arq")
done
if [ -f "$AMOSTRA_MANIFESTO" ]; then
    cp -f "$AMOSTRA_MANIFESTO" "$DADOS/amostra.manifest.json"
    LISTA+=(amostra.manifest.json)
fi

echo "Calculando SHA-256 de ${#LISTA[@]} arquivos ..."
( cd "$DADOS" && sha256sum "${LISTA[@]}" > SHA256SUMS )
cat "$DADOS/SHA256SUMS"
LISTA+=(SHA256SUMS)

echo "Linhas por arquivo:"
for arq in "${ESSENCIAIS[@]}"; do
    printf '  %-34s %12s\n' "$arq" "$(wc -l < "$DADOS/$arq")"
done

COMPACTADOR="gzip"
command -v pigz >/dev/null 2>&1 && COMPACTADOR="pigz"
echo "Compactando com $COMPACTADOR (alguns minutos) ..."
caminhos=()
for arq in "${LISTA[@]}"; do caminhos+=("$NOME/$arq"); done
tar -C "$(dirname "$DADOS")" -cf - "${caminhos[@]}" | "$COMPACTADOR" > "$SAIDA/$NOME.tar.gz"

# Ambiente e modelo do node2, para o Colab repetir as versoes e a mesma revisao do modelo
pip freeze > "$SAIDA/pip_freeze_node2.txt" 2>/dev/null || true
python -c "import torch, sys; print(f'torch=={torch.__version__}'); print(f'python {sys.version.split()[0]}')" \
    >> "$SAIDA/pip_freeze_node2.txt" 2>/dev/null || true
CACHE="${HF_HOME:-$HOME/.cache/huggingface}/hub/models--Qwen--Qwen3-Embedding-4B"
if [ -f "$CACHE/refs/main" ]; then
    cat "$CACHE/refs/main" > "$SAIDA/modelo_revisao_node2.txt"
else
    ls "$CACHE/snapshots" 2>/dev/null | head -1 > "$SAIDA/modelo_revisao_node2.txt" || true
fi
{ git rev-parse --abbrev-ref HEAD; git rev-parse HEAD; git status --short; } > "$SAIDA/git_node2.txt" 2>/dev/null || true

echo
ls -lh "$SAIDA"
echo
echo "Revisao do modelo no node2: $(cat "$SAIDA/modelo_revisao_node2.txt" 2>/dev/null)"
echo "  -> copie este hash para modelo.revisao_hub em colab/configs/config_train.yaml (ou na copia do Drive)."
echo
echo "Baixe para o seu PC (PowerShell), pelo jump host:"
echo "  scp -J fabio.lobato@200.139.13.39 -r fabio.lobato@10.10.151.14:$SAIDA C:\\Users\\alekx\\Downloads\\"
echo "Depois suba os arquivos de export_colab para o Drive em: MyDrive/BumbaLM_colab/dados/"
