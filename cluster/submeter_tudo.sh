#!/bin/bash
# Submete a rodada completa encadeada: cada job so comeca se o anterior terminar sem erro.
#   bash cluster/submeter_tudo.sh            # todas as tuplas
#   bash cluster/submeter_tudo.sh 50000      # sorteia 50 mil tuplas por variante
set -e
cd "$(dirname "$0")/.."
mkdir -p logs
LIMITE="$1"

J_DADOS=$(qsub cluster/pbs/01_preparar_dados.pbs)
J_MINERAR=$(qsub -W depend=afterok:$J_DADOS cluster/pbs/02_minerar.pbs)
echo "Dados   : $J_DADOS"
echo "Mineracao: $J_MINERAR"

TREINOS=""
for V in nenhum aleatorios dificeis; do
    VARS="NEGATIVOS=$V"
    [ -n "$LIMITE" ] && VARS="$VARS,LIMITE_TUPLAS=$LIMITE"
    J=$(qsub -W depend=afterok:$J_MINERAR -v "$VARS" cluster/pbs/03_treinar.pbs)
    echo "Treino $V: $J"
    TREINOS="$TREINOS:$J"
done

J_AVALIAR=$(qsub -W depend=afterok$TREINOS cluster/pbs/04_avaliar.pbs)
echo "Avaliacao: $J_AVALIAR"
echo "Acompanhe com: qstat -u \$USER"
