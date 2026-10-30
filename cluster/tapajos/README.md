# Tapajos — teste TJMA com 500 processos

Destino no cluster:

```text
/projetos/ctcca/nlp/bumbalm
```

Fila GPU: `accel`, no `gpunode01`, Tesla V100S 32 GB. O teste usa a receita
de `codigos_antigos/train_4b.py`: bfloat16, lote 16, mini-lote 1, acumulacao 4
(lote efetivo 64), checkpoint ligado, AdamW. O walltime e 12 horas.

```powershell
cd E:\BumbaLM-Embedding-4B-Train\BumbaLM-Embedding-4B-Train
$dst = "fabio.lobato@HOST:/projetos/ctcca/nlp/bumbalm"

scp cluster\tapajos\tjma_500.pbs cluster\tapajos\README.md "${dst}/cluster/tapajos/"
scp configs\config_train.yaml "${dst}/configs/"
scp scripts\04_train.py scripts\05_evaluate.py scripts\run_pipeline.py "${dst}/scripts/"
scp src\training\trainer.py src\training\model_builder.py "${dst}/src/training/"
scp src\mining\hard_negatives.py "${dst}/src/mining/"
```

Se o remoto ja estiver nesta branch, um `git pull` depois do push substitui esses `scp`.

## Dados TJMA

```bash
# no cluster
mkdir -p /projetos/ctcca/nlp/bumbalm/data/raw
# copie pares_tjma.jsonl para data/raw/
ls -lh /projetos/ctcca/nlp/bumbalm/data/raw/pares_tjma.jsonl
```

## Submeter o teste

Atualize o codigo no cluster antes do `qsub` (o `.pbs` sozinho nao traz o conserto do treino).
O stderr do job entra no mesmo `.out` (`#PBS -j oe`).

```bash
cd /projetos/ctcca/nlp/bumbalm
mkdir -p cluster/tapajos/logs
qsub cluster/tapajos/tjma_500.pbs
qstat
tail -f cluster/tapajos/logs/tjma_500.out
```

O script apaga `experiments/exp_tjma_500` no inicio. Os dados em `data/processed_tjma/`
podem ficar: split, chunk e mine retomam em poucos segundos.

Saida esperada:

- `data/processed_tjma/`
- `experiments/exp_tjma_500/best_model_4b/`
- `experiments/exp_tjma_500/avaliacoes/`
