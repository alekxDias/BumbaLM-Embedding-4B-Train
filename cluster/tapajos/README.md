# Tapajos (A100) — deploy TJMA

Destino no cluster:

```text
/projetos/ctcca/nlp/bumbalm
```

Fila GPU: `accel` | Conda: `gpu-test` | Teste inicial: 500 processos TJMA

## Enviar so o .pbs (PowerShell)

Codigo ja clonado no cluster. Para mandar so o job:

```powershell
cd E:\BumbaLM-Embedding-4B-Train\BumbaLM-Embedding-4B-Train

scp cluster\tapajos\tjma_500.pbs fabio.lobato@HOST:/projetos/ctcca/nlp/bumbalm/cluster/tapajos/
```

Troque `HOST` pelo endereco SSH do Tapajos.

## Dados TJMA

```bash
# no cluster
mkdir -p /projetos/ctcca/nlp/bumbalm/data/raw
# copie pares_tjma.jsonl para data/raw/
ls -lh /projetos/ctcca/nlp/bumbalm/data/raw/pares_tjma.jsonl
```

## Submeter o teste

```bash
cd /projetos/ctcca/nlp/bumbalm
mkdir -p cluster/tapajos/logs
qsub cluster/tapajos/tjma_500.pbs
qstat
tail -f cluster/tapajos/logs/tjma_500.out
```

Saida esperada:

- `data/processed_tjma/`
- `experiments/exp_tjma_500/best_model_4b/`
- `experiments/exp_tjma_500/avaliacoes/`
