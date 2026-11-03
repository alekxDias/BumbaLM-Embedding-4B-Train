# Rodando a ablação do TCC no cluster (tapajos)

Ablação de negativos com os pares do TJMA, numa Tesla V100S de 32 GB:

| Variante | Negativos no treino |
|---|---|
| `baseline` | Qwen3-Embedding-4B sem ajuste (só avaliação) |
| `nenhum` | Só os do próprio lote (in-batch) |
| `aleatorios` | 5 chunks sorteados de outros processos |
| `dificeis` | 5 hard negatives minerados (faixa 0,60–0,85), como no artigo |

As três variantes treinadas usam as mesmas âncoras e positivos.

---

## 1. O que é um `.pbs`, em 1 minuto

O cluster não roda os programas direto no terminal em que você entra (o **nó de login**). Você descreve o trabalho num script `.pbs` e o entrega ao **escalonador** (PBS/Torque). Ele espera um **nó de computação** com GPU ficar livre e roda o script lá.

As linhas `#PBS` no topo são pedidos ao escalonador:

| Linha | Significado |
|---|---|
| `#PBS -N bumbalm_treino` | Nome do job (aparece no `qstat`) |
| `#PBS -q accel` | Fila. `accel` é a das GPUs |
| `#PBS -l nodes=1:ppn=15` | 1 nó, 15 núcleos de CPU |
| `#PBS -l walltime=72:00:00` | Tempo máximo. Ao estourar, o job é morto |
| `#PBS -j oe` / `#PBS -o arquivo` | Junta saída e erro num arquivo, copiado **só no fim** do job |

Comandos do dia a dia, rodados no nó de login:

```bash
qsub cluster/pbs/00_smoke.pbs                 # submete; devolve o ID do job
qsub -v NEGATIVOS=nenhum cluster/pbs/03_treinar.pbs   # passa variáveis ao job
qstat -u $USER                                # seus jobs: Q = na fila, R = rodando, C = terminou
qdel <ID>                                     # cancela
tail -f logs/bumbalm_treino_<ID>.log          # acompanha ao vivo
```

Os jobs gravam o log ao vivo em `logs/<nome>_<ID>.log`. O `logs/<nome>.pbs.log` só aparece quando o job termina.

**Sempre rode o `qsub` a partir da raiz do projeto.** Os jobs entram na pasta de onde foram submetidos (`$PBS_O_WORKDIR`).

---

## 2. Preparação (uma vez, no nó de login)

```bash
cd /projetos/ctcca/nlp
git clone -b tcc/ablacao-negativos-tjma https://github.com/alekxDias/BumbaLM-Embedding-4B-Train.git
cd BumbaLM-Embedding-4B-Train

bash cluster/criar_ambiente.sh   # cria o conda em /projetos/ctcca/nlp/conda_envs/bumbalm (python 3.11, torch 2.7.1+cu118, peft...)
bash cluster/baixar_modelo.sh    # baixa o Qwen3-Embedding-4B (~8 GB) para /projetos/ctcca/nlp/hf_cache
```

- Os pares são lidos de `/projetos/ctcca/nlp/dados_tjrs/pares_tjrs_amostra60k_camaras.jsonl`, gerado por `scripts/amostrar_pares.py` (`configs/config_base.yaml`).
- Os jobs rodam **offline** (`HF_HUB_OFFLINE=1` em `cluster/env.sh`), porque nós de computação costumam não ter internet. Por isso o modelo é baixado antes.
- **O `/home` tem cota pequena.** Por isso o ambiente conda, os caches do pip e do conda e o modelo ficam em `/projetos/ctcca/nlp`. Todos esses caminhos estão em `cluster/caminhos.sh`.
- O pip da conta aponta para `pypi.ngc.nvidia.com`, inacessível daqui. O script usa `pip --isolated` para ignorar essa configuração.

Se aparecer `Cota da disco excedida`, veja o uso com `quota -s` e `du -sh ~/.cache ~/miniconda3/pkgs ~/miniconda3/envs/*`.

---

## 3. Teste rápido (obrigatório antes da rodada real)

```bash
qsub cluster/pbs/00_smoke.pbs
```

O teste roda tudo com 500 processos e 640 tuplas por variante, em pastas separadas (`data/smoke_tjrs`, `experiments/smoke_tjrs`). Ele confere:

1. Se o job recebeu a GPU: a primeira linha do log mostra `Tesla V100S`. Se aparecer `NENHUMA`, a fila precisa de um pedido explícito, provavelmente `#PBS -l nodes=1:ppn=15:gpus=1`. Confirme com o orientador.
2. Se cabe na memória (LoRA + fp16 + gradient checkpointing).
3. **Quanto tempo leva cada passo** de treino. O log do Trainer mostra `s/it`. Com isso escolhemos `LIMITE_TUPLAS` e o `walltime`.

---

## 4. Rodada real

Tudo encadeado (cada job espera o anterior terminar sem erro):

```bash
bash cluster/submeter_tudo.sh 50000   # 50 mil tuplas por variante; sem número = todas
```

Ou passo a passo:

```bash
qsub cluster/pbs/01_preparar_dados.pbs          # split 80/10/10 + chunks
qsub cluster/pbs/02_minerar.pbs                 # hard negatives + negativos aleatórios
qsub -v NEGATIVOS=nenhum,LIMITE_TUPLAS=50000     cluster/pbs/03_treinar.pbs
qsub -v NEGATIVOS=aleatorios,LIMITE_TUPLAS=50000 cluster/pbs/03_treinar.pbs
qsub -v NEGATIVOS=dificeis,LIMITE_TUPLAS=50000   cluster/pbs/03_treinar.pbs
qsub cluster/pbs/04_avaliar.pbs                 # baseline + 3 variantes no teste
```

- **Use o mesmo `LIMITE_TUPLAS` nas três variantes**, senão a comparação deixa de ser justa.
- Se um treino estourar o `walltime`, submeta de novo com os mesmos valores: ele retoma do último checkpoint. A mineração também retoma sozinha.

## 5. Resultados

| Arquivo | Conteúdo |
|---|---|
| `experiments/tcc_tjrs_ablacao_negativos/avaliacoes/comparativo_test.txt` | Tabela baseline × nenhum × aleatorios × dificeis, com ganho sobre o baseline |
| `.../avaliacoes/avaliacao_consolidada.json` | Todas as métricas |
| `.../<variante>/metricas_treino_val.csv` | Curva de validação por passo (para os gráficos) |
| `.../<variante>/melhor_checkpoint.json` | Checkpoint escolhido pela validação e quantas tuplas treinaram |
| `data/processed_tjrs/relatorio_*.txt` | Contagens exatas de cada etapa (pedido do Revisor 1) |
