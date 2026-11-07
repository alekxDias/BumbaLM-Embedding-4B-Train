# Ablação de negativos no Google Colab (A100 80 GB)

Esta pasta roda o treino e a avaliação das três variantes do TCC (**nenhum**, **difíceis**, **aleatórios**) num Colab Pro com **uma A100 de 80 GB**. Os dados vêm prontos do node2.

O código de treino e avaliação é o mesmo do node2 (`scripts/04_train.py`, `scripts/05_evaluate.py`, `src/`). Esta pasta só tem as configs do Colab, o executor `rodar.py` e o notebook.

| Arquivo | Para que serve |
|---|---|
| `BumbaLM_colab.ipynb` | Notebook com todas as etapas, na ordem |
| `rodar.py` | Executor: `preparar`, `checar`, `testar_modelo`, `calibrar`, `treinar`, `avaliar`, `status` |
| `configs/` | Configs do Colab. São copiadas para o Drive na 1ª sessão, e é a cópia do Drive que vale |
| `../node2/exportar_para_colab.sh` | Empacota no node2 o que o Colab precisa |

## O que muda em relação ao node2, e por quê

| | node2 (2x H200) | Colab (1x A100 80 GB) | Por quê |
|---|---|---|---|
| Ajuste | completo, pesos fp32 | **LoRA** r=16, α=32, todas as camadas lineares | Pesos fp32 + Adam do modelo 4B passam de 64 GB antes das ativações. Um checkpoint teria ~48 GB, contra ~0,4 GB com LoRA |
| Taxa de aprendizado | 2e-5 | **1e-4** | Valor usual para LoRA |
| Lote contrastivo | 2 × 64 (com `gather_across_devices`) | **128 numa GPU** | Cada âncora continua disputando com as outras 127 tuplas |
| Tuplas por variante | todas | **N, definido pela calibração** | Custo. Os N pares são os mesmos nas 3 variantes (sorteio por chave) |

**O que não muda:**
- as tuplas (mineradas no node2), com a faixa [0,60; 0,85], o teto de 0,95 e os 5 negativos;
- a perda CachedMNRL com escala 50;
- o treino: 1 época, cosine com 5% de aquecimento, 768 tokens, a instrução e a semente 42;
- a validação (300 processos, nDCG@10) e a avaliação no teste completo, com arquivos por consulta.

**Os checkpoints do node2 não são usados.** O modelo base e as 3 variantes são treinados e avaliados do zero no Colab.

## Passo 1: exportar do node2 (uma vez)

No node2, a partir da raiz do projeto:

```bash
bash node2/exportar_para_colab.sh
```

O script confere que a mineração terminou e que os negativos aleatórios existem. Depois ele calcula o SHA-256, compacta e grava em `~/bumbalm/export_colab/` (troque o destino com `SAIDA=...`):

| Arquivo | Conteúdo |
|---|---|
| `processed_tjrs100k.tar.gz` (~3 GB) | `train_triplets.jsonl`, `train_tuplas_aleatorias.jsonl`, `val_*`, `test_*`, relatórios, manifesto da amostra e `SHA256SUMS` |
| `pip_freeze_node2.txt` | versões do ambiente do node2. O notebook instala as mesmas |
| `modelo_revisao_node2.txt` | hash do Qwen3-Embedding-4B do node2. O Colab baixa essa revisão |
| `git_node2.txt` | branch e commit do código no node2 |

**Não vão:** checkpoints, `corpus_embeddings_f16.npy`, `train_triplets_consultas/`, `train_processos.jsonl` e `train_corpus_chunks.jsonl`. Eles só servem para minerar de novo.

No seu PC (PowerShell), baixe pelo jump host:

```bash
scp -J fabio.lobato@200.139.13.39 -r fabio.lobato@10.10.151.14:~/bumbalm/export_colab C:\Users\alekx\Downloads\
```

Depois suba os 4 arquivos para o Drive em `MyDrive/BumbaLM_colab/dados/`.

## Passo 2: código no Colab

O notebook clona a branch `tcc/ablacao-negativos-tjrs-h200`, então as mudanças desta pasta precisam estar no commit dessa branch.

Se o repositório for privado, crie o segredo `GITHUB_TOKEN` no Colab. A alternativa é subir um `codigo.zip` do repositório para `MyDrive/BumbaLM_colab/` e usar `USAR_ZIP = True`.

## Passo 3: rodar o notebook

Abra `colab/BumbaLM_colab.ipynb` no Colab, com o ambiente **A100 GPU** e RAM alta. As etapas:

1. **Células 1–4, em toda sessão:** Drive, código e dependências. Depois o `preparar` descompacta os dados no disco local e confere o SHA-256. Ele também baixa o modelo na revisão do node2.
2. **Calibração, uma vez (~1–2 h):**
   - o `calibrar` mede segundos por passo e memória em 3 configurações (mini-lote 16 ou 8, com e sem *gradient checkpointing*);
   - confere a validação do passo 0 com o modelo base, que tem que dar MRR@10 ~0,5;
   - gera `MyDrive/BumbaLM_colab/calibracao/calibracao.txt`: uma tabela com N tuplas, horas, unidades, `save_steps` e `eval_steps`.
3. **Fixar N:** a célula 6 grava N, `save_steps`, `eval_steps` e a configuração de memória na config do Drive. Depois que a 1ª variante começar, não mude mais. O `treinar` recusa hiperparâmetros diferentes entre as variantes, salvo com `--forcar`.
4. **Treino:** primeiro **nenhum**, que é a mais barata. Na 1ª vez, interrompa depois do primeiro checkpoint e rode a célula de novo para testar a retomada. Depois **difíceis** e **aleatórios**.
5. **Avaliação:** o `avaliar` roda o modelo base e as variantes concluídas no teste completo. Pode ser rodado depois de cada variante, porque reaproveita os modelos já avaliados.

## Quando o Colab cai

Rode as células 1–4 e depois a mesma célula de onde parou:
- **Treino:** o `rodar.py` retoma do último checkpoint **completo** no Drive. Um checkpoint gravado pela metade é renomeado para `incompleto-…` e ignorado. Na retomada, a validação do passo 0 não é refeita.
- **Variantes concluídas** (com `melhor_checkpoint.json`) são puladas.
- **Avaliação:** `--pular_avaliados` reaproveita os modelos que já têm `metricas_test_<modelo>.json`.

## Garantias de que a comparação é justa

- **Mesmos pares nas três variantes.** O `limite_tuplas` sorteia pela chave (processo + trecho positivo), entre os pares válidos nos dois arquivos de tuplas, com semente 42. A lista fica em `tuplas_sorteadas_n<N>_s42.txt` na pasta do experimento. O hash dela vai para o `melhor_checkpoint.json` de cada variante. O `status` e o `avaliar` acusam se os hashes forem diferentes.
- **Mesmos hiperparâmetros.** A primeira variante grava `configuracao_congelada.json` e as seguintes são comparadas com ele. Só podem variar as opções que mexem em memória ou velocidade, não no resultado: mini-lote do GradCache, *gradient checkpointing*, *workers* e log.
- **Mesmo modelo base.** A revisão do Hub fica fixa em `MyDrive/BumbaLM_colab/modelo_revisao.txt`. A validação do passo 0 tem que dar o mesmo valor nas três variantes.
- **Mesmos dados do node2.** O SHA-256 de cada arquivo é conferido no `preparar`.

## Pastas no Drive

```
MyDrive/BumbaLM_colab/
  dados/          processed_tjrs100k.tar.gz, pip_freeze_node2.txt, modelo_revisao_node2.txt, git_node2.txt
  configs/        config_base.yaml, config_data.yaml, config_train.yaml (cópia editável; é esta que vale)
  calibracao/     calibracao.txt, calibracao.json e uma rodada por configuração
  experiments/tcc_tjrs100k_ablacao_negativos_colab_a100/
    configuracao_congelada.json, tuplas_sorteadas_n<N>_s42.txt
    nenhum/ dificeis/ aleatorios/   checkpoint-*, modelo_final, melhor_checkpoint.json, trainer_state.json
    avaliacoes/                     comparativo_test.txt, avaliacao_consolidada.json, test/<modelo>/por_consulta_*.jsonl
  logs/           um log por execução do rodar.py
  modelo_revisao.txt
```
