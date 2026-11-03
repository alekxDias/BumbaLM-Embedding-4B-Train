# Ablação de negativos no node2 (2x H200, tmux)

Branch `tcc/ablacao-negativos-h200`. Mesmo pipeline da `tcc/ablacao-negativos-tjma`, ajustado para as duas H200 do node2 e sem PBS:

| | V100S do tapajos (branch `tjma`) | 2x H200 do node2 (esta branch) |
|---|---|---|
| Treino | LoRA, fp16, uma placa | Full fine-tuning como no artigo, bf16, DDP nas duas |
| Lote | 64, GradCache de 4, gradient checkpointing | 64 por placa (128 por passo), GradCache de 16, sem checkpointing |
| LR | 1e-4 (LoRA) | 2e-5 (do artigo) |
| Mineração | lote de 32 | lote de 256, nas duas placas |
| Tuplas por variante | todas | 100 mil sorteadas (`limite_tuplas`) |

| Variante | Negativos no treino |
|---|---|
| `baseline` | Qwen3-Embedding-4B sem ajuste (só avaliação) |
| `nenhum` | Só os do próprio lote (in-batch) |
| `aleatorios` | 5 chunks sorteados de outros processos |
| `dificeis` | 5 hard negatives minerados (faixa 0,60–0,85) |

## O que é o `limite_tuplas`

Uma **tupla** é um exemplo de treino: ementa (âncora), um chunk do teor do mesmo acórdão (positivo) e, conforme a variante, 0 ou 5 negativos. Cada acórdão vira ~10 tuplas, uma por chunk. Por isso os 60 mil acórdãos geram ~500 mil tuplas.

O `limite_tuplas: 100000` sorteia 100 mil dessas tuplas com a semente 42 e treina só com elas. As **mesmas** 100 mil entram nas três variantes, com as mesmas âncoras e os mesmos positivos. Só os negativos mudam, e é isso que mantém a ablação justa. O sorteio é feito no treino: a mineração continua usando o corpus inteiro para buscar os negativos.

| Tuplas por variante | `nenhum` | `aleatorios` / `dificeis` |
|---|---|---|
| 100 mil (padrão) | ~3 h | ~8 h cada |
| ~500 mil (todas) | ~12 h | 30 a 40 h cada |

São estimativas. O tempo real por passo aparece no smoke test e no log do treino. Para mudar o limite sem editar o YAML: `LIMITE_TUPLAS=50000 bash node2/rodar.sh treinar`.

## 1. Levar o código ao node2

O node2 não tem internet. Se o `~/bumbalm` do node2 for o mesmo do tapajos (`/home` compartilhado), clone no tapajos:

```bash
ssh fabio.lobato@200.139.13.39
cd ~/bumbalm
git clone -b tcc/ablacao-negativos-h200 https://github.com/alekxDias/BumbaLM-Embedding-4B-Train.git BumbaLM-h200
```

Se não for compartilhado, gere um pacote no seu computador e envie pulando pelo tapajos:

```bash
git archive --format=tar.gz -o bumbalm-h200.tar.gz tcc/ablacao-negativos-h200
scp -J fabio.lobato@200.139.13.39 bumbalm-h200.tar.gz fabio.lobato@10.10.151.14:~/bumbalm/
ssh -J fabio.lobato@200.139.13.39 fabio.lobato@10.10.151.14 "mkdir -p ~/bumbalm/BumbaLM-h200 && tar xzf ~/bumbalm/bumbalm-h200.tar.gz -C ~/bumbalm/BumbaLM-h200"
```

Use o mesmo ambiente conda da main (`bumbalm`). O full fine-tuning não precisa do `peft`.

## 2. Rodar no tmux

```bash
ssh fabio.lobato@200.139.13.39
ssh fabio.lobato@10.10.151.14
tmux new -s bumbalm          # ou: tmux attach -t bumbalm
cd ~/bumbalm/BumbaLM-h200

bash node2/rodar.sh checar   # GPUs, modelo no cache, amostra e disco
bash node2/rodar.sh amostra  # uma vez: pares_tjrs_amostra60k_camaras.jsonl em ~/bumbalm/Pares_Tribunais
bash node2/rodar.sh smoke    # ~30-60 min: valida tudo com 500 acordaos e 640 tuplas por variante
bash node2/rodar.sh completo # rodada do TCC
```

Para sair do tmux sem parar nada: `Ctrl+b` e depois `d`. O log fica em `logs/node2_<modo>_<data>.log`. Para acompanhar de outra janela: `tail -f logs/node2_completo_*.log`.

A rodada completa também pode ir por partes: `preparar`, depois `treinar` e depois `avaliar`. Para uma variante só: `VARIANTES=dificeis bash node2/rodar.sh treinar`.

**Se cair:** rode o mesmo comando de novo. A mineração retoma do último bloco de vetores, o treino retoma do último checkpoint e as variantes já concluídas são puladas.

## 3. Espaço em disco

O `rodar.sh` confere o espaço livre antes de começar e para com erro se faltar.

| Modo | Livre necessário | Onde vai o espaço |
|---|---|---|
| `amostra` | ~5 GB | ~1,4 GB de amostra em `Pares_Tribunais` |
| `smoke` | ~60 GB | 1 checkpoint full FT (~24 GB) por vez + 3 modelos finais de 8 GB |
| `completo` | ~130 GB | ~40 GB de dados (tuplas com texto dos negativos ~24 GB, embeddings ~3 GB, chunks ~3 GB) + 2 checkpoints de ~24 GB + 3 modelos finais de 8 GB |

No full fine-tuning, cada `checkpoint-*` tem ~24 GB (pesos + Adam). Ao fim de cada variante o script apaga os `checkpoint-*` e mantém o `best_model_4b`, que é o que a Etapa 5 avalia. Para mantê-los: `LIMPAR_CHECKPOINTS=0`.

Para trocar para LoRA (checkpoints de ~0,4 GB, treino ~1,3x mais rápido), use `lora.ativo: true` e `learning_rate: 1.0e-4` em `configs/config_train.yaml`. Nesse caso o `peft` precisa estar no ambiente.
