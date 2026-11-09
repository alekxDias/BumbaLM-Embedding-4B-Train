# Ablação de negativos no node2 (2x H200, tmux)

Branch `tcc/ablacao-negativos-h200`. Mesmo pipeline da `tcc/ablacao-negativos-tjma`, ajustado para as duas H200 do node2 e sem PBS:

| | V100S do tapajos (branch `tjma`) | 2x H200 do node2 (esta branch) |
|---|---|---|
| Treino | LoRA, fp16, uma placa | Full fine-tuning como no artigo, pesos em fp32 com autocast bf16, DDP nas duas |
| Lote | 64, GradCache de 4, gradient checkpointing | 64 por placa (128 por passo), GradCache de 16, gradient checkpointing |
| LR | 1e-4 (LoRA) | 2e-5 (do artigo) |
| Mineração | lote de 32 | lote de 256, nas duas placas |
| Tuplas por variante | todas | todas (~195 mil, de 100 mil acórdãos do TJRS) |

| Variante | Negativos no treino |
|---|---|
| `baseline` | Qwen3-Embedding-4B sem ajuste (só avaliação) |
| `nenhum` | Só os do próprio lote (in-batch) |
| `aleatorios` | 5 chunks sorteados de outros processos |
| `dificeis` | 5 hard negatives minerados (faixa 0,60–0,85) |

## Quantas tuplas entram no treino

Uma **tupla** é um exemplo de treino: ementa (âncora), um chunk do teor do mesmo acórdão (positivo) e, conforme a variante, 0 ou 5 negativos. A mineração só gera a tupla quando acha 5 negativos abaixo de 95% da nota do positivo (teto relativo). Por isso cada acórdão de treino rende só ~2,4 tuplas, e não uma por chunk.

| Amostra | Acórdãos de treino / val / teste | Tuplas |
|---|---|---|
| 60 mil (rodada de 03/11/2026, em `data/processed_tjrs`) | 48k / 6k / 6k | 116.419 |
| **100 mil (atual, em `data/processed_tjrs100k`)** | 80k / 10k / 10k | ~195 mil |
| Artigo do TJMA (sem teto relativo nem filtro de vazamento) | — | 212.633 |

`limite_tuplas: null` usa todas. As **mesmas** tuplas entram nas três variantes, com as mesmas âncoras e os mesmos positivos. Só os negativos mudam, e é isso que mantém a ablação justa. Para testar com menos sem editar o YAML: `LIMITE_TUPLAS=50000 bash node2/rodar.sh treinar`.

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

bash node2/rodar.sh checar        # GPUs, modelo no cache, amostra e disco
bash node2/rodar.sh testar_modelo # ~5 min: confere que o treino carrega o modelo igual ao Hub
bash node2/rodar.sh amostra       # uma vez: pares_tjrs_amostra100k.jsonl em ~/bumbalm/Pares_Tribunais
bash node2/rodar.sh preparar      # split, chunks, vetores do corpus, mineração e aleatórios
bash node2/rodar.sh diagnostico   # ~1,5 h: base no teste + treino curto na configuração final
bash node2/rodar.sh treinar       # as três variantes
bash node2/rodar.sh avaliar       # teste completo e tabela comparativa
```

Para sair do tmux sem parar nada: `Ctrl+b` e depois `d`. O log fica em `logs/node2_<modo>_<data>.log`. Para acompanhar de outra janela: `tail -f logs/node2_completo_*.log`.

A rodada completa também pode ir por partes: `preparar`, depois `treinar` e depois `avaliar`. Para uma variante só: `VARIANTES=dificeis bash node2/rodar.sh treinar`.

**Se cair:** rode o mesmo comando de novo. A mineração retoma do último bloco de vetores, o treino retoma do último checkpoint e as variantes já concluídas são puladas.

## 2a. Por que a amostra não filtra por órgão

A rodada de 60 mil usou `--somente_camaras`, que classifica pelas palavras dos primeiros 4.000 caracteres do teor ("turma recursal", "recurso inominado", "juizado especial"). Numa conferência de 30 mil linhas do `pares_tjrs.jsonl`, os acórdãos marcados como Turma Recursal eram, nos exemplos, de Câmaras (3ª Câmara Criminal, 24ª Câmara Cível) que só citavam essas palavras. O grupo tinha teor completo (mediana de 13 mil caracteres) e menos ementa copiada no teor (0,23 contra 0,36). O filtro tirava ~29% dos acórdãos e enviesava os temas. A amostra de 100 mil é aleatória sobre o arquivo todo, e os riscos de atalho são tratados direto no texto: recorte do bloco da ementa, exclusão de chunks com vazamento, quase duplicatas no mesmo split e teto relativo na mineração.

## 2b. Diagnóstico do treino (antes do `treinar`)

O diagnóstico de 03/11/2026 (60 mil acórdãos) mostrou que o modelo do treino já começava quebrado: validação no passo 0 com MRR@10 0,04 e loss 16 no passo 1, contra MRR 0,54 do modelo base no teste. O `model_builder` montava os módulos à mão (`models.Transformer` + `Pooling` + `Normalize`) e, no sentence-transformers do node2, isso não reproduz o modelo do Hub. Agora ele carrega com `SentenceTransformer(model_id)`, como a mineração e a Etapa 5. O mesmo diagnóstico descartou o bf16 puro: com pesos em fp32 o grad_norm ficou 3 a 5x menor e a validação melhor.

O `testar_modelo` carrega o modelo das duas formas e com o `model_builder` corrigido e compara o MRR em 100 processos de validação. O veredito no fim diz se pode seguir.

O `diagnostico` não grava nenhum modelo:

1. Avalia o **modelo base no teste completo** (`experiments/diagnostico_h200_100k/base_teste/`).
2. **config_final**: treino curto da variante `nenhum` (5.120 tuplas, 40 passos) na configuração do experimento, com loss a cada passo e validação no passo 0 e a cada 5 passos.

Como ler `experiments/diagnostico_h200_100k/resumo.txt`:

| O que aparece | Conclusão |
|---|---|
| Validação do passo 0 perto do base e loss do passo 1 abaixo de 4,85 | Modelo montado certo: pode seguir para o `treinar` |
| Validação do passo 40 acima da do passo 0 | Primeiro sinal de que o ajuste ajuda |
| Treino parado com "ERRO: o modelo ja comeca muito abaixo do base" | A trava `mrr_minimo_passo0` pegou um modelo quebrado. Mande a saída do `testar_modelo` |

## 3. Espaço em disco

O `rodar.sh` confere o espaço livre antes de começar e para com erro se faltar.

| Modo | Livre necessário | Onde vai o espaço |
|---|---|---|
| `amostra` | ~5 GB | ~2,5 GB de amostra em `Pares_Tribunais` |
| `preparar` | ~50 GB | ~20 GB de dados (tuplas difíceis e aleatórias ~3,5 GB cada, vetores do corpus ~4,3 GB, chunks) |
| `treinar` | ~145 GB | até 2 checkpoints de ~48 GB (pesos fp32 + Adam) + 3 modelos finais de ~16 GB (gravados em fp32) |

Ao fim de cada variante o script apaga os `checkpoint-*` e mantém o `best_model_4b`, que é o que a Etapa 5 avalia. Para mantê-los: `LIMPAR_CHECKPOINTS=0`. Se o disco apertar, a rodada de 60 mil pode sair depois do `testar_modelo`: `rm -rf data/processed_tjrs data/smoke_tjrs`.

Para trocar para LoRA (checkpoints de ~0,4 GB, treino ~1,3x mais rápido), use `lora.ativo: true` e `learning_rate: 1.0e-4` em `configs/config_train.yaml`. Nesse caso o `peft` precisa estar no ambiente.
