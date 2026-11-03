# Ablação do TJRS nas 2× H200

Mesmo experimento da branch `tcc/ablacao-negativos-tjma` (TJRS, mesma limpeza, mesmo tratamento de vazamento e de quase duplicatas), com o perfil de hardware das 2× H200 de 141 GB.

| | V100 (tapajós) | 2× H200 (este perfil) |
|---|---|---|
| Configs | `configs/*.yaml` | `configs/h200/*.yaml` |
| Treino | LoRA, fp16, 1 GPU | **Fine-tuning completo** (como no artigo), bf16, DDP em 2 GPUs |
| Lote contrastivo | 64 | 2 × 32 = **64** (mesmo número de concorrentes) |
| LR | 1e-4 (LoRA) | 2e-5 (artigo) |
| Tuplas | pode precisar de `limite_tuplas` | todas |
| Saída | `experiments/tcc_tjrs_ablacao_negativos/` | `experiments/tcc_tjrs_ablacao_negativos_h200/` |

Todas as diferenças entre os perfis são de hardware. Para conferir:
```bash
diff <(python -c "import yaml;print(yaml.safe_load(open('configs/config_train.yaml')))") <(python -c "import yaml;print(yaml.safe_load(open('configs/h200/config_train.yaml')))")
```

## Antes da primeira execução
1. **Ambiente:** `cluster/h200/caminhos.sh` ativa o conda `bumbalm` de `~/miniconda3`, o mesmo da `main`. O cache do Hugging Face é o padrão (`~/.cache/huggingface`), também o mesmo da `main`.
2. **Dados:** `~/bumbalm/Pares_Tribunais/pares_tjrs_amostra60k_camaras.jsonl` (`diretorio_base_pares` em `configs/h200/config_base*.yaml`). Com o mesmo arquivo e a mesma semente (42), o split é idêntico ao da V100.
3. **Pacotes:** `sentence-transformers>=5` e `transformers>=4.51,<5`. O `peft` só é necessário com LoRA, que está desligado neste perfil.

## gpunode02 sem internet
- O código chega por `git bundle` (ver passo a passo abaixo).
- O modelo já está no cache padrão. Os scripts rodam com `HF_HUB_OFFLINE=1`, para não esperar o timeout de rede.
- Pacotes que faltarem: `pip download` na sua máquina e `pip install --no-index` lá.

## Disco
O nó tem pouco espaço livre. Um checkpoint de fine-tuning completo ocupa cerca de 24 GB (pesos + Adam), e o Trainer mantém 2 durante o treino. Com `apagar_checkpoints_ao_fim: true` (`configs/h200/config_train.yaml`), ao fim de cada variante fica só o `best_model_4b` (~8 GB) e o `trainer_state.json` com o histórico. Mesmo assim, o pico durante um treino é de cerca de 80 GB contando os dados: confira `df -h ~` antes da rodada real.

## Execução
Da raiz do projeto, dentro de `tmux` ou `screen` (ou com `nohup`), para não perder o processo ao fechar o terminal:

```bash
bash cluster/h200/rodar.sh smoke              # teste rápido: 500 processos, 640 tuplas por variante
bash cluster/h200/rodar.sh tudo               # dados -> mineração -> 3 treinos -> avaliação
```

Ou por etapa:
```bash
bash cluster/h200/rodar.sh dados
bash cluster/h200/rodar.sh minerar
bash cluster/h200/rodar.sh treinar nenhum
bash cluster/h200/rodar.sh treinar aleatorios
bash cluster/h200/rodar.sh treinar dificeis
bash cluster/h200/rodar.sh avaliar
```

- Logs em `logs/h200_<modo>_<data>.log`.
- Se um treino cair, rode `treinar <variante>` de novo: ele retoma do último checkpoint. A mineração também retoma sozinha.

## O que conferir no log do smoke
- **`Processos (GPUs) : 2`** e **`Negativos entre GPUs : True`**. Se aparecer `False`, a versão do sentence-transformers não junta os lotes das duas placas, e cada âncora vê só 31 concorrentes em vez de 63. Atualize a biblioteca antes da rodada real.
- **`Precisao mista : bf16`** e **`Estrategia : Full Fine-Tuning`**.
- **O tempo por passo** (`s/it`), para estimar a rodada inteira.
- No relatório da mineração (`data/smoke_tjrs_h200/relatorio_mineracao_tripletos.txt`): quantas tuplas o teto relativo descartou e quantas ficaram com negativo acima do positivo (deve ser 0).

## Regra de ouro da ablação
As três variantes (`nenhum`, `aleatorios`, `dificeis`) precisam rodar **com este mesmo perfil**. Não misture variantes treinadas na V100 com variantes treinadas na H200: a diferença viria do regime de treino, não dos negativos.
