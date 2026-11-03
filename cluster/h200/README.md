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
1. **Caminhos:** confira `cluster/h200/caminhos.sh` (conda, nome do ambiente, `HF_HOME`) e `diretorio_base_pares` em `configs/h200/config_base.yaml` e `configs/h200/config_base_smoke.yaml`.
2. **Dados:** copie para essa pasta **o mesmo** `pares_tjrs_amostra60k_camaras.jsonl` gerado no tapajós. Com o mesmo arquivo e a mesma semente (42), o split é idêntico ao da V100.
3. **Ambiente:** precisa de `peft`, `sentence-transformers>=5` e `transformers>=4.51,<5` (`requirements.txt`). Nas H200 prefira torch com CUDA 12.

## gpunode02 sem internet
Os jobs rodam offline (`HF_HUB_OFFLINE=1` em `cluster/h200/caminhos.sh`). Tudo chega pela sua máquina:

| O que | Como |
|---|---|
| Código | `git bundle` da branch, copiado com `scp` e clonado no gpunode02 |
| Amostra TJRS | Gerada no próprio gpunode02 com `scripts/amostrar_pares.py` (só biblioteca padrão), conferindo o SHA-256 com o manifesto do tapajós |
| Modelo | Conferir se já está em `~/bumbalm/hf_cache/hub/models--Qwen--Qwen3-Embedding-4B`; se não, baixar na sua máquina e copiar |
| Pacotes Python | Conferir as versões no ambiente; se faltar algo, `pip download` na sua máquina e `pip install --no-index` lá |

O `peft` só é necessário com LoRA. Neste perfil o LoRA está desligado, então ele não precisa estar instalado.

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

## Regra de ouro da ablação
As três variantes (`nenhum`, `aleatorios`, `dificeis`) precisam rodar **com este mesmo perfil**. Não misture variantes treinadas na V100 com variantes treinadas na H200: a diferença viria do regime de treino, não dos negativos.
