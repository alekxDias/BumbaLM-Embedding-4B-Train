# Treino nas duas GPUs: o que mudou e por quê

Este texto explica, linha a linha, a alteração ainda não commitada que permite treinar o Qwen3-4B nas duas GPUs. O treino em si continua o mesmo. O que entrou é a forma de abrir dois processos, um em cada placa, e de fazer a perda enxergar os exemplos das duas.

Arquivos alterados:

- `configs/config_train.yaml`
- `scripts/04_train.py`
- `scripts/run_pipeline.py`
- `src/training/model_builder.py`
- `src/training/trainer.py`

Como rodar no servidor, sem digitar `torchrun`:

```bash
python scripts/04_train.py --gpus 0,1
```

Uma placa só:

```bash
python scripts/04_train.py --gpus 1
```

O campo `gpus` vazio preserva o comando antigo, inclusive `CUDA_VISIBLE_DEVICES=1 python scripts/04_train.py`.

---

## De onde vem cada ideia

| Trecho do código | Referência |
| --- | --- |
| Relançar com `torch.distributed.run --nproc_per_node` | [Distributed Training — Sentence Transformers](https://sbert.net/docs/sentence_transformer/training/distributed.html) e [torchrun](https://docs.pytorch.org/docs/main/elastic/run.html) |
| `CUDA_VISIBLE_DEVICES=0,1` e o índice `cuda:0` / `cuda:1` | [GPU selection — Hugging Face](https://huggingface.co/docs/transformers/main/gpu_selection) |
| Lote 64 por GPU, passo com 128 | [Trainer — Hugging Face](https://huggingface.co/docs/transformers/main/en/main_classes/trainer) e [issue 2732](https://github.com/UKPLab/sentence-transformers/issues/2732) |
| `gather_across_devices=True` | [CachedMultipleNegativesRankingLoss](https://github.com/huggingface/sentence-transformers/blob/main/sentence_transformers/sentence_transformer/losses/cached_multiple_negatives_ranking.py) e [guia de hardware](https://github.com/huggingface/skills/blob/main/skills/train-sentence-transformers/references/hardware_guide.md) |
| Juntar vetores é tarefa da perda; o DDP junta os gradientes | [GradCache](https://github.com/luyug/GradCache) |
| Avaliador só na primeira GPU; log e gravação só no processo 0 | [Distributed Training — Sentence Transformers](https://sbert.net/docs/sentence_transformer/training/distributed.html) |

O guia do Sentence Transformers recomenda o DDP, não o Data Parallel antigo. O Data Parallel é o que acontece se você só der `python` com as duas placas visíveis: a GPU 0 concentra o trabalho. Por isso o script se relança com `torchrun` em vez de deixar o Python puro decidir.

---

## 1. `configs/config_train.yaml`

```yaml
hardware:
  # Vazio: um processo na GPU ja visivel (CUDA_VISIBLE_DEVICES continua valendo).
  # "1": somente a segunda placa. "0,1": DDP nas duas, 64 exemplos por placa.
  gpus: ""
```

- `hardware:` abre um bloco novo, separado de modelo e de hiperparâmetros. A escolha de placa não é um hiperparâmetro do otimizador.
- O comentário da primeira linha diz o padrão: string vazia não mexe no ambiente. Se você já exportou `CUDA_VISIBLE_DEVICES=1`, essa escolha continua valendo.
- O comentário da segunda linha registra os dois usos previstos. `"1"` é uma placa. `"0,1"` pede o DDP. O 64 é o `batch_size_train` que já existia: ele passa a ser o lote de cada placa, como o Trainer define `per_device_train_batch_size`.
- `gpus: ""` deixa o arquivo explícito sem ligar as duas GPUs sozinho. Quem quiser o padrão das duas troca para `"0,1"` ou usa a flag.

---

## 2. `scripts/04_train.py`

Este arquivo é o que você executa. A maior parte do que entrou está aqui.

### Exemplos no topo do arquivo

```text
python scripts/04_train.py --gpus 1
python scripts/04_train.py --gpus 0,1
```

São só documentação do script. Não executam nada. Mostram os dois comandos novos ao lado do comando antigo.

### `import subprocess`

Entra porque o processo inicial precisa chamar outro programa, o `torchrun`, e esperar o código de saída dele. `subprocess.run` faz essa espera. Sem isso o pai não saberia se o treino nas duas placas terminou bem.

### `_ids_gpus`

```python
def _ids_gpus(texto: str) -> list[str]:
    return [parte.strip() for parte in str(texto or "").split(",") if parte.strip()]
```

- Recebe o texto da flag ou do YAML, por exemplo `"0,1"` ou `"0, 1"`.
- `str(texto or "")` transforma `None` em string vazia, para o `split` não quebrar.
- `split(",")` separa em `"0"` e `"1"`.
- `strip()` tira espaço. O filtro `if parte.strip()` descarta pedaço vazio, como uma vírgula sobrando no fim.
- O resultado é a lista `["0", "1"]`. O tamanho dessa lista decide se haverá um processo ou dois.

### `_ja_distribuido`

```python
def _ja_distribuido() -> bool:
    return os.environ.get("LOCAL_RANK") is not None
```

O `torchrun` cria a variável de ambiente `LOCAL_RANK` em cada processo filho antes de rodar o script de novo. `0` no primeiro, `1` no segundo. Esta função só pergunta se essa variável existe.

Ela existe para impedir um laço. O filho também lê `--gpus 0,1`. Se ele se relançasse outra vez, abriria mais processos para sempre. Ver `LOCAL_RANK` significa: este processo já nasceu de dentro do `torchrun`, então segue para o treino.

Isso vem do contrato do [torchrun](https://docs.pytorch.org/docs/main/elastic/run.html): um processo por GPU, cada um com o próprio rank local.

### `_processo_principal` e `_imprimir`

```python
def _processo_principal() -> bool:
    return int(os.environ.get("RANK", "0")) == 0

def _imprimir(mensagem: str) -> None:
    if _processo_principal():
        print(mensagem)
```

`RANK` também é criado pelo `torchrun`. É o número do processo no grupo inteiro. Numa máquina com duas placas, `RANK` e `LOCAL_RANK` coincidem: 0 e 1.

- Se `RANK` não existe, o padrão `"0"` trata o processo único de sempre como principal.
- `_imprimir` só escreve no processo 0. Os dois processos executam o mesmo arquivo. Sem esse filtro, cada linha de log sairia duplicada.
- O processo 0 é também quem grava o modelo no fim. A documentação do Sentence Transformers coloca o evaluator na primeira GPU pelo mesmo motivo: um só processo fala com o disco e com o relatório.

### `_resolver_gpus`

```python
def _resolver_gpus(args: argparse.Namespace, cfg_train: dict[str, Any]) -> str:
    if args.gpus is not None:
        return args.gpus.strip()
    bruto = cfg_train.get("hardware", {}).get("gpus")
    if bruto is None:
        return ""
    return str(bruto).strip()
```

- A flag `--gpus` ganha da configuração. `args.gpus is not None` distingue "a pessoa não passou a flag" de "a pessoa passou string vazia".
- Sem flag, lê `hardware.gpus` do YAML.
- `bruto is None` cobre YAML sem a chave. Não se usa `or ""` direto no valor, porque o número `0` em YAML seria tratado como ausência e a placa 0 sumiria.
- `str(bruto).strip()` aceita o valor mesmo que o YAML tenha lido um número.

### `_preparar_gpus`

```python
def _preparar_gpus(gpus: str) -> None:
    """Fixa uma GPU ou relanca o script em DDP quando ha duas ou mais."""
    if _ja_distribuido():
        return

    ids = _ids_gpus(gpus)
    if not ids:
        return

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(ids)
    if len(ids) == 1:
        return

    comando = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node",
        str(len(ids)),
        str(Path(__file__).resolve()),
        *sys.argv[1:],
    ]
    print(f"Relancando treino em {len(ids)} GPUs ({','.join(ids)}) com torch.distributed.run ...")
    resultado = subprocess.run(comando)
    sys.exit(resultado.returncode)
```

Linha a linha, para `--gpus 0,1`:

1. `_ja_distribuido()` é falso no processo que você digitou. Segue.
2. `ids` vira `["0", "1"]`.
3. A lista não está vazia, então não retorna.
4. `CUDA_VISIBLE_DEVICES` vira `"0,1"`. É o mesmo efeito do prefixo do comando manual, descrito no [guia de seleção de GPU](https://huggingface.co/docs/transformers/main/gpu_selection). As duas placas físicas ficam visíveis, renumeradas como `cuda:0` e `cuda:1`.
5. Há mais de um id, então não retorna. Com um id só (`--gpus 1`), a função pararia aqui: um processo, uma placa, sem `torchrun`.
6. `sys.executable` é o Python que está rodando o script.
7. `"-m"` e `"torch.distributed.run"` chamam o `torchrun` como módulo. É o mesmo programa do comando `torchrun`, sem depender dele estar no PATH.
8. `"--standalone"` diz que as duas placas estão nesta máquina. Não há outro servidor no grupo.
9. `"--nproc_per_node"` e `str(len(ids))` pedem 2 processos, um por placa. É o `--nproc_per_node=2` da [documentação de treino distribuído](https://sbert.net/docs/sentence_transformer/training/distributed.html).
10. `Path(__file__).resolve()` é o caminho deste próprio `04_train.py`. O `torchrun` vai executá-lo de novo, duas vezes.
11. `*sys.argv[1:]` repete os argumentos que você passou, inclusive `--gpus 0,1`. Os filhos precisam deles.
12. O `print` aparece uma vez, no processo pai, antes dos dois treinos. É a frase `Relancando treino em 2 GPUs (0,1)`.
13. `subprocess.run` espera os dois processos terminarem.
14. `sys.exit` devolve o código do `torchrun` e impede o pai de continuar e treinar uma terceira cópia. Se o treino falhar, o orquestrador vê a falha.

Nos filhos, a função entra de novo, vê `LOCAL_RANK` e retorna na primeira linha. Aí o treino começa.

### `_dispositivo_local`

```python
def _dispositivo_local() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    return f"cuda:{local_rank}"
```

- Sem CUDA, o treino vai para CPU. O caminho das duas GPUs nem chega aqui sem placa.
- `LOCAL_RANK` ausente vira 0. É o caso de uma GPU só: a única placa visível é `cuda:0`, mesmo que no servidor ela seja a física 1, por causa do `CUDA_VISIBLE_DEVICES`.
- `set_device` fixa a placa deste processo antes de carregar o modelo. Sem isso, os dois processos criariam o Qwen3 na placa 0 e a memória estouraria.
- O retorno `cuda:0` ou `cuda:1` é o que segue para `construir_modelo_embedding`.

### Flag `--gpus`

```python
parser.add_argument(
    "--gpus",
    type=str,
    default=None,
    help="GPUs a usar, separadas por virgula. Ex.: '1' ou '0,1'. Vazio mantem o processo unico atual.",
)
```

Registra a flag. `default=None` permite distinguir "não passou" de "passou vazio", que é o que `_resolver_gpus` usa para cair no YAML.

### Chamada no início de `main`

```python
_preparar_gpus(_resolver_gpus(args, cfg_train))
```

Roda logo depois de ler os YAML e antes da semente, do dataset e do modelo. O pai, quando vai relançar, não chega a carregar o Qwen3. Quem carrega o modelo são os filhos, cada um na sua placa.

### Banner e device

```python
dispositivo = _dispositivo_local()
num_processos = max(1, int(os.environ.get("WORLD_SIZE", "1")))
```

- A primeira linha escolhe `cuda:0` ou `cuda:1` como acima.
- `WORLD_SIZE` é outra variável do `torchrun`: quantos processos existem no grupo. Com duas placas vale 2. Sem `torchrun` não existe, e o `"1"` mantém a conta antiga de uma GPU.
- `max(1, ...)` evita divisão por zero se alguém exportar `WORLD_SIZE=0`.

As linhas seguintes trocam `print` por `_imprimir` e acrescentam duas informações:

```text
GPUs no passo        : 2 | Dispositivo local: cuda:0
Batch por GPU        : 64 (Acumulacao: 1)
```

"Batch por GPU" substitui o texto antigo "Batch Size Real", porque 64 deixou de ser o lote do passo inteiro quando há duas placas. O processo 1 não imprime esse bloco.

### Passagem do device

```python
device=dispositivo,
```

Antes, `construir_modelo_embedding` escolhia sozinho `"cuda"`, sempre a placa padrão. Agora recebe a placa deste processo.

### Trocas de `print` por `_imprimir` no restante do `main`

Avaliador, aviso de validação ausente, `--no_eval` e a mensagem de retomar ou disparar o treino. O conteúdo é o mesmo. Só o processo 0 escreve. O avaliador em si continua sendo criado nos dois processos: os dois precisam da mesma programação de avaliação, senão o DDP espera um passo que o outro não dá. Quem executa a validação, segundo a documentação, é a primeira GPU.

### Gravação final

```python
if _processo_principal():
    modelo_salvar = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
    best_model_path = dir_saida / "best_model_4b"
    modelo_salvar.save(str(best_model_path))
    ...
if torch.distributed.is_available() and torch.distributed.is_initialized():
    torch.distributed.barrier()
```

- O `if` impede os dois processos de escrever `best_model_4b/` ao mesmo tempo.
- Depois do `trainer.train()`, o Trainer pode ter embrulhado o modelo em `DistributedDataParallel`. Esse embrulho guarda o SentenceTransformer em `.module`. `hasattr(..., "module")` desfaz o embrulho quando ele existe. Numa GPU só, não existe, e salva `trainer.model` direto.
- Salva-se `trainer.model`, não o objeto `model` de antes do treino, porque o Trainer é quem aplicou o melhor checkpoint da validação.
- Os `print` finais ficam dentro do `if`, então só o processo 0 anuncia o caminho.
- `barrier()` espera os dois. O processo 1 não pode morrer enquanto o 0 ainda grava. `is_initialized()` é falso no treino de uma GPU, e a barreira não roda.

---

## 3. `scripts/run_pipeline.py`

Três acréscimos. Nenhum deles abre o DDP. Só repassa a escolha para o `04_train.py`, que faz o relançamento.

### Exemplo no comentário do arquivo

```text
7b. Treino nas duas GPUs (DDP, lote de 64 por placa):
   python scripts/run_pipeline.py --steps train --gpus 0,1
```

Documentação do orquestrador. O lote citado é o 64 por placa do YAML.

### Flag

```python
parser.add_argument(
    "--gpus",
    type=str,
    default=None,
    help="GPUs da etapa train. Ex.: '1' ou '0,1'. Repassado ao script de treino.",
)
```

A mesma ideia do `04_train.py`. O padrão `None` significa "não sobrescrever o YAML".

### Repasse só na etapa de treino

```python
if args.gpus:
    cmd.extend(["--gpus", args.gpus])
```

Fica dentro do bloco `if etapa == "train"`. Mineração e avaliação continuam num processo só. `if args.gpus` ignora `None` e string vazia. O comando montado fica equivalente a `python scripts/04_train.py --gpus 0,1`.

---

## 4. `src/training/model_builder.py`

O construtor precisava aceitar `cuda:1` como GPU de verdade. Antes, só a string exata `"cuda"` ligava bfloat16 e a atenção rápida.

### Import e documentação

```python
import os
```

Entra para ler `RANK` e calar o log detalhado no processo 1.

A docstring do argumento `device` passa a listar `'cuda'`, `'cuda:0'`, `'cuda:1'` e `'cpu'`.

### Reconhecer qualquer GPU CUDA

```python
em_cuda = str(device).startswith("cuda")
if em_cuda and ":" in str(device):
    torch.cuda.set_device(int(str(device).split(":", 1)[1]))

dtype_modelo = torch.bfloat16 if em_cuda and torch.cuda.is_bf16_supported() else (torch.float16 if em_cuda else torch.float32)
```

- `startswith("cuda")` trata `"cuda"`, `"cuda:0"` e `"cuda:1"` como GPU. A comparação antiga `device == "cuda"` classificaria `cuda:1` como se não fosse CUDA: precisão errada e atenção `eager`.
- Se houver `:`, o número depois dela vira o índice e `set_device` repete a fixação feita em `_dispositivo_local`. O construtor fica seguro mesmo se for chamado direto.
- `split(":", 1)` separa só na primeira ocorrência, então `cuda:1` vira índice `1`.
- A precisão usa `em_cuda`. Nas H200 isso continua bfloat16. Em CPU, float32.

### Log por rank

```python
rank = os.environ.get("RANK", "0")
print(f"[rank {rank}] Construindo modelo em {device}", flush=True)
if rank == "0":
    print(...)
```

- A primeira linha imprime nos dois processos, com `flush=True` para não ficar presa no buffer. É o que você procura no log: `[rank 0] Construindo modelo em cuda:0` e `[rank 1] Construindo modelo em cuda:1`.
- O bloco de precisão, comprimento e checkpoint só sai no rank 0, para não duplicar.

### Atenção e checkpoint

```python
"attn_implementation": attn_implementation if em_cuda else "eager",
```

A atenção `sdpa` do YAML vale para `cuda:0` e `cuda:1`. Antes exigia a string `"cuda"`.

```python
if gradient_checkpointing and em_cuda:
    ...
    if rank == "0":
        print("  - Gradient Checkpointing ativado com sucesso.")
elif rank == "0":
    print("  - Gradient Checkpointing desativado (priorizando velocidade na H200).")
```

A condição de ligar o checkpoint usa `em_cuda` pelo mesmo motivo. O YAML atual deixa isso desligado. Os dois `print` ficaram restritos ao rank 0. O `elif` evita imprimir "desativado" quando o checkpoint acabou de ser ativado.

O print final de parâmetros treináveis também ficou dentro de `if rank == "0"`.

---

## 5. `src/training/trainer.py`

Aqui o lote do passo e a perda passam a conhecer as duas placas.

### Três funções novas

```python
import inspect
import os
```

`os` lê `RANK` e `WORLD_SIZE`. `inspect` olha se a classe da perda aceita `gather_across_devices` na versão instalada no servidor.

```python
def _processo_principal() -> bool:
    return int(os.environ.get("RANK", "0")) == 0

def _imprimir(mensagem: str) -> None:
    if _processo_principal():
        print(mensagem)

def _num_processos() -> int:
    return max(1, int(os.environ.get("WORLD_SIZE", "1")))
```

As duas primeiras repetem a regra do `04_train.py`: log só no processo 0. `_num_processos` é 2 dentro do `torchrun` e 1 fora dele. O dataset carregado também usa `_imprimir`, então a frase "Dataset de Treino Carregado" sai uma vez.

### A perda com os vetores das duas placas

```python
num_processos = _num_processos()
juntar_gpus = num_processos > 1

kwargs_loss: dict[str, Any] = {
    "model": model,
    "mini_batch_size": mini_batch_size,
    "scale": scale,
}
if juntar_gpus:
    parametros_loss = inspect.signature(CachedMultipleNegativesRankingLoss.__init__).parameters
    if "gather_across_devices" in parametros_loss:
        kwargs_loss["gather_across_devices"] = True
    else:
        juntar_gpus = False
        _imprimir(
            "Aviso: esta versao do Sentence Transformers nao aceita gather_across_devices. "
            "Cada GPU usara apenas os negativos do proprio lote."
        )
loss = CachedMultipleNegativesRankingLoss(**kwargs_loss)
```

- Com uma GPU, `juntar_gpus` é falso. O dicionário tem os três argumentos de sempre. O comportamento da perda não muda.
- Com duas, o código consulta a assinatura do construtor. Se o parâmetro existir, liga `gather_across_devices=True`.
- Esse parâmetro, na [implementação da perda](https://github.com/huggingface/sentence-transformers/blob/main/sentence_transformers/sentence_transformer/losses/cached_multiple_negatives_ranking.py), reúne os vetores das placas antes do cálculo. Cada ementa passa a ser comparada com os negativos dos dois lotes: cerca de 127, em vez de 63. O [guia de hardware](https://github.com/huggingface/skills/blob/main/skills/train-sentence-transformers/references/hardware_guide.md) diz que, sem isso, o negativo in-batch fica preso na GPU local.
- O [GradCache](https://github.com/luyug/GradCache) separa as duas comunicações. O DDP, feito pelo Trainer, soma os gradientes. Juntar as representações para a perda contrastiva é esta flag, não o `torchrun`.
- Se a biblioteca do servidor for antiga e não tiver o argumento, `juntar_gpus` volta a falso, o aviso sai no processo 0, e o treino continua com os negativos locais. Não quebra na hora de construir a perda.
- `**kwargs_loss` monta a classe com o dicionário, com ou sem a chave extra.

### Conta de passos

```python
lote_passo = batch_size * gradient_accumulation_steps * num_processos
passos_por_epoca = max(1, len(train_dataset) // lote_passo)
```

Antes o divisor era só `batch_size * gradient_accumulation_steps`. Com 64 e acumulação 1, uma época inteira era `tamanho / 64`.

Com duas GPUs o passo consome 128 exemplos, porque cada placa pega 64 ao mesmo tempo. Essa fórmula é a do [Trainer](https://huggingface.co/docs/transformers/main/en/main_classes/trainer): lote por dispositivo vezes número de dispositivos vezes acumulação. A [issue 2732](https://github.com/UKPLab/sentence-transformers/issues/2732) confirma que o `per_device_train_batch_size` não é o lote global.

`warmup_steps` usa `total_steps`, que agora é a metade do que seria com uma GPU, para o mesmo dataset. Sem esse ajuste, o aquecimento de 5% duraria o dobro dos passos reais.

O argumento passado ao Trainer continua sendo `per_device_train_batch_size=batch_size`. O 64 do YAML não foi dividido. Quem multiplica por 2 é o número de processos.

### Log da configuração

As linhas de configuração passaram para `_imprimir` e ganharam três campos:

- `Batch por GPU`, o 64.
- `Processos (GPUs)`, o `WORLD_SIZE`.
- `Batch do passo`, o produto, 128 com duas placas.
- `Negativos entre GPUs`, verdadeiro só quando o `gather_across_devices` foi mesmo ligado.

### Telemetria só no processo 0

```python
callbacks = [telemetry_callback] if _processo_principal() else []
```

O callback que escreve JSON e CSV entra só na lista do processo 0. O processo 1 cria o objeto, mas não o registra, então não grava `relatorio_treinamento.json` por cima do outro. O `evaluator` continua nos dois, como explicado no `04_train.py`.

---

## O que não mudou

- Mineração (`03_mine_negatives.py`) e avaliação final (`05_evaluate.py`) seguem num processo e numa GPU.
- `batch_size_train: 64` e `mini_batch_size: 16` continuam os valores do YAML. O 16 do GradCache também é por GPU.
- Com `gpus` vazio e sem `--gpus`, não há `torchrun`, não há gather e a contagem de passos é a de uma GPU.
