# BumbaLM-Embedding-4B-Train

O pipeline realiza o fine-tuning supervisionado do modelo base `Qwen/Qwen3-Embedding-4B` utilizando a funcao de perda contrastiva com cache (`CachedMultipleNegativesRankingLoss`).

---

## Visao Geral da Arquitetura

1. **Vazamento de dados**:
   A divisao dos dados em Treino (80%), Validacao (10%) e Teste Cego (10%) e agrupada estritamente pelo identificador unico do processo judicial (`id_global`), garantindo que nenhuma parte de uma decisao de treino apareca nos conjuntos de teste ou validacao. A distribuicao proporcional e preservada.

2. **Auditoria e remocao do Chunk 0**:
    O cabecalho da decisao frequentemente reproduz o texto da ementa. O pipeline identifica automaticamente essas reproducoes integrais ou parciais longas, marcando esses fragmentos.

3. **Mineracao de Negativos Dificeis (Hard Negatives)**:
    Calcula a matriz de similaridade e aplica mascaras booleanas (eliminando chunks do proprio processo) inteiramente na memoria da GPU em precisao `bfloat16`, empregando `torch.topk`. A similaridade contra o chunk positivo e restrita apenas aos top candidatos, reduzindo o custo computacional de O(N) para O(K).

4. **Treinamento Otimizado**:
   Explora a VRAM massiva com lote real expandido (`batch_size = 96`), sem necessidade de acumulacao de gradientes (`gradient_accumulation_steps = 1`) ou gradient checkpointing desnecessario.

5. **Avaliacao de Information Retrieval (IR)**:
   Avaliacao direta sobre o conjunto de teste dos tribunais, mensurando nDCG@10, MRR@10, MAP@100 e Recall@1/5/10 com comparativo direto contra o modelo base zero-shot.

---

## Estrutura do Repositorio

```text
BumbaLM-Embedding-4B-Train/
├── configs/                      # Arquivos de configuracao declarativa (YAML)
│   ├── config_base.yaml          # Configuracoes gerais, sementes, caminhos e mapeamento de colunas
│   ├── config_data.yaml          # Parametros de limpeza, tokenizacao, chunking e mineracao
│   └── config_train.yaml         # Hiperparametros de treinamento, hardware H200 e logging
│
├── src/                          # Codigo-fonte modular da aplicacao
│   ├── data/
│   │   ├── schema.py             # Modelos de dados
│   │   ├── cleaner.py            # Limpeza profunda
│   │   ├── splitter.py           # Divisao estratificada Train/Val/Test isolada por processo
│   │   └── chunker.py            # Fragmentacao do inteiro teor em janelas deslizantes de tokens
│   │
│   ├── mining/
│   │   └── hard_negatives.py     # Mineracao de negativos dificeis
│   │
│   ├── evaluation/
│   │   ├── leakage_audit.py      # Deteccao automatica de vazamento de ementa
│   │   └── ir_evaluator.py       # Avaliador Information Retrieval com descontaminacao
│   │
│   └── training/
│       ├── model_builder.py      # Construcao do SentenceTransformer
│       ├── trainer.py            # Treinamento com CachedMNRL e NO_DUPLICATES sampler
│       └── telemetry.py          # Callback de telemetria de hardware
│
├── scripts/                      # Scripts executaveis de cada etapa do pipeline
│   ├── 01_prepare_and_split.py   # Etapa 1: Preparacao, limpeza e divisao dos dados
│   ├── 02_generate_chunks.py     # Etapa 2: Geracao dos chunks de inteiro teor e relatorio de auditoria
│   ├── 03_mine_negatives.py      # Etapa 3: Mineracao de hard negatives no split de treino
│   ├── 04_train.py               # Etapa 4: Execucao do fine-tuning contrastivo com telemetria
│   ├── 05_evaluate.py            # Etapa 5: Avaliacao de IR no conjunto de teste e comparativo A/B
│   └── run_pipeline.py           # Orquestrador master
│
├── data/                         # Diretorio padrao de dados locais
│   ├── raw/                      # Arquivos brutos fornecidos pelo usuario
│   └── processed/                # Dados particionados, chunks e tripletos gerados
│
└── experiments/                  # Diretorio de execucoes 
```

---

## Pipeline

### Etapa 1: Preparacao, Limpeza e Divisao dos Dados
- **Script**: `scripts/01_prepare_and_split.py`
- **Descricao**:
  - Le o arquivo bruto de processos em formato JSONL. O caminho e configurado em `configs/config_base.yaml` ou fornecido via `--input`.
  - Mapeia dinamicamente os nomes dos campos para o esquema canonico (`str_tribunal`, `str_numero_unico`, `str_ementa`, `str_inteiro_teor`). O campo de tribunal e opcional: se ausente, o identificador sera composto apenas pelo numero do processo.
  - Executa limpeza profunda do texto: desescapamento de entidades HTML, remocao de tags, normalizacao Unicode NFKC, correcao de palavras partidas por hifen em quebras de linha (`juris- \nprudencia` -> `jurisprudencia`), reducao de separadores repetitivos e remocao de carimbos de assinatura eletronica.
  - Divide a base em `train_processos.jsonl` (80%), `val_processos.jsonl` (10%) e `test_processos.jsonl` (10%), com garantia estrita de isolamento de processo e distribuicao percentual balanceada por tribunal.

### Etapa 2: Geracao de Chunks por Tokens e Auditoria de Ementa
- **Script**: `scripts/02_generate_chunks.py`
- **Descricao**:
  - Carrega o tokenizador nativo do modelo base (`Qwen/Qwen3-Embedding-4B`).
  - Fragmenta o inteiro teor em janelas deslizantes de 768 tokens com sobreposicao (overlap) de 128 tokens.
  - Audita cada chunk contra a ementa do mesmo processo, detectando copias integrais ou trechos contiguos longos (>= 200 caracteres).
  - Marca os metadados `is_ementa_leak` e `motivo_leak` no JSONL de chunks.
  - Emite o arquivo estatistico consolidado `relatorio_auditoria_chunks.txt`, mapeando a taxa de contaminacao por posicao de chunk.

### Etapa 3: Mineracao de Negativos Dificeis (Hard Negatives)
- **Script**: `scripts/03_mine_negatives.py`
- **Descricao**:
  - Vetoriza todas as ementas e todos os chunks do split de treino utilizando o modelo base em precisao `bfloat16`.
  - Processa as consultas em lotes matriciais na GPU.
  - Aplica mascara em VRAM para impedir que chunks pertencentes ao mesmo processo judicial sejam selecionados como negativos (eliminando falsos negativos).
  - Filtra candidatos por faixa de similaridade semantica (ex.: entre 0.60 e 0.85) para evitar tanto negativos triviais quanto positivos com rotulo incorreto.
  - Ordena os candidatos pela proximidade semantica com o chunk positivo e seleciona os 5 mais desafiadores.
  - Grava o arquivo estruturado de treino `train_triplets.jsonl`.

### Etapa 4: Treinamento Contrastivo com Telemetria Cientifica
- **Script**: `scripts/04_train.py`
- **Descricao**:
  - Constroi a arquitetura do SentenceTransformer: Backbone Transformer Qwen3 + Last Token Pooling + Normalizacao L2 na esfera unitaria. Realiza Full Fine-Tuning.
  - Configura a `CachedMultipleNegativesRankingLoss`.
  - Instancia o `SentenceTransformerTrainer` com `batch_size = 96`, `bf16 = True`, `lr = 2.0e-5` com scheduler cosseno, dataloader multi-worker e `BatchSamplers.NO_DUPLICATES`.
  - Executa o callback cientifico `ScientificMetricsCallback`, registrando consumo de VRAM.
  - Ao final, persiste os pesos consolidados em `best_model_4b/` .

### Etapa 5: Avaliacao em Teste Cego e Comparativo
- **Script**: `scripts/05_evaluate.py`
- **Descricao**:
  - Carrega o modelo treinado em `best_model_4b/` e, opcionalmente, o modelo base (`--compare_baseline`).
  - Carrega `test_processos.jsonl` e `test_corpus_chunks.jsonl`.
  - Exclui integralmente chunks contaminados (`is_ementa_leak = True`) do corpus de avaliacao e da lista de relevantes.
  - Executa o InformationRetrievalEvaluator da SentenceTransformers calculando nDCG@10, MRR@10, MAP@100 e Accuracy/Recall@1, 5 e 10.
  - Produz relatorio tabular comparativo (`comparativo_test.txt`) proporcionado pelo treinamento contrastivo.

---

## Como Executar

### 1. Pre-requisitos de Ambiente

Recomenda-se Python 3.10+ e ambiente com PyTorch com suporte a CUDA:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install sentence-transformers transformers datasets pyyaml psutil
```

### 2. Configuracao dos Dados

Defina o caminho do seu arquivo bruto JSONL em `configs/config_base.yaml`:

```yaml
dados:
  caminho_arquivo_bruto: "data/raw/acordaos_tribunais.jsonl"
  diretorio_processados: "data/processed"
```

Caso os nomes das colunas do seu arquivo sejam diferentes, ajuste o mapeamento correspondente no mesmo arquivo YAML:

```yaml
mapeamento_colunas:
  tribunal: "str_tribunal"
  numero_processo: "str_numero_unico"
  ementa: "str_ementa"
  inteiro_teor: "str_inteiro_teor"
```

### 3. Execucao pelo Orquestrador Master

O modo mais simples de rodar todo o fluxo sequencialmente:

```bash
# Executa todas as 5 etapas de ponta a ponta
python scripts/run_pipeline.py --all

# Executa com sobrescrita de parametros de treino
python scripts/run_pipeline.py --all --batch_size 96 --epochs 1 --compare_baseline
```

Para rodar apenas etapas selecionadas:

```bash
# Executa apenas preparacao, split e chunking
python scripts/run_pipeline.py --steps split,chunk

# Executa apenas treinamento e avaliacao
python scripts/run_pipeline.py --steps train,eval --compare_baseline
```

### 4. Execucao Etapa por Etapa

Cada script pode ser executado de maneira totalmente independente:

```bash
# Etapa 1: Limpeza e Divisao em Train/Val/Test
python scripts/01_prepare_and_split.py

# Etapa 2: Geracao dos Chunks e Auditoria de Ementa
python scripts/02_generate_chunks.py

# Etapa 3: Mineracao de Hard Negatives em GPU
python scripts/03_mine_negatives.py

# Etapa 4: Treinamento Contrastivo
python scripts/04_train.py

# Etapa 5: Avaliacao no Conjunto de Teste
python scripts/05_evaluate.py --compare_baseline
```

---

## Metadados e Relatorios Cientificos Gerados

A cada execucao do pipeline sao produzidos relatorios rastreaveis para composicao de artigos cientificos e auditoria de modelos:

| Artefato Gerado | Localizacao | Conteudo |
| :--- | :--- | :--- |
| `relatorio_limpeza_splits.txt` | `data/processed/` | Estatisticas de filtros, descarte por tamanho e volume por tribunal. |
| `relatorio_auditoria_chunks.txt` | `data/processed/` | Ementa e distribuicao de vazamento por indice de chunk. |
| `metricas_telemetria.json` | `experiments/<exp>/` | Resumo de duracao total, pico de VRAM e consumo medio de RAM. |
| `telemetria_execucao.csv` | `experiments/<exp>/` | Metricas step a step. |
| `comparativo_test.txt` | `experiments/<exp>/avaliacoes/` | Tabela comparativa de metricas de IR (Baseline vs. BumbaLM Treinado). |
| `avaliacao_consolidada.json` | `experiments/<exp>/avaliacoes/` | Metricas globais e segmentadas por tribunal em formato estruturado. |
