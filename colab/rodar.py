#!/usr/bin/env python3
"""
Ablacao de negativos do TCC no Google Colab (1x A100 80 GB, LoRA). Faz no Colab o papel do node2/rodar.sh.

Rode a partir da raiz do repositorio (o notebook colab/BumbaLM_colab.ipynb ja faz isso):

    python colab/rodar.py preparar           # configs no Drive, dados no disco local, modelo baixado
    python colab/rodar.py checar             # GPU, versoes, dados, configuracao e andamento
    python colab/rodar.py testar_modelo      # compara as formas de carregar o modelo (~5 min)
    python colab/rodar.py calibrar           # mede s/passo e memoria; sugere N, eval_steps e save_steps
    python colab/rodar.py treinar nenhum     # uma variante (retoma do ultimo checkpoint, se houver)
    python colab/rodar.py treinar            # as tres, na ordem nenhum, dificeis, aleatorios
    python colab/rodar.py avaliar            # modelo base + variantes prontas no teste completo
    python colab/rodar.py status             # andamento das variantes e da avaliacao

Se a sessao cair, rode o mesmo comando de novo: o treino retoma do ultimo checkpoint completo no Drive,
as variantes concluidas sao puladas e a avaliacao reaproveita os modelos ja avaliados.

Pastas (troque a raiz do Drive com a variavel BUMBALM_DRIVE):
    MyDrive/BumbaLM_colab/dados/        processed_tjrs100k.tar.gz (do node2) e pip_freeze_node2.txt
    MyDrive/BumbaLM_colab/configs/      copia editavel de colab/configs (e esta que vale)
    MyDrive/BumbaLM_colab/experiments/  checkpoints, modelos e avaliacoes
    MyDrive/BumbaLM_colab/calibracao/   tabela de tempo e custo
    MyDrive/BumbaLM_colab/logs/         log de cada execucao
"""

from __future__ import annotations

import argparse
import collections
import difflib
import hashlib
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

REPO = Path(__file__).resolve().parent.parent
DRIVE = Path(os.environ.get("BUMBALM_DRIVE", "/content/drive/MyDrive/BumbaLM_colab"))
CONFIGS_REPO = REPO / "colab" / "configs"
CONFIGS = DRIVE / "configs"
DADOS_DRIVE = DRIVE / "dados"
LOGS = DRIVE / "logs"
CALIBRACAO = DRIVE / "calibracao"
CALIBRACAO_LOCAL = Path(os.environ.get("BUMBALM_CALIBRACAO_LOCAL", "/content/calibracao"))
ARQ_REVISAO = DRIVE / "modelo_revisao.txt"

NOMES_CONFIG = ("config_base.yaml", "config_data.yaml", "config_train.yaml")
ARQUIVOS_TREINO = ["train_triplets.jsonl", "train_tuplas_aleatorias.jsonl", "val_processos.jsonl", "val_corpus_chunks.jsonl"]
ARQUIVOS_TESTE = ["test_processos.jsonl", "test_corpus_chunks.jsonl"]
ORDEM_VARIANTES = ["nenhum", "dificeis", "aleatorios"]
PACOTES = ["torch", "transformers", "sentence-transformers", "peft", "accelerate", "datasets", "huggingface-hub"]

# Hiperparametros que so mudam memoria/velocidade ou o log. Podem variar entre as variantes.
CHAVES_LIVRES = {"mini_batch_size", "gradient_checkpointing", "logging_steps", "dataloader_num_workers",
                 "dataloader_pin_memory", "apagar_checkpoints_ao_fim", "save_total_limit"}
# VRAM reservada maxima aceita na calibracao (A100 de 80 GB, com folga para picos)
VRAM_MAXIMA_GB = 76.0
# Checkpoint a cada ~25 min, ~10 validacoes por variante
SEGUNDOS_ENTRE_CHECKPOINTS = 25 * 60
VALIDACOES_POR_VARIANTE = 10


# ----------------------------------------------------------------------------------------------
# Utilidades
# ----------------------------------------------------------------------------------------------
def ler_yaml(caminho: Path) -> dict[str, Any]:
    import yaml

    with caminho.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def gravar_yaml(dados: dict[str, Any], caminho: Path) -> None:
    import yaml

    caminho.parent.mkdir(parents=True, exist_ok=True)
    with caminho.open("w", encoding="utf-8") as f:
        yaml.safe_dump(dados, f, allow_unicode=True, sort_keys=False)


def caminho_config(nome: str) -> Path:
    """A copia do Drive vale; a do repositorio so e usada antes do 'preparar'."""
    no_drive = CONFIGS / nome
    return no_drive if no_drive.is_file() else CONFIGS_REPO / nome


def carregar_configs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    return tuple(ler_yaml(caminho_config(n)) for n in NOMES_CONFIG)  # type: ignore[return-value]


def args_configs() -> list[str]:
    return [
        "--config_base", str(caminho_config("config_base.yaml")),
        "--config_data", str(caminho_config("config_data.yaml")),
        "--config_train", str(caminho_config("config_train.yaml")),
    ]


def dir_dados(cfg_base: dict[str, Any]) -> Path:
    return Path(cfg_base["dados"]["diretorio_processados"])


def dir_experimento(cfg_base: dict[str, Any]) -> Path:
    saidas = cfg_base.get("saidas", {})
    return Path(saidas["diretorio_experimentos"]) / saidas["nome_experimento"]


def titulo(texto: str) -> None:
    print("\n" + "=" * 72 + f"\n{texto}\n" + "=" * 72, flush=True)


def ambiente_execucao() -> dict[str, str]:
    env = os.environ.copy()
    env.update({
        "PYTHONUNBUFFERED": "1",
        "TOKENIZERS_PARALLELISM": "false",
        # Menos fragmentacao com os lotes de tamanho variavel do GradCache
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        # O modelo ja esta numa pasta local: nada de consultar o Hub no meio do experimento
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    })
    env["PYTHONPATH"] = str(REPO) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return env


def executar(comando: list[str], nome_log: str) -> tuple[int, str]:
    """Roda o comando mostrando a saida e gravando em DRIVE/logs. Devolve (codigo, fim da saida)."""
    LOGS.mkdir(parents=True, exist_ok=True)
    caminho_log = LOGS / f"{nome_log}_{time.strftime('%Y%m%d_%H%M%S')}.log"
    print(f"$ {' '.join(comando)}\n(log: {caminho_log})", flush=True)
    fim: collections.deque[str] = collections.deque(maxlen=400)
    with caminho_log.open("w", encoding="utf-8") as log:
        processo = subprocess.Popen(
            comando, cwd=REPO, env=ambiente_execucao(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
        )
        assert processo.stdout is not None
        for linha in processo.stdout:
            sys.stdout.write(linha)
            sys.stdout.flush()
            log.write(linha)
            fim.append(linha)
        codigo = processo.wait()
    return codigo, "".join(fim)


def sha256_arquivo(caminho: Path) -> str:
    h = hashlib.sha256()
    with caminho.open("rb") as f:
        for bloco in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(bloco)
    return h.hexdigest()


def contar_linhas(caminho: Path) -> int:
    total = 0
    with caminho.open("rb") as f:
        for bloco in iter(lambda: f.read(16 * 1024 * 1024), b""):
            total += bloco.count(b"\n")
    return total


def versao_pacote(nome: str) -> str:
    try:
        from importlib.metadata import version

        return version(nome)
    except Exception:
        return "-"


# ----------------------------------------------------------------------------------------------
# preparar
# ----------------------------------------------------------------------------------------------
def preparar_configs() -> None:
    titulo("1. Configs no Drive")
    CONFIGS.mkdir(parents=True, exist_ok=True)
    for nome in NOMES_CONFIG:
        origem, destino = CONFIGS_REPO / nome, CONFIGS / nome
        if not destino.is_file():
            shutil.copy2(origem, destino)
            print(f"Copiado: {destino}")
            continue
        diferencas = list(difflib.unified_diff(
            origem.read_text(encoding="utf-8").splitlines(), destino.read_text(encoding="utf-8").splitlines(),
            fromfile=f"repositorio/{nome}", tofile=f"Drive/{nome}", lineterm="", n=0,
        ))
        if diferencas:
            print(f"Usando a copia do Drive de {nome} (difere do repositorio):")
            for linha in diferencas[:40]:
                print(f"    {linha}")
        else:
            print(f"OK: {destino} (igual ao repositorio)")


def _origem_dados(nome_pasta: str) -> Optional[Path]:
    candidatos = [DADOS_DRIVE / f"{nome_pasta}.tar.gz", DADOS_DRIVE / f"{nome_pasta}.tar", DADOS_DRIVE / nome_pasta]
    return next((c for c in candidatos if c.exists()), None)


def conferir_sha256(pasta: Path) -> tuple[int, list[str]]:
    """Confere os arquivos listados em SHA256SUMS (formato do sha256sum). Devolve (conferidos, falhas)."""
    lista = pasta / "SHA256SUMS"
    if not lista.is_file():
        return 0, []
    conferidos, falhas = 0, []
    for linha in lista.read_text(encoding="utf-8").splitlines():
        if not linha.strip():
            continue
        esperado, nome = linha.split(maxsplit=1)
        nome = nome.lstrip("*").strip()
        alvo = pasta / nome
        if not alvo.is_file():
            falhas.append(f"{nome} (ausente)")
            continue
        if sha256_arquivo(alvo) != esperado:
            falhas.append(f"{nome} (SHA-256 diferente)")
        conferidos += 1
    return conferidos, falhas


def preparar_dados(cfg_base: dict[str, Any]) -> None:
    titulo("2. Dados no disco local do Colab")
    destino = dir_dados(cfg_base)
    marcador = destino / ".pronto.json"
    if marcador.is_file():
        print(f"Dados ja preparados nesta sessao: {destino}")
        return

    origem = _origem_dados(destino.name)
    if origem is None:
        raise SystemExit(
            f"ERRO: nao achei os dados no Drive. Esperado: {DADOS_DRIVE / (destino.name + '.tar.gz')}\n"
            "Gere o pacote no node2 com: bash node2/exportar_para_colab.sh (veja colab/README.md)."
        )
    destino.parent.mkdir(parents=True, exist_ok=True)
    inicio = time.time()
    if origem.is_dir():
        print(f"Copiando {origem} -> {destino} ...", flush=True)
        shutil.copytree(origem, destino, dirs_exist_ok=True)
    else:
        print(f"Descompactando {origem} em {destino.parent} ...", flush=True)
        subprocess.run(["tar", "-xf", str(origem), "-C", str(destino.parent)], check=True)
    print(f"  pronto em {time.time() - inicio:.0f} s")

    faltando = [n for n in ARQUIVOS_TREINO + ARQUIVOS_TESTE if not (destino / n).is_file()]
    if faltando:
        raise SystemExit(f"ERRO: faltam arquivos em {destino}: {', '.join(faltando)}")

    print("Conferindo SHA-256 com o SHA256SUMS do node2 ...", flush=True)
    conferidos, falhas = conferir_sha256(destino)
    if falhas:
        raise SystemExit("ERRO: arquivos diferentes dos do node2:\n  " + "\n  ".join(falhas))
    if conferidos == 0:
        print("  AVISO: sem SHA256SUMS no pacote; nao deu para conferir com o node2.")
    else:
        print(f"  {conferidos} arquivos iguais aos do node2.")
    marcador.write_text(json.dumps({"origem": str(origem), "sha256_conferidos": conferidos,
                                    "quando": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2), encoding="utf-8")


def preparar_modelo(cfg_train: dict[str, Any]) -> None:
    titulo("3. Modelo base")
    modelo = cfg_train.get("modelo", {})
    pasta = Path(modelo["model_id"])
    repo = modelo.get("repo_hub", "Qwen/Qwen3-Embedding-4B")
    revisao = modelo.get("revisao_hub")
    # Sem revisao no config: a da primeira sessao (todas as sessoes usam a mesma) ou, na primeira,
    # a do node2 (modelo_revisao_node2.txt, gerado pelo node2/exportar_para_colab.sh).
    revisao_node2 = DADOS_DRIVE / "modelo_revisao_node2.txt"
    if not revisao and ARQ_REVISAO.is_file():
        revisao = ARQ_REVISAO.read_text(encoding="utf-8").strip() or None
    if not revisao and revisao_node2.is_file():
        revisao = revisao_node2.read_text(encoding="utf-8").strip() or None
        if revisao:
            print(f"Usando a revisao do node2: {revisao}")
    marcador = pasta / ".revisao"
    if marcador.is_file() and (pasta / "config.json").is_file():
        print(f"Modelo ja baixado nesta sessao: {pasta} (revisao {marcador.read_text(encoding='utf-8').strip()})")
        return

    from huggingface_hub import HfApi, snapshot_download

    sha = HfApi().model_info(repo, revision=revisao).sha
    print(f"Baixando {repo} na revisao {sha} para {pasta} ...", flush=True)
    inicio = time.time()
    snapshot_download(repo_id=repo, revision=sha, local_dir=str(pasta))
    marcador.write_text(sha, encoding="utf-8")
    print(f"  pronto em {time.time() - inicio:.0f} s")
    if ARQ_REVISAO.is_file():
        anterior = ARQ_REVISAO.read_text(encoding="utf-8").strip()
        if anterior != sha:
            print(f"  AVISO: sessoes anteriores usaram a revisao {anterior}; esta usa {sha}.")
    else:
        ARQ_REVISAO.parent.mkdir(parents=True, exist_ok=True)
        ARQ_REVISAO.write_text(sha, encoding="utf-8")
        print(f"  Revisao registrada em {ARQ_REVISAO}: as proximas sessoes baixam esta mesma.")


def modo_preparar(_args: argparse.Namespace) -> None:
    if not DRIVE.parent.is_dir():
        raise SystemExit(f"ERRO: {DRIVE.parent} nao existe. Monte o Drive antes (drive.mount('/content/drive')).")
    preparar_configs()
    cfg_base, _cfg_data, cfg_train = carregar_configs()
    preparar_dados(cfg_base)
    preparar_modelo(cfg_train)
    print("\nPreparacao concluida. Proximo passo: python colab/rodar.py checar")


# ----------------------------------------------------------------------------------------------
# checar e status
# ----------------------------------------------------------------------------------------------
def resumo_treino(cfg_train: dict[str, Any], n_tuplas: Optional[int] = None) -> dict[str, Any]:
    t = cfg_train.get("treinamento", {})
    lote = int(t.get("batch_size_train", 128)) * int(t.get("gradient_accumulation_steps", 1))
    limite = t.get("limite_tuplas")
    n = int(limite) if limite else n_tuplas
    return {
        "lora": cfg_train.get("lora", {}).get("ativo", False),
        "learning_rate": t.get("learning_rate"),
        "lote_por_passo": lote,
        "mini_batch_size": t.get("mini_batch_size"),
        "gradient_checkpointing": t.get("gradient_checkpointing"),
        "limite_tuplas": limite,
        "passos_por_variante": (n // lote) if n else None,
        "eval_steps": t.get("eval_steps"),
        "save_steps": t.get("save_steps"),
    }


def modo_checar(_args: argparse.Namespace) -> None:
    cfg_base, _cfg_data, cfg_train = carregar_configs()
    titulo("GPU e versoes")
    try:
        import torch

        if torch.cuda.is_available():
            prop = torch.cuda.get_device_properties(0)
            mem = prop.total_memory / 1024**3
            print(f"GPU: {prop.name} | {mem:.0f} GB | capacidade {prop.major}.{prop.minor} | bf16: {prop.major >= 8}")
            if "A100" not in prop.name or mem < 70:
                print("  AVISO: o perfil foi calibrado para a A100 de 80 GB (Ambiente > Alterar tipo > A100).")
        else:
            print("AVISO: sem GPU. Ambiente de execucao > Alterar o tipo > A100 GPU.")
    except ImportError:
        print("AVISO: torch nao instalado.")
    print(f"Python {sys.version.split()[0]}")
    freeze_node2 = DADOS_DRIVE / "pip_freeze_node2.txt"
    versoes_node2: dict[str, str] = {}
    if freeze_node2.is_file():
        for linha in freeze_node2.read_text(encoding="utf-8", errors="replace").splitlines():
            if "==" in linha:
                nome, ver = linha.split("==", 1)
                versoes_node2[nome.strip().lower().replace("_", "-")] = ver.strip()
    for pacote in PACOTES:
        local = versao_pacote(pacote)
        node2 = versoes_node2.get(pacote, "")
        aviso = "" if not node2 or local.split("+")[0] == node2.split("+")[0] else "   <- difere do node2"
        print(f"  {pacote:24s} {local:18s} node2: {node2 or '?'}{aviso}")

    titulo("Dados")
    pasta = dir_dados(cfg_base)
    for nome in ARQUIVOS_TREINO + ARQUIVOS_TESTE:
        arq = pasta / nome
        if arq.is_file():
            print(f"  {nome:34s} {arq.stat().st_size / 1024**2:10,.0f} MB {contar_linhas(arq):12,} linhas")
        else:
            print(f"  {nome:34s} AUSENTE (rode: python colab/rodar.py preparar)")
    marcador = pasta / ".pronto.json"
    if marcador.is_file():
        info = json.loads(marcador.read_text(encoding="utf-8"))
        print(f"  SHA-256 conferidos com o node2: {info.get('sha256_conferidos')}")
    revisao = (Path(cfg_train["modelo"]["model_id"]) / ".revisao")
    print(f"  Modelo: {cfg_train['modelo']['model_id']} | revisao {revisao.read_text().strip() if revisao.is_file() else 'nao baixado'}")

    titulo("Configuracao do treino (copia do Drive)")
    n_tuplas = contar_linhas(pasta / "train_triplets.jsonl") if (pasta / "train_triplets.jsonl").is_file() else None
    for chave, valor in resumo_treino(cfg_train, n_tuplas).items():
        print(f"  {chave:24s} {valor}")
    t = cfg_train.get("treinamento", {})
    if t.get("limite_tuplas") is None:
        print("  PENDENTE: limite_tuplas = null. Defina depois do 'calibrar' (ou treine com --todas_as_tuplas).")
    if int(t.get("eval_steps", 1)) % int(t.get("save_steps", 1)):
        print("  ERRO: eval_steps tem que ser multiplo de save_steps.")

    total, _usado, livre = shutil.disk_usage("/content" if Path("/content").is_dir() else REPO)
    print(f"\nDisco local livre: {livre / 1024**3:,.0f} GB de {total / 1024**3:,.0f} GB")
    modo_status(_args)


def _checkpoints(pasta: Path) -> list[Path]:
    def passo(p: Path) -> int:
        return int(p.name.rsplit("-", 1)[-1])

    return sorted((p for p in pasta.glob("checkpoint-*") if p.is_dir() and p.name.rsplit("-", 1)[-1].isdigit()), key=passo)


def checkpoint_completo(pasta: Path) -> bool:
    """Checkpoint gravado por inteiro no Drive (uma queda no meio da gravacao deixa arquivos faltando)."""
    tem_pesos = any((pasta / n).is_file() for n in ("adapter_model.safetensors", "model.safetensors", "pytorch_model.bin")) or \
        any(pasta.glob("model-*.safetensors"))
    exigidos = ["trainer_state.json", "optimizer.pt", "scheduler.pt"]
    return tem_pesos and all((pasta / n).is_file() and (pasta / n).stat().st_size > 0 for n in exigidos)


def _estado_checkpoint(pasta: Path) -> dict[str, Any]:
    estado = json.loads((pasta / "trainer_state.json").read_text(encoding="utf-8"))
    passo0 = next((r for r in estado.get("log_history", []) if r.get("step") == 0 and
                   any(k.endswith("mrr@10") for k in r)), None)
    mrr0 = next((v for k, v in (passo0 or {}).items() if k.endswith("mrr@10")), None)
    return {"passo": estado.get("global_step"), "total": estado.get("max_steps"),
            "melhor": estado.get("best_metric"), "melhor_ckpt": estado.get("best_model_checkpoint"), "mrr_passo0": mrr0}


def modo_status(_args: argparse.Namespace) -> None:
    cfg_base, _cfg_data, _cfg_train = carregar_configs()
    exp = dir_experimento(cfg_base)
    titulo(f"Andamento: {exp}")
    hashes: dict[str, Any] = {}
    for variante in ORDEM_VARIANTES:
        pasta = exp / variante
        registro = pasta / "melhor_checkpoint.json"
        if registro.is_file():
            r = json.loads(registro.read_text(encoding="utf-8"))
            hashes[variante] = r.get("hash_tuplas_sorteadas")
            print(f"  {variante:11s} CONCLUIDA | {r.get('n_tuplas_treino'):,} tuplas | {r.get('metrica_selecao')} "
                  f"= {r.get('melhor_metrica')} | modelo: {Path(str(r.get('modelo_para_avaliar'))).name}")
            continue
        ckpts = _checkpoints(pasta)
        completos = [c for c in ckpts if checkpoint_completo(c)]
        if completos:
            e = _estado_checkpoint(completos[-1])
            print(f"  {variante:11s} em andamento | passo {e['passo']}/{e['total']} | melhor {e['melhor']} | "
                  f"MRR@10 passo 0: {e['mrr_passo0']}")
        elif ckpts:
            print(f"  {variante:11s} so checkpoints incompletos ({', '.join(c.name for c in ckpts)})")
        else:
            print(f"  {variante:11s} nao iniciada")
    if len({h for h in hashes.values()}) > 1:
        print("  ERRO: as variantes treinaram com pares diferentes (hash_tuplas_sorteadas diferente)!")
    elif len(hashes) > 1:
        print(f"  Pares sorteados iguais nas variantes concluidas (hash {str(next(iter(hashes.values())))[:16]}...).")

    avaliacoes = exp / "avaliacoes"
    prontos = sorted(p.parent.name for p in avaliacoes.glob("test/*/metricas_test_*.json"))
    print(f"  Avaliacao no teste: {', '.join(prontos) if prontos else 'nenhum modelo ainda'}")
    comparativo = avaliacoes / "comparativo_test.txt"
    if comparativo.is_file():
        print("\n" + comparativo.read_text(encoding="utf-8"))


# ----------------------------------------------------------------------------------------------
# testar_modelo
# ----------------------------------------------------------------------------------------------
def modo_testar_modelo(_args: argparse.Namespace) -> None:
    cfg_base, _cfg_data, cfg_train = carregar_configs()
    codigo, _ = executar([
        sys.executable, "-u", "scripts/testar_carregamento.py",
        "--model_id", cfg_train["modelo"]["model_id"], "--dados", str(dir_dados(cfg_base)), "--processos", "100",
    ], "testar_modelo")
    if codigo:
        raise SystemExit(codigo)


# ----------------------------------------------------------------------------------------------
# calibrar
# ----------------------------------------------------------------------------------------------
def _tempo_por_passo(relatorio: Path, descartar: int = 2) -> tuple[Optional[float], Optional[float]]:
    """Mediana dos segundos entre passos de treino (sem os primeiros) e pico de VRAM reservada."""
    dados = json.loads(relatorio.read_text(encoding="utf-8"))
    passos = [(r["step"], r["tempo_segundos"]) for r in dados.get("historico_metricas", []) if "loss" in r]
    deltas = [(t2 - t1) / max(1, s2 - s1) for (s1, t1), (s2, t2) in zip(passos, passos[1:]) if s2 > s1]
    deltas = deltas[descartar:] or deltas
    return (statistics.median(deltas) if deltas else None), dados.get("vram_pico_reservada_gb")


def _rodada_calibracao(nome: str, variante: str, mini_batch: int, checkpointing: bool, passos: int,
                       com_validacao: bool, cfg_train: dict[str, Any]) -> dict[str, Any]:
    saida = CALIBRACAO_LOCAL / nome
    relatorio = saida / "relatorio_treinamento.json"
    resultado: dict[str, Any] = {"nome": nome, "variante": variante, "mini_batch_size": mini_batch,
                                 "gradient_checkpointing": checkpointing}
    salvo = CALIBRACAO / f"{nome}.json"
    if salvo.is_file():
        print(f"Calibracao '{nome}' ja feita: {salvo}")
        return json.loads(salvo.read_text(encoding="utf-8"))

    cfg = json.loads(json.dumps(cfg_train))
    cfg["treinamento"]["mini_batch_size"] = mini_batch
    cfg["treinamento"]["gradient_checkpointing"] = checkpointing
    cfg_tmp = CALIBRACAO_LOCAL / f"config_train_{nome}.yaml"
    gravar_yaml(cfg, cfg_tmp)

    lote = int(cfg["treinamento"].get("batch_size_train", 128))
    comando = [
        sys.executable, "-u", "scripts/04_train.py",
        "--config_base", str(caminho_config("config_base.yaml")),
        "--config_data", str(caminho_config("config_data.yaml")),
        "--config_train", str(cfg_tmp),
        "--negativos", variante, "--diagnostico", "--limite_tuplas", str(passos * lote),
        "--output_dir", str(saida), "--warmup_ratio", "0.1",
    ]
    comando += ["--eval_steps", "100000"] if com_validacao else ["--no_eval"]
    titulo(f"Calibracao '{nome}': {variante}, mini-lote {mini_batch}, gradient checkpointing {checkpointing}")
    inicio = time.time()
    codigo, fim = executar(comando, f"calibrar_{nome}")
    resultado["segundos_total"] = round(time.time() - inicio, 1)
    if codigo != 0 or not relatorio.is_file():
        resultado["status"] = "sem_memoria" if "OutOfMemory" in fim or "out of memory" in fim else f"erro_{codigo}"
    else:
        s_passo, vram = _tempo_por_passo(relatorio)
        resultado.update({"status": "ok", "segundos_por_passo": s_passo, "vram_pico_reservada_gb": vram})
        if vram is not None and vram > VRAM_MAXIMA_GB:
            resultado["status"] = "memoria_no_limite"
        if com_validacao:
            dados = json.loads(relatorio.read_text(encoding="utf-8"))
            passo0 = next((r for r in dados["historico_metricas"] if r.get("step") == 0 and "eval_runtime" in r), {})
            resultado["mrr10_passo0"] = next((v for k, v in passo0.items() if k.endswith("mrr@10")), None)
            resultado["ndcg10_passo0"] = next((v for k, v in passo0.items() if k.endswith("ndcg@10")), None)
            resultado["segundos_validacao"] = passo0.get("eval_runtime")
            achado = re.search(r"Avaliador 'val': ([\d,.]+) queries \| ([\d,.]+) chunks", fim)
            if achado:
                resultado["val_queries"] = int(re.sub(r"\D", "", achado.group(1)))
                resultado["val_chunks"] = int(re.sub(r"\D", "", achado.group(2)))
    CALIBRACAO.mkdir(parents=True, exist_ok=True)
    salvo.write_text(json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Resultado: {resultado}")
    return resultado


def modo_calibrar(args: argparse.Namespace) -> None:
    cfg_base, _cfg_data, cfg_train = carregar_configs()
    pasta = dir_dados(cfg_base)
    lote = int(cfg_train["treinamento"].get("batch_size_train", 128))
    opcoes = []
    for item in args.memoria.split(","):
        mb, gc = item.split(":")
        opcoes.append((int(mb), gc.strip().lower() in ("sim", "s", "1", "true")))

    # 1. Memoria x velocidade na variante mais pesada (7 textos por tupla). Aleatorios tem o mesmo formato.
    rodadas = [
        _rodada_calibracao(f"dificeis_mb{mb}_gc{int(gc)}", "dificeis", mb, gc, args.passos, False, cfg_train)
        for mb, gc in opcoes
    ]
    validas = [r for r in rodadas if r.get("status") == "ok" and r.get("segundos_por_passo")]
    if not validas:
        raise SystemExit("ERRO: nenhuma configuracao de memoria funcionou. Veja os logs em " + str(LOGS))
    melhor = min(validas, key=lambda r: r["segundos_por_passo"])
    print(f"\nConfiguracao escolhida: mini-lote {melhor['mini_batch_size']}, "
          f"gradient checkpointing {melhor['gradient_checkpointing']} ({melhor['segundos_por_passo']:.1f} s/passo)")

    # 2. Variante 'nenhum' (2 textos por tupla) com a validacao do passo 0: confere o modelo base.
    nenhum = _rodada_calibracao(
        f"nenhum_mb{melhor['mini_batch_size']}_gc{int(melhor['gradient_checkpointing'])}", "nenhum",
        melhor["mini_batch_size"], melhor["gradient_checkpointing"], args.passos, True, cfg_train,
    )
    if nenhum.get("status") != "ok":
        raise SystemExit(f"ERRO na calibracao da variante 'nenhum': {nenhum.get('status')}. Veja os logs.")

    # 3. Tabela de tempo e custo
    s_dif = melhor["segundos_por_passo"]
    s_nen = nenhum["segundos_por_passo"]
    s_val = nenhum.get("segundos_validacao") or 180.0
    textos_val = (nenhum.get("val_queries") or 300) + (nenhum.get("val_chunks") or 3400)
    # Teste completo: consultas + trechos limpos (9.999 + 92.101 na amostra de 100 mil)
    textos_teste = args.textos_teste
    s_teste_modelo = s_val * textos_teste / max(1, textos_val)
    disponiveis = min(contar_linhas(pasta / "train_triplets.jsonl"), contar_linhas(pasta / "train_tuplas_aleatorias.jsonl"))
    tamanhos = sorted({n for n in (20000, 30000, 50000, 75000, 100000) if n < disponiveis} | {disponiveis})

    linhas = []
    for n in tamanhos:
        passos = n // lote
        save = max(5, round(SEGUNDOS_ENTRE_CHECKPOINTS / s_dif))
        evals = max(1, round(passos / VALIDACOES_POR_VARIANTE / save)) * save
        n_val = passos // evals + 1
        h_treino = (passos * (2 * s_dif + s_nen) + 3 * n_val * s_val) / 3600
        h_teste = 4 * s_teste_modelo / 3600
        horas = h_treino + h_teste
        linhas.append({"n_tuplas": n, "passos": passos, "horas_dificeis": passos * s_dif / 3600,
                       "horas_nenhum": passos * s_nen / 3600, "horas_validacoes": 3 * n_val * s_val / 3600,
                       "horas_teste_4_modelos": h_teste, "horas_total": horas,
                       "unidades": horas * args.unidades_por_hora, "save_steps": save, "eval_steps": evals})

    texto = [
        "CALIBRACAO NA " + _nome_gpu(),
        f"Mini-lote {melhor['mini_batch_size']} | gradient checkpointing {melhor['gradient_checkpointing']}",
        f"Segundos por passo (lote de {lote}): dificeis/aleatorios {s_dif:.1f} | nenhum {s_nen:.1f}",
        f"Validacao (300 processos): {s_val:.0f} s | MRR@10 do modelo base no passo 0: {nenhum.get('mrr10_passo0')}",
        f"Teste completo por modelo (estimado): {s_teste_modelo / 60:.0f} min | tuplas disponiveis: {disponiveis:,}",
        f"Custo com {args.unidades_por_hora} unidades/hora (confira a taxa no painel Recursos do Colab)",
        "",
        f"{'N tuplas':>10} {'passos':>7} {'dificeis':>9} {'aleat.':>7} {'nenhum':>7} {'valid.':>7} {'teste':>6} "
        f"{'TOTAL h':>8} {'unidades':>9} {'save':>5} {'eval':>5}",
    ]
    for l in linhas:
        texto.append(
            f"{l['n_tuplas']:>10,} {l['passos']:>7,} {l['horas_dificeis']:>8.1f}h {l['horas_dificeis']:>6.1f}h "
            f"{l['horas_nenhum']:>6.1f}h {l['horas_validacoes']:>6.1f}h {l['horas_teste_4_modelos']:>5.1f}h "
            f"{l['horas_total']:>8.1f} {l['unidades']:>9,.0f} {l['save_steps']:>5} {l['eval_steps']:>5}"
        )
    texto += [
        "",
        "Depois de escolher N, edite a copia do Drive (configs/config_train.yaml):",
        f"  treinamento.mini_batch_size: {melhor['mini_batch_size']}",
        f"  treinamento.gradient_checkpointing: {str(melhor['gradient_checkpointing']).lower()}",
        "  treinamento.limite_tuplas: <N escolhido>",
        "  treinamento.save_steps e eval_steps: os da linha do N escolhido",
    ]
    CALIBRACAO.mkdir(parents=True, exist_ok=True)
    (CALIBRACAO / "calibracao.txt").write_text("\n".join(texto) + "\n", encoding="utf-8")
    (CALIBRACAO / "calibracao.json").write_text(json.dumps({
        "rodadas": rodadas + [nenhum], "escolhida": melhor, "tabela": linhas,
        "unidades_por_hora": args.unidades_por_hora}, indent=2, ensure_ascii=False), encoding="utf-8")
    titulo("Tabela de tempo e custo")
    print("\n".join(texto))
    mrr0 = nenhum.get("mrr10_passo0")
    if mrr0 is not None and mrr0 < 0.3:
        print(f"\nATENCAO: MRR@10 do modelo base no passo 0 = {mrr0:.3f}. Esperado ~0,5. Rode 'testar_modelo' antes de treinar.")


def _nome_gpu() -> str:
    try:
        import torch

        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    except Exception:
        return "?"


# ----------------------------------------------------------------------------------------------
# treinar
# ----------------------------------------------------------------------------------------------
def configuracao_congelada(cfg_train: dict[str, Any], cfg_data: dict[str, Any]) -> dict[str, Any]:
    """O que tem que ser igual nas tres variantes para a ablacao valer."""
    modelo = dict(cfg_train.get("modelo", {}))
    modelo.pop("model_id", None)
    modelo["revisao_baixada"] = ARQ_REVISAO.read_text(encoding="utf-8").strip() if ARQ_REVISAO.is_file() else None
    return {
        "modelo": modelo,
        "lora": cfg_train.get("lora", {}),
        "treinamento": {k: v for k, v in cfg_train.get("treinamento", {}).items() if k not in CHAVES_LIVRES},
        "loss": cfg_train.get("loss", {}),
        "validacao": {"max_processos_val": cfg_train.get("avaliacao", {}).get("max_processos_val")},
        "n_negativos": cfg_data.get("mineracao_negativos", {}).get("n_negativos"),
    }


def _diferencas(a: Any, b: Any, prefixo: str = "") -> list[str]:
    if isinstance(a, dict) and isinstance(b, dict):
        saida = []
        for chave in sorted(set(a) | set(b)):
            saida += _diferencas(a.get(chave), b.get(chave), f"{prefixo}{chave}.")
        return saida
    return [] if a == b else [f"{prefixo.rstrip('.')}: {a!r} -> {b!r}"]


def conferir_configuracao(exp: Path, cfg_train: dict[str, Any], cfg_data: dict[str, Any], forcar: bool) -> None:
    atual = configuracao_congelada(cfg_train, cfg_data)
    arquivo = exp / "configuracao_congelada.json"
    if not arquivo.is_file():
        exp.mkdir(parents=True, exist_ok=True)
        arquivo.write_text(json.dumps(atual, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Configuracao do experimento registrada em {arquivo}")
        return
    anterior = json.loads(arquivo.read_text(encoding="utf-8"))
    diferencas = _diferencas(anterior, atual)
    if not diferencas:
        print("Configuracao igual a das variantes anteriores.")
        return
    print("A configuracao mudou desde a primeira variante:\n  " + "\n  ".join(diferencas))
    if not forcar:
        raise SystemExit(
            "ERRO: as variantes tem que treinar com os mesmos hiperparametros. Volte a config do Drive ao que "
            f"esta em {arquivo} ou, se a mudanca for intencional (e nenhuma variante foi concluida), use --forcar."
        )
    arquivo.write_text(json.dumps(atual, indent=2, ensure_ascii=False), encoding="utf-8")


def ultimo_checkpoint_completo(pasta: Path) -> Optional[Path]:
    """Marca como incompletos (renomeando) os checkpoints que a queda deixou pela metade."""
    completo = None
    for ckpt in _checkpoints(pasta):
        if checkpoint_completo(ckpt):
            completo = ckpt
        else:
            destino = ckpt.with_name(f"incompleto-{ckpt.name}-{int(time.time())}")
            ckpt.rename(destino)
            print(f"Checkpoint incompleto (queda durante a gravacao) separado: {destino.name}")
    return completo


def modo_treinar(args: argparse.Namespace) -> None:
    cfg_base, cfg_data, cfg_train = carregar_configs()
    t = cfg_train.get("treinamento", {})
    if t.get("limite_tuplas") is None and not args.todas_as_tuplas:
        raise SystemExit(
            "ERRO: limite_tuplas ainda e null na config do Drive. Rode 'python colab/rodar.py calibrar', escolha N "
            "e edite configs/config_train.yaml no Drive. Para usar todas as tuplas, passe --todas_as_tuplas."
        )
    if int(t.get("eval_steps", 1)) % int(t.get("save_steps", 1)):
        raise SystemExit("ERRO: eval_steps tem que ser multiplo de save_steps (o melhor checkpoint precisa existir).")
    if not (dir_dados(cfg_base) / ".pronto.json").is_file():
        raise SystemExit("ERRO: dados nao preparados nesta sessao. Rode: python colab/rodar.py preparar")

    exp = dir_experimento(cfg_base)
    conferir_configuracao(exp, cfg_train, cfg_data, args.forcar)
    variantes = args.variantes or ORDEM_VARIANTES
    for variante in variantes:
        if variante not in ORDEM_VARIANTES:
            raise SystemExit(f"ERRO: variante desconhecida '{variante}'. Use: {', '.join(ORDEM_VARIANTES)}")
        pasta = exp / variante
        if (pasta / "melhor_checkpoint.json").is_file():
            print(f"Variante '{variante}' ja concluida ({pasta / 'melhor_checkpoint.json'}). Pulando.")
            continue
        comando = [sys.executable, "-u", "scripts/04_train.py", *args_configs(), "--negativos", variante]
        ckpt = ultimo_checkpoint_completo(pasta) if pasta.is_dir() else None
        if ckpt is not None:
            print(f"Retomando '{variante}' de {ckpt.name}")
            comando += ["--resume_from_checkpoint", str(ckpt)]
        titulo(f"Treino da variante '{variante}'")
        codigo, _ = executar(comando, f"treinar_{variante}")
        if codigo != 0:
            raise SystemExit(f"ERRO: o treino de '{variante}' terminou com codigo {codigo}. Veja o log em {LOGS}.")
    modo_status(args)


# ----------------------------------------------------------------------------------------------
# avaliar
# ----------------------------------------------------------------------------------------------
def modo_avaliar(args: argparse.Namespace) -> None:
    cfg_base, _cfg_data, _cfg_train = carregar_configs()
    exp = dir_experimento(cfg_base)
    concluidas = [v for v in ORDEM_VARIANTES if (exp / v / "melhor_checkpoint.json").is_file()]
    faltando = [v for v in ORDEM_VARIANTES if v not in concluidas]
    hashes = {json.loads((exp / v / "melhor_checkpoint.json").read_text(encoding="utf-8")).get("hash_tuplas_sorteadas")
              for v in concluidas}
    if len(hashes) > 1:
        raise SystemExit("ERRO: as variantes concluidas treinaram com pares diferentes. Nao da para comparar.")
    if faltando:
        print(f"Aviso: variantes ainda nao concluidas ({', '.join(faltando)}). Avaliando o que ja existe; "
              "rode 'avaliar' de novo depois (os modelos ja avaliados sao reaproveitados).")
    comando = [
        sys.executable, "-u", "scripts/05_evaluate.py",
        "--config_base", str(caminho_config("config_base.yaml")),
        "--config_train", str(caminho_config("config_train.yaml")),
        "--split", "test", "--pular_avaliados",
    ]
    if not args.sem_base:
        comando.append("--compare_baseline")
    codigo, _ = executar(comando, "avaliar")
    if codigo != 0:
        raise SystemExit(f"ERRO: a avaliacao terminou com codigo {codigo}. Rode de novo para retomar.")


# ----------------------------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="modo", required=True)
    sub.add_parser("preparar", help="configs no Drive, dados no disco local e modelo baixado")
    sub.add_parser("checar", help="GPU, versoes, dados, configuracao e andamento")
    sub.add_parser("status", help="andamento das variantes e da avaliacao")
    sub.add_parser("testar_modelo", help="compara as formas de carregar o modelo (~5 min)")

    p_cal = sub.add_parser("calibrar", help="mede s/passo e memoria e monta a tabela de custo")
    p_cal.add_argument("--passos", type=int, default=10, help="passos de treino por rodada (padrao: 10)")
    p_cal.add_argument("--memoria", default="16:nao,8:nao,16:sim",
                       help="opcoes mini_lote:gradient_checkpointing a testar (padrao: 16:nao,8:nao,16:sim)")
    p_cal.add_argument("--unidades_por_hora", type=float, default=8.88,
                       help="taxa da A100 no painel Recursos do Colab (padrao: 8.88)")
    p_cal.add_argument("--textos_teste", type=int, default=9999 + 92101,
                       help="consultas + trechos limpos do teste (padrao: 9.999 + 92.101)")

    p_tr = sub.add_parser("treinar", help="treina as variantes (retoma do ultimo checkpoint)")
    p_tr.add_argument("variantes", nargs="*", help="nenhum, dificeis, aleatorios (padrao: as tres, nessa ordem)")
    p_tr.add_argument("--todas_as_tuplas", action="store_true", help="aceita limite_tuplas: null (todas as tuplas)")
    p_tr.add_argument("--forcar", action="store_true", help="aceita mudanca de hiperparametros entre variantes")

    p_av = sub.add_parser("avaliar", help="modelo base + variantes no teste completo")
    p_av.add_argument("--sem_base", action="store_true", help="nao avalia o modelo base")

    args = parser.parse_args()
    {
        "preparar": modo_preparar, "checar": modo_checar, "status": modo_status,
        "testar_modelo": modo_testar_modelo, "calibrar": modo_calibrar,
        "treinar": modo_treinar, "avaliar": modo_avaliar,
    }[args.modo](args)


if __name__ == "__main__":
    main()
