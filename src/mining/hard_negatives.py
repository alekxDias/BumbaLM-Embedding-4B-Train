"""
Mineracao vetorial acelerada de Hard Negatives em GPU para aprendizado contrastivo.
"""

from __future__ import annotations

import gc
import hashlib
import inspect
import json
import os
import re
import shutil
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.data.schema import ChunkProcesso, ProcessoJudicial
from src.hardware import dtype_para_dispositivo


DEFAULT_QUERY_INSTRUCTION = (
    "Instruct: Dado um trecho de um documento jurídico brasileiro, "
    "recupere o trecho mais relevante semanticamente.\nQuery: "
)

NOME_ARQUIVO_EMBEDDINGS = "corpus_embeddings_f16.npy"
_BLOCO_GRAVACAO_EMBEDDINGS = 100_000
# Valor das similaridades mascaradas. Cabe em float16 (maximo 65504): -1e9 so funcionava em
# bfloat16 e quebrava na V100 ("value cannot be converted to type at::Half without overflow").
# Cosseno nunca passa de -1, entao qualquer score abaixo de -2 e mascarado.
_VALOR_MASCARA = -1e4
_LIMITE_MASCARADO = -2.0
# Le so o id_global na retomada, sem montar o JSON inteiro da linha.
_RE_ID_GLOBAL = re.compile(rb'"id_global"\s*:\s*"([^"\\]+)"')


def preparar_query_com_instrucao(query: str, instrucao: str = DEFAULT_QUERY_INSTRUCTION) -> str:
    """
    Aplica a instrucao formal na ementa (query).
    Os documentos candidatos (positivos e negativos) nao devem receber o prefixo.
    """
    texto = str(query or "").strip()
    if texto.startswith("Instruct:"):
        return texto
    return f"{instrucao}{texto}"


def carregar_modelo_mineracao(
    model_id: str = "Qwen/Qwen3-Embedding-4B",
    max_seq_len: int = 768,
    device: Optional[str] = None,
    attn_implementation: str = "sdpa",
) -> SentenceTransformer:
    """
    Inicializa o modelo SentenceTransformer para inferencia vetorial e calculo de similaridades.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # V100 nao tem bf16 nativo: usa fp16. A100/H200 seguem em bf16.
    dtype_modelo = dtype_para_dispositivo(device)

    print(f"Carregando modelo de mineracao: {model_id} no dispositivo: {device} (dtype: {dtype_modelo})...")
    modelo = SentenceTransformer(
        model_id,
        model_kwargs={
            "torch_dtype": dtype_modelo,
            "trust_remote_code": True,
            "attn_implementation": attn_implementation if device == "cuda" else "eager",
        },
        tokenizer_kwargs={"padding_side": "left"},
        device=device,
    )
    modelo.max_seq_length = max_seq_len
    return modelo


def _encode_aceita_duas_gpus(modelo: SentenceTransformer) -> bool:
    """A API nova de encode aceita device=["cuda:0", "cuda:1"]. A antiga usa o pool explicito."""
    # pool e chunk_size so existem no encode que reparte o corpus entre as duas placas.
    parametros = inspect.signature(modelo.encode).parameters
    return "pool" in parametros or "chunk_size" in parametros


def _encodar_uma_gpu(
    modelo: SentenceTransformer,
    textos: list[str],
    batch_size: int,
    show_progress: bool,
) -> np.ndarray:
    return modelo.encode(
        textos,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )


def _iniciar_pool_gpus(modelo: SentenceTransformer, dispositivos: list[str]):
    # Versoes antigas nomeiam o argumento target_devices; as novas recebem a lista direto.
    assinatura = inspect.signature(modelo.start_multi_process_pool)
    if "target_devices" in assinatura.parameters:
        return modelo.start_multi_process_pool(target_devices=dispositivos)
    return modelo.start_multi_process_pool(dispositivos)


def _encodar_nas_duas_gpus(
    modelo: SentenceTransformer,
    textos: list[str],
    batch_size: int,
    show_progress: bool,
    dispositivos: list[str],
) -> np.ndarray:
    """
    Um processo por GPU. A biblioteca devolve as linhas na ordem original dos textos.
    O pool e fechado antes de voltar, para liberar a copia extra do modelo.
    """
    print(f"Vetorizando o corpus em {len(dispositivos)} GPUs: {', '.join(dispositivos)}")
    # O pool do Sentence Transformers move o modelo para a CPU e o deixa la. Sem devolver,
    # as ementas da busca eram vetorizadas na CPU: 7 s por processo com as GPUs a 0%.
    dispositivo_original = modelo.device
    try:
        return _encodar_com_pool(modelo, textos, batch_size, show_progress, dispositivos)
    finally:
        modelo.to(dispositivo_original)


def _encodar_com_pool(
    modelo: SentenceTransformer,
    textos: list[str],
    batch_size: int,
    show_progress: bool,
    dispositivos: list[str],
) -> np.ndarray:
    # Nao e DDP: cada processo vetoriza um pedaco e a biblioteca junta na ordem original.
    if _encode_aceita_duas_gpus(modelo):
        embeddings = modelo.encode(
            textos,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,
            convert_to_numpy=True,
            device=dispositivos,
        )
        return _como_matriz_numpy(embeddings)

    if not hasattr(modelo, "start_multi_process_pool") or not hasattr(modelo, "encode_multi_process"):
        print("Aviso: esta versao do Sentence Transformers nao vetoriza nas duas GPUs. Usando uma placa.")
        return _encodar_uma_gpu(modelo, textos, batch_size, show_progress)

    pool = None
    try:
        pool = _iniciar_pool_gpus(modelo, dispositivos)
        aceitos = inspect.signature(modelo.encode_multi_process).parameters
        kwargs = {"batch_size": batch_size, "normalize_embeddings": True}
        if "show_progress_bar" in aceitos:
            kwargs["show_progress_bar"] = show_progress
        embeddings = modelo.encode_multi_process(textos, pool, **kwargs)
        return _como_matriz_numpy(embeddings)
    finally:
        # Fecha os processos antes da busca. Sem isso a placa 0 ficaria com dois modelos e a matriz.
        if pool is not None:
            modelo.stop_multi_process_pool(pool)


def _como_matriz_numpy(embeddings) -> np.ndarray:
    # O encode multi-GPU pode devolver tensor, lista de blocos ou ndarray. A gravacao espera ndarray.
    if isinstance(embeddings, torch.Tensor):
        return embeddings.detach().float().cpu().numpy()
    if isinstance(embeddings, list):
        return np.concatenate([_como_matriz_numpy(parte) for parte in embeddings], axis=0)
    return np.asarray(embeddings)


def encodar_textos_corpus(
    modelo: SentenceTransformer,
    textos: list[str],
    batch_size: int = 256,
    show_progress: bool = True,
    dispositivos: list[str] | None = None,
) -> np.ndarray:
    """
    Gera representacoes vetoriais normalizadas para todos os textos do corpus.
    Com as duas GPUs, reparte os textos entre dois processos da biblioteca.
    """
    if dispositivos and len(dispositivos) >= 2:
        return _encodar_nas_duas_gpus(modelo, textos, batch_size, show_progress, dispositivos)
    return _encodar_uma_gpu(modelo, textos, batch_size, show_progress)


def _caminho_sidecar_embeddings(caminho: Path) -> Path:
    return caminho.with_suffix(".json")


def assinatura_corpus(ids_chunks: list[str]) -> str:
    """SHA-1 dos chunk_ids em ordem. Muda se o corpus mudar, mesmo com o mesmo numero de chunks."""
    sha = hashlib.sha1()
    for cid in ids_chunks:
        sha.update(cid.encode("utf-8"))
        sha.update(b"\n")
    return sha.hexdigest()


def _embeddings_compativeis(caminho: Path, n_chunks: int, model_id: str, assinatura: str | None = None) -> bool:
    """Confere se o .npy e o sidecar descrevem este corpus e este modelo."""
    sidecar = _caminho_sidecar_embeddings(caminho)
    if not caminho.is_file() or not sidecar.is_file():
        return False
    try:
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
        if int(meta.get("n_chunks", -1)) != n_chunks or meta.get("model_id") != model_id:
            return False
        # Cache antigo, sem assinatura, continua valendo; com assinatura diferente, nao.
        if assinatura and meta.get("assinatura_corpus") not in (None, assinatura):
            return False
        if meta.get("dtype") != "float16":
            return False
        arr = np.load(caminho, mmap_mode="r")
        dim = int(meta.get("dim", -1))
        compativel = (
            arr.ndim == 2
            and arr.shape[0] == n_chunks
            and arr.dtype == np.float16
            and (dim < 1 or arr.shape[1] == dim)
        )
        del arr
        return compativel
    except Exception as exc:
        print(f"Aviso: embeddings em {caminho} ignorados ({exc}).")
        return False


def salvar_embeddings_float16(
    caminho: Path | str,
    embeddings: np.ndarray,
    model_id: str,
    bloco: int = _BLOCO_GRAVACAO_EMBEDDINGS,
    assinatura: str | None = None,
) -> None:
    """
    Grava a matriz em float16 por blocos.

    O arquivo final so substitui o anterior depois da escrita completa.
    O sidecar JSON e gravado em seguida, para um .npy pela metade nao ser reutilizado.
    """
    destino = Path(caminho)
    if embeddings.ndim != 2:
        raise ValueError(f"Matriz de embeddings deve ser 2D, recebeu shape {embeddings.shape}.")

    destino.parent.mkdir(parents=True, exist_ok=True)
    temporario = destino.with_name(destino.name + ".tmp")
    if temporario.exists():
        temporario.unlink()

    mapa = np.lib.format.open_memmap(
        os.fspath(temporario),
        mode="w+",
        dtype=np.float16,
        shape=embeddings.shape,
    )
    n_linhas = embeddings.shape[0]
    for inicio in tqdm(range(0, n_linhas, bloco), desc="Salvando embeddings float16"):
        fim = min(inicio + bloco, n_linhas)
        mapa[inicio:fim] = embeddings[inicio:fim]
    mapa.flush()
    mmap = getattr(mapa, "_mmap", None)
    if mmap is not None:
        mmap.close()
    del mapa
    gc.collect()
    os.replace(temporario, destino)

    meta = {
        "model_id": model_id,
        "n_chunks": int(n_linhas),
        "dim": int(embeddings.shape[1]),
        "dtype": "float16",
        "normalizado": True,
        "assinatura_corpus": assinatura,
    }
    sidecar = _caminho_sidecar_embeddings(destino)
    sidecar_tmp = sidecar.with_name(sidecar.name + ".tmp")
    sidecar_tmp.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(sidecar_tmp, sidecar)
    print(f"Embeddings salvos: {destino} ({n_linhas:,} x {embeddings.shape[1]}, float16)")


def carregar_embeddings_float16(caminho: Path | str) -> np.ndarray:
    """Carrega a matriz em mmap."""
    return np.load(caminho, mmap_mode="c")


def obter_embeddings_corpus(
    modelo: SentenceTransformer,
    textos: list[str],
    caminho: Path | str,
    model_id: str,
    encode_batch_size: int,
    recomputar: bool,
    dispositivos: list[str] | None = None,
    ids_chunks: list[str] | None = None,
    bloco_vetorizacao: int = 20_000,
) -> np.ndarray:
    """
    Reutiliza o .npy quando ele bate com o corpus. Senao, vetoriza e grava em float16.

    A vetorizacao e feita em blocos de `bloco_vetorizacao` chunks, cada um gravado em
    <nome>_partes/ assim que termina. Se o job cair (walltime, falha), a proxima execucao
    pula os blocos prontos. No fim os blocos viram um .npy so e a pasta e apagada.
    """
    destino = Path(caminho)
    n_chunks = len(textos)
    assinatura = assinatura_corpus(ids_chunks) if ids_chunks is not None else None

    if not recomputar and _embeddings_compativeis(destino, n_chunks, model_id, assinatura):
        print(f"Reutilizando embeddings: {destino}")
        print("Vetorizacao do corpus ignorada.")
        return carregar_embeddings_float16(destino)

    if recomputar:
        print("Recalculo de embeddings solicitado.")
    elif destino.is_file():
        print("Embeddings existentes nao batem com este corpus ou modelo. Vetorizando de novo...")
    else:
        print("Arquivo de embeddings ausente. Vetorizando o corpus...")

    pasta_partes = destino.with_name(destino.stem + "_partes")
    if recomputar and pasta_partes.is_dir():
        shutil.rmtree(pasta_partes)
    pasta_partes.mkdir(parents=True, exist_ok=True)

    bloco = max(1, int(bloco_vetorizacao))
    n_blocos = (n_chunks + bloco - 1) // bloco
    meta_base = {"model_id": model_id, "n_chunks_total": n_chunks, "assinatura_corpus": assinatura, "bloco": bloco}
    partes: list[Path] = []
    for i in range(n_blocos):
        ini, fim = i * bloco, min((i + 1) * bloco, n_chunks)
        parte = pasta_partes / f"parte_{i:05d}.npy"
        meta_parte = parte.with_suffix(".json")
        esperado = {**meta_base, "inicio": ini, "fim": fim}
        partes.append(parte)
        if parte.is_file() and meta_parte.is_file():
            try:
                if json.loads(meta_parte.read_text(encoding="utf-8")) == esperado:
                    print(f"Bloco {i + 1}/{n_blocos} (chunks {ini:,}-{fim - 1:,}) ja vetorizado. Pulando.")
                    continue
            except Exception:
                pass
        print(f"Vetorizando bloco {i + 1}/{n_blocos}: chunks {ini:,}-{fim - 1:,} ...")
        embeddings = encodar_textos_corpus(
            modelo=modelo,
            textos=textos[ini:fim],
            batch_size=encode_batch_size,
            show_progress=True,
            dispositivos=dispositivos,
        )
        # .npy primeiro, metadado depois: bloco sem metadado e refeito na retomada
        tmp = parte.with_name(parte.stem + ".tmp.npy")
        np.save(tmp, np.asarray(embeddings, dtype=np.float16))
        os.replace(tmp, parte)
        meta_parte.write_text(json.dumps(esperado), encoding="utf-8")
        del embeddings
        gc.collect()

    # Junta os blocos no .npy final, na ordem do corpus
    primeira = np.load(partes[0], mmap_mode="r")
    dim = int(primeira.shape[1])
    del primeira
    temporario = destino.with_name(destino.name + ".tmp")
    mapa = np.lib.format.open_memmap(os.fspath(temporario), mode="w+", dtype=np.float16, shape=(n_chunks, dim))
    for i, parte in enumerate(tqdm(partes, desc="Juntando blocos de embeddings")):
        dados = np.load(parte, mmap_mode="r")
        mapa[i * bloco : i * bloco + dados.shape[0]] = dados
        del dados
    mapa.flush()
    mmap = getattr(mapa, "_mmap", None)
    if mmap is not None:
        mmap.close()
    del mapa
    gc.collect()
    os.replace(temporario, destino)
    meta = {
        "model_id": model_id,
        "n_chunks": n_chunks,
        "dim": dim,
        "dtype": "float16",
        "normalizado": True,
        "assinatura_corpus": assinatura,
    }
    _caminho_sidecar_embeddings(destino).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.rmtree(pasta_partes, ignore_errors=True)
    print(f"Embeddings salvos: {destino} ({n_chunks:,} x {dim}, float16, {n_blocos} blocos)")
    return carregar_embeddings_float16(destino)


def _enviar_fatia_para_gpu(fatia, dispositivo: str, dtype_tensor: torch.dtype) -> torch.Tensor:
    # torch.from_numpy exige um array.
    arr = np.ascontiguousarray(fatia)
    if not arr.flags.writeable:
        arr = np.array(arr, copy=True)
    return torch.from_numpy(arr).to(device=dispositivo, dtype=dtype_tensor)


def _montar_shards(
    corpus_embs_cpu: np.ndarray,
    dispositivos: list[str],
    dtype_tensor: torch.dtype,
) -> list[tuple[int, torch.Tensor]]:
    """Cada GPU fica com uma fatia contigua da matriz, na ordem original das linhas."""
    n_linhas = corpus_embs_cpu.shape[0]
    shards: list[tuple[int, torch.Tensor]] = []
    for indice, dispositivo in enumerate(dispositivos):
        # Corte inteiro: a ultima placa absorve o resto se a divisao nao for exata.
        inicio = (n_linhas * indice) // len(dispositivos)
        fim = (n_linhas * (indice + 1)) // len(dispositivos)
        if inicio == fim:
            continue
        print(f"  Fatia {indice}: linhas {inicio:,}-{fim - 1:,} em {dispositivo}")
        shards.append((inicio, _enviar_fatia_para_gpu(corpus_embs_cpu[inicio:fim], dispositivo, dtype_tensor)))
    return shards


def _mascarar_mesmo_processo(
    sims: torch.Tensor,
    offset: int,
    n_local: int,
    batch_procs: list[ProcessoJudicial],
    indices_por_processo: dict[str, list[int]],
    indices_vazados: Optional[np.ndarray] = None,
) -> None:
    """
    indices_por_processo: chunks que nao podem ser negativos de cada processo
    (os dele e os dos processos quase duplicados).
    indices_vazados: chunks com a ementa copiada, fora de todas as consultas. No teste esses
    chunks nao existem; como negativos, ensinariam o modelo a afastar "texto com cara de ementa".
    """
    # O indice global vira indice local da fatia. Chunk de outro processo, ou de outra placa, fica de fora.
    for i_local, proc in enumerate(batch_procs):
        locais = [
            idx - offset
            for idx in indices_por_processo[proc.id_global]
            if offset <= idx < offset + n_local
        ]
        if locais:
            sims[i_local, locais] = _VALOR_MASCARA
    if indices_vazados is not None and len(indices_vazados):
        na_fatia = indices_vazados[(indices_vazados >= offset) & (indices_vazados < offset + n_local)] - offset
        if len(na_fatia):
            sims[:, torch.as_tensor(na_fatia, device=sims.device, dtype=torch.long)] = _VALOR_MASCARA


def _topk_numa_fatia(
    q_embs: torch.Tensor,
    matriz: torch.Tensor,
    offset: int,
    batch_procs: list[ProcessoJudicial],
    indices_por_processo: dict[str, list[int]],
    k: int,
    dtype_tensor: torch.dtype,
    indices_vazados: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    # As ementas nascem numa placa so. A copia para a fatia e pequena: 256 vetores.
    q_local = q_embs.to(device=matriz.device, dtype=dtype_tensor)
    sims = torch.matmul(q_local, matriz.T)
    _mascarar_mesmo_processo(sims, offset, matriz.shape[0], batch_procs, indices_por_processo, indices_vazados)
    k_local = min(k, matriz.shape[0])
    scores, indices = torch.topk(sims, k=k_local, dim=-1)
    # Soma o deslocamento da fatia para o indice voltar a ser o da matriz completa.
    return scores.float().cpu().numpy(), indices.cpu().numpy() + offset


def _juntar_topk(
    partes_scores: list[np.ndarray],
    partes_indices: list[np.ndarray],
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    O top-k global esta dentro do top-k de cada metade.
    Juntar os topos e cortar de novo em k devolve o mesmo conjunto de uma matriz so.
    """
    scores = np.concatenate(partes_scores, axis=1)
    indices = np.concatenate(partes_indices, axis=1)
    k_final = min(k, scores.shape[1])
    # O sinal negativo coloca os mais similares primeiro. O mascarado (_VALOR_MASCARA) vai para o fim.
    ordem = np.argsort(-scores, axis=1)[:, :k_final]
    return np.take_along_axis(scores, ordem, axis=1), np.take_along_axis(indices, ordem, axis=1)


def _caminho_sidecar_progresso(saida: Path) -> Path:
    # train_triplets.jsonl -> train_triplets.progress.json
    return saida.with_suffix(".progress.json")


def _progresso_vazio(n_chunks_corpus: int) -> dict[str, Any]:
    return {
        "n_processos_concluidos": 0,
        "ultimo_id_global": "",
        "tripletos_salvos": 0,
        "byte_offset": 0,
        "n_chunks_corpus": int(n_chunks_corpus),
    }


def _gravar_progresso(caminho: Path, progresso: dict[str, Any]) -> None:
    # tmp + replace: um sidecar pela metade nao e lido na proxima execucao.
    temporario = caminho.with_name(caminho.name + ".tmp")
    temporario.write_text(json.dumps(progresso, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporario, caminho)


def _ler_progresso(caminho: Path) -> dict[str, Any] | None:
    if not caminho.is_file():
        return None
    try:
        meta = json.loads(caminho.read_text(encoding="utf-8"))
    except Exception:
        return None
    return meta if isinstance(meta, dict) else None


def _progresso_compativel(
    progresso: dict[str, Any] | None,
    saida: Path,
    n_chunks_corpus: int,
    processos_validos: list[ProcessoJudicial],
) -> dict[str, Any] | None:
    """Confere se o sidecar descreve este corpus e este JSONL."""
    if not progresso or not saida.is_file():
        return None
    try:
        n_proc = int(progresso.get("n_processos_concluidos", 0))
        offset = int(progresso.get("byte_offset", -1))
        n_chunks = int(progresso.get("n_chunks_corpus", -1))
        n_trip = int(progresso.get("tripletos_salvos", 0))
    except (TypeError, ValueError):
        return None
    if n_chunks != n_chunks_corpus or n_proc < 0 or n_proc > len(processos_validos) or n_trip < 0:
        return None
    tamanho = saida.stat().st_size
    if offset < 0 or offset > tamanho:
        return None
    # Sidecar zerado com arquivo cheio: nao e retomada, e um sidecar velho. Varre o JSONL.
    if n_proc == 0 and tamanho > 0:
        return None
    if n_proc > 0 and processos_validos[n_proc - 1].id_global != str(progresso.get("ultimo_id_global", "")).strip():
        return None
    return {
        "n_processos_concluidos": n_proc,
        "ultimo_id_global": str(progresso.get("ultimo_id_global", "")),
        "tripletos_salvos": n_trip,
        "byte_offset": offset,
        "n_chunks_corpus": n_chunks,
    }


def _cortar_jsonl(caminho: Path, byte_offset: int) -> None:
    # Descarta o processo gravado pela metade. O byte_offset e o inicio daquela linha.
    with caminho.open("r+b") as arquivo:
        arquivo.truncate(max(0, int(byte_offset)))


def _apagar_saida_tripletos(saida: Path) -> None:
    sidecar = _caminho_sidecar_progresso(saida)
    if saida.is_file():
        saida.unlink()
    if sidecar.is_file():
        sidecar.unlink()


def _id_global_da_linha(linha: bytes) -> str | None:
    # Regex primeiro: a linha do tripleto tem ementa e seis textos. json.loads e o plano B.
    achado = _RE_ID_GLOBAL.search(linha)
    if achado:
        return achado.group(1).decode("ascii", errors="ignore") or None
    texto = linha.decode("utf-8", errors="ignore").strip()
    if not texto:
        return None
    try:
        dado = json.loads(texto)
    except json.JSONDecodeError:
        return None
    valor = str(dado.get("id_global") or "").strip()
    return valor or None


def _varrer_jsonl_existente(saida: Path, n_chunks_corpus: int) -> dict[str, Any]:
    """Le o JSONL sem sidecar. Descarta o ultimo processo, que pode estar incompleto."""
    tamanho = saida.stat().st_size
    inicio_ultimo = 0
    ultimo_id = ""
    id_anterior = ""
    n_processos_completos = 0
    n_tripletos_completos = 0
    n_tripletos_ultimo = 0
    offset = 0

    with saida.open("rb") as arquivo, tqdm(
        total=tamanho,
        desc="Lendo tripletos ja gravados",
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as barra:
        for linha in arquivo:
            if not linha.strip():
                offset += len(linha)
                barra.update(len(linha))
                continue
            identificador = _id_global_da_linha(linha)
            if identificador is None:
                # Linha quebrada no fim do arquivo: para no comeco desta linha.
                break
            if identificador != ultimo_id:
                if ultimo_id:
                    n_processos_completos += 1
                    n_tripletos_completos += n_tripletos_ultimo
                    id_anterior = ultimo_id
                inicio_ultimo = offset
                ultimo_id = identificador
                n_tripletos_ultimo = 0
            n_tripletos_ultimo += 1
            offset += len(linha)
            barra.update(len(linha))

    # Sem sidecar nao da para saber se o ultimo processo terminou. Ele e refeito.
    _cortar_jsonl(saida, inicio_ultimo)
    return {
        "n_processos_concluidos": n_processos_completos,
        "ultimo_id_global": id_anterior,
        "tripletos_salvos": n_tripletos_completos,
        "byte_offset": inicio_ultimo,
        "n_chunks_corpus": int(n_chunks_corpus),
    }


def _indice_apos_processo(
    processos_validos: list[ProcessoJudicial],
    ultimo_id: str,
) -> int:
    """Indice do proximo processo depois de ultimo_id. 0 se a lista ainda nao comecou.
    Nao usa a contagem de linhas: processo sem 5 negativos nao aparece no JSONL.
    """
    if not ultimo_id:
        return 0
    for indice, proc in enumerate(processos_validos):
        if proc.id_global == ultimo_id:
            return indice + 1
    return -1


def _progresso_bate_com_corpus(
    progresso: dict[str, Any],
    processos_validos: list[ProcessoJudicial],
) -> bool:
    n_proc = int(progresso["n_processos_concluidos"])
    if n_proc < 0 or n_proc > len(processos_validos):
        return False
    if n_proc > 0 and processos_validos[n_proc - 1].id_global != progresso["ultimo_id_global"]:
        return False
    return True


def _preparar_saida_tripletos(
    saida: Path,
    n_chunks_corpus: int,
    processos_validos: list[ProcessoJudicial],
    retomar: bool,
) -> dict[str, Any]:
    """
    Decide se a busca recomeça, retoma pelo sidecar ou varre o JSONL antigo.
    Devolve n_processos_concluidos para o laco pular o que ja esta no disco.
    """
    sidecar = _caminho_sidecar_progresso(saida)
    if not retomar:
        print("Recalculo de tripletos solicitado. Arquivo anterior sera apagado.")
        _apagar_saida_tripletos(saida)
        saida.touch()
        progresso = _progresso_vazio(n_chunks_corpus)
        _gravar_progresso(sidecar, progresso)
        return progresso

    if saida.is_file() and saida.stat().st_size > 0:
        progresso = _progresso_compativel(
            _ler_progresso(sidecar), saida, n_chunks_corpus, processos_validos
        )
        if progresso is not None:
            sobra = saida.stat().st_size - progresso["byte_offset"]
            if sobra > 0:
                print(f"Retomando tripletos: cortando {sobra:,} bytes incompletos.")
            _cortar_jsonl(saida, progresso["byte_offset"])
            print(
                f"Retomando busca: {progresso['n_processos_concluidos']:,} processos ja gravados, "
                f"{progresso['tripletos_salvos']:,} tripletos no disco."
            )
            return progresso

        print("Sidecar ausente ou incompativel. Varrendo train_triplets.jsonl para retomar...")
        progresso = _varrer_jsonl_existente(saida, n_chunks_corpus)
        # Processos sem 5 negativos nao entram no JSONL. O indice vem do id, nao da contagem de linhas.
        indice = _indice_apos_processo(processos_validos, progresso["ultimo_id_global"])
        if indice < 0:
            print("Aviso: o ultimo processo do JSONL nao esta no corpus atual. Recomecando a busca.")
            _apagar_saida_tripletos(saida)
            saida.touch()
            progresso = _progresso_vazio(n_chunks_corpus)
            _gravar_progresso(sidecar, progresso)
            return progresso
        progresso["n_processos_concluidos"] = indice
        if not _progresso_bate_com_corpus(progresso, processos_validos):
            print("Aviso: a ordem dos processos no JSONL nao bate com o corpus. Recomecando a busca.")
            _apagar_saida_tripletos(saida)
            saida.touch()
            progresso = _progresso_vazio(n_chunks_corpus)
            _gravar_progresso(sidecar, progresso)
            return progresso
        _gravar_progresso(sidecar, progresso)
        print(
            f"Retomando busca: {progresso['n_processos_concluidos']:,} processos ja gravados, "
            f"{progresso['tripletos_salvos']:,} tripletos no disco."
        )
        return progresso

    saida.touch()
    progresso = _progresso_vazio(n_chunks_corpus)
    _gravar_progresso(sidecar, progresso)
    return progresso


def _posicao_bytes(arquivo) -> int:
    # tell() no buffer binario e o offset real. fsync grava o lote antes do sidecar.
    arquivo.flush()
    buffer = getattr(arquivo, "buffer", None)
    alvo = buffer if buffer is not None else arquivo
    try:
        os.fsync(alvo.fileno())
    except OSError:
        pass
    return int(alvo.tell())


def minerar_tripletos_contrastivos(
    modelo: SentenceTransformer,
    processos: list[ProcessoJudicial],
    chunks_corpus: list[ChunkProcesso],
    arquivo_saida: Path | str,
    n_negativos: int = 5,
    limiar_min: float = 0.60,
    limiar_max: float = 0.85,
    top_k: int = 200,
    encode_batch_size: int = 256,
    query_batch_size: int = 256,
    query_instruction: str = DEFAULT_QUERY_INSTRUCTION,
    ignorar_chunk_0_como_positivo: bool = True,
    model_id: str = "Qwen/Qwen3-Embedding-4B",
    arquivo_embeddings: Path | str | None = None,
    recomputar_embeddings: bool = False,
    dispositivos: list[str] | None = None,
    retomar: bool = True,
    excluir_vazados_como_negativos: bool = True,
    bloco_vetorizacao: int = 20_000,
    max_relativo_positivo: float | None = 0.95,
) -> dict[str, Any]:
    """
    Executa a mineracao em lote de tripletos.

    Otimizacoes:
    - Processamento Vetorial Massivo: Multiplicacao matricial nativa na GPU utilizando precisao bfloat16.
    - Mascaramento In-Place: Anulação imediata de candidatos pertencentes ao mesmo processo diretamente na memoria da GPU.
    - Mitigacao de Gargalo de I/O: Execucao do torch.topk diretamente na GPU. Apenas os indices estritamente necessarios sao transferidos para a CPU.
    - Avaliacao Focada: O calculo da similaridade de cosseno contra o texto positivo (ancora) e restrito ao subconjunto Top-K.
    - Vetores do ranking: candidatos e positivos saem da matriz na GPU, sem reler o .npy.
    - Retomada: o JSONL e aberto em append. O sidecar guarda o ultimo processo completo.

    Protecoes contra vazamento e falsos negativos:
    - Chunks do proprio processo e dos processos do mesmo grupo de quase duplicatas nunca sao negativos.
    - Chunks com a ementa copiada (contem_vazamento_ementa) nao sao negativos de ninguem
      (excluir_vazados_como_negativos) e nunca sao positivos (ignorar_chunk_0_como_positivo).
      Processo sem chunk limpo nao gera tuplas.
    - Filtro relativo ao positivo (NV-Retriever, Moreira et al., 2024): o negativo precisa ter
      similaridade com a consulta menor que max_relativo_positivo x a do positivo da tupla.
      O teto fixo de limiar_max deixa passar negativos acima de positivos fracos. None desliga.
    - Os vetores das consultas sao gravados em <saida>_consultas/, para a Etapa 3b aplicar
      o mesmo filtro aos negativos aleatorios.
    """
    saida = Path(arquivo_saida)
    saida.parent.mkdir(parents=True, exist_ok=True)

    device = modelo.device
    dtype_tensor = dtype_para_dispositivo(device)

    # 1. Indexacao rapida de processos
    indices_por_processo: dict[str, list[int]] = defaultdict(list)
    for idx, ch in enumerate(chunks_corpus):
        indices_por_processo[ch.id_global_processo].append(idx)

    processos_validos = [p for p in processos if p.id_global in indices_por_processo]
    textos_corpus = [ch.texto for ch in chunks_corpus]
    total_chunks = len(chunks_corpus)

    # Mascara de negativos: chunks do grupo de quase duplicatas inteiro, nao so do processo.
    indices_por_grupo: dict[str, list[int]] = defaultdict(list)
    for proc in processos_validos:
        indices_por_grupo[proc.grupo_duplicata].extend(indices_por_processo[proc.id_global])
    mascara_por_processo: dict[str, list[int]] = {
        proc.id_global: indices_por_grupo[proc.grupo_duplicata] for proc in processos_validos
    }
    processos_com_gemeos = sum(
        1 for proc in processos_validos
        if len(mascara_por_processo[proc.id_global]) > len(indices_por_processo[proc.id_global])
    )

    indices_vazados = (
        np.array([i for i, ch in enumerate(chunks_corpus) if ch.contem_vazamento_ementa], dtype=np.int64)
        if excluir_vazados_como_negativos
        else None
    )

    print(f"Total de chunks a vetorizar: {total_chunks:,}")
    print(f"Total de processos para consulta: {len(processos_validos):,}")
    print(f"Processos com quase duplicatas mascaradas: {processos_com_gemeos:,}")
    if indices_vazados is not None:
        print(f"Chunks com vazamento fora dos negativos : {len(indices_vazados):,}")

    # 2. Vetorizacao do corpus, ou reuso do .npy float16 ja gravado
    caminho_embeddings = Path(arquivo_embeddings) if arquivo_embeddings else (saida.parent / NOME_ARQUIVO_EMBEDDINGS)
    corpus_embs_cpu = obter_embeddings_corpus(
        modelo=modelo,
        textos=textos_corpus,
        caminho=caminho_embeddings,
        model_id=model_id,
        encode_batch_size=encode_batch_size,
        recomputar=recomputar_embeddings,
        dispositivos=dispositivos,
        ids_chunks=[ch.chunk_id for ch in chunks_corpus],
        bloco_vetorizacao=bloco_vetorizacao,
    )

    # 2b. Tuplas ja gravadas so valem se foram mineradas com as mesmas regras.
    # Sem isso, mudar a faixa ou o teto relativo e rodar de novo reaproveitaria tuplas antigas.
    parametros = {
        "n_negativos": n_negativos,
        "limiar_min": limiar_min,
        "limiar_max": limiar_max,
        "top_k": top_k,
        "max_relativo_positivo": max_relativo_positivo,
        "excluir_vazados_como_negativos": excluir_vazados_como_negativos,
        "ignorar_chunk_0_como_positivo": ignorar_chunk_0_como_positivo,
        "model_id": model_id,
    }
    arq_parametros = saida.with_suffix(".params.json")
    if retomar and saida.is_file() and saida.stat().st_size > 0:
        try:
            anteriores = json.loads(arq_parametros.read_text(encoding="utf-8"))
        except Exception:
            anteriores = None
        if anteriores != parametros:
            print("Tuplas existentes foram mineradas com outras regras (ou sem registro delas). Recomecando a busca.")
            retomar = False
    arq_parametros.write_text(json.dumps(parametros, ensure_ascii=False, indent=2), encoding="utf-8")

    # Retomada: corta linha incompleta e devolve de qual processo seguir.
    progresso = _preparar_saida_tripletos(
        saida=saida,
        n_chunks_corpus=total_chunks,
        processos_validos=processos_validos,
        retomar=retomar,
    )
    inicio = int(progresso["n_processos_concluidos"])
    caminho_progresso = _caminho_sidecar_progresso(saida)

    stats = {
        "processos_avaliados": inicio,
        "processos_pulados": inicio,
        "tripletos_salvos": int(progresso["tripletos_salvos"]),
        "tripletos_ja_gravados": int(progresso["tripletos_salvos"]),
        "tripletos_novos": 0,
        "chunks_positivos_avaliados": 0,
        "chunks_sem_negativos_suficientes": 0,
        "chunks_vazados_ignorados_como_positivo": 0,
        "processos_com_quase_duplicatas": processos_com_gemeos,
        "chunks_vazados_fora_dos_negativos": int(len(indices_vazados)) if indices_vazados is not None else 0,
        "tuplas_com_negativo_acima_do_positivo": 0,
        "tuplas_descartadas_filtro_relativo": 0,
        "max_relativo_positivo": max_relativo_positivo,
        "media_margem_tripletos": 0.0,
    }
    amostras_scores: dict[str, list] = {
        "melhor_candidato": [],
        "candidato_50": [],
        "melhor_positivo": [],
        "candidatos_na_faixa": [],
    }

    # Vetores das consultas, um arquivo por lote (a retomada mantem os lotes ja feitos)
    pasta_consultas = saida.with_name(saida.stem + "_consultas")
    if inicio == 0 and pasta_consultas.is_dir():
        shutil.rmtree(pasta_consultas)
    pasta_consultas.mkdir(parents=True, exist_ok=True)

    if inicio >= len(processos_validos):
        print("Todos os processos ja estao no JSONL. Busca ignorada.")
        return stats

    usar_duas_gpus = bool(dispositivos and len(dispositivos) >= 2)
    shards: list[tuple[int, torch.Tensor]] | None = None
    if usar_duas_gpus:
        print(f"Dividindo a matriz entre {len(dispositivos)} GPUs ...")
        shards = _montar_shards(corpus_embs_cpu, list(dispositivos), dtype_tensor) or None
    corpus_embs_gpu = None
    if shards is None:
        print(f"Transferindo matriz de embeddings {tuple(corpus_embs_cpu.shape)} para {device} em {dtype_tensor}...")
        corpus_embs_gpu = torch.from_numpy(corpus_embs_cpu).to(device=device, dtype=dtype_tensor)

    soma_margens = 0.0
    # top_k do YAML e 200; busca o dobro para a faixa 0.60-0.85 ainda ter 5 negativos.
    k_candidatos_gpu = min(max(top_k * 2, n_negativos * 10), total_chunks)

    # 3. Processamento em lote das consultas. Append preserva o que ja estava no disco.
    with saida.open("a", encoding="utf-8", newline="\n") as f_out:
        with tqdm(
            total=len(processos_validos),
            initial=inicio,
            desc="Minerando Tripletos (GPU Otimizada)",
            unit="proc",
        ) as barra:
            for start in range(inicio, len(processos_validos), query_batch_size):
                t_inicio = time.perf_counter()
                batch_procs = processos_validos[start : start + query_batch_size]
                queries_formatadas = [
                    preparar_query_com_instrucao(p.ementa, query_instruction)
                    for p in batch_procs
                ]

                with torch.no_grad():
                    # Vetorizacao em lote das consultas
                    q_embs_gpu = modelo.encode(
                        queries_formatadas,
                        batch_size=encode_batch_size,
                        normalize_embeddings=True,
                        convert_to_tensor=True,
                        show_progress_bar=False,
                    ).to(dtype=dtype_tensor, device=device)

                    # As contas por processo ficam na CPU, com estes vetores e a matriz do .npy
                    q_embs_cpu = q_embs_gpu.float().cpu().numpy()
                    # Grava antes do progresso: lote no sidecar sempre tem os vetores no disco
                    np.save(pasta_consultas / f"lote_{start:08d}.npy", q_embs_cpu.astype(np.float16))
                    (pasta_consultas / f"lote_{start:08d}.json").write_text(
                        json.dumps([p.id_global for p in batch_procs]), encoding="utf-8"
                    )

                    # Duas placas: top-k em cada metade e juncao. Uma placa: a matriz inteira, como antes.
                    t_consultas = time.perf_counter()
                    if shards is not None:
                        q_embs_host = torch.from_numpy(q_embs_cpu)
                        partes_scores = []
                        partes_indices = []
                        for offset, matriz in shards:
                            scores_fatia, indices_fatia = _topk_numa_fatia(
                                q_embs=q_embs_host,
                                matriz=matriz,
                                offset=offset,
                                batch_procs=batch_procs,
                                indices_por_processo=mascara_por_processo,
                                k=k_candidatos_gpu,
                                dtype_tensor=dtype_tensor,
                                indices_vazados=indices_vazados,
                            )
                            partes_scores.append(scores_fatia)
                            partes_indices.append(indices_fatia)
                        top_scores_cpu, top_indices_cpu = _juntar_topk(
                            partes_scores, partes_indices, k_candidatos_gpu
                        )
                    else:
                        # Multiplicacao de matrizes em lote na GPU
                        sims_batch_gpu = torch.matmul(q_embs_gpu, corpus_embs_gpu.T)

                        # Mascara o processo, seus quase duplicados e os chunks vazados, direto na GPU
                        _mascarar_mesmo_processo(
                            sims_batch_gpu, 0, total_chunks, batch_procs, mascara_por_processo, indices_vazados
                        )

                        # Top-K direto na GPU: extrai apenas os candidatos mais proximos
                        top_scores_gpu, top_indices_gpu = torch.topk(sims_batch_gpu, k=k_candidatos_gpu, dim=-1)

                        # Transfere para CPU apenas os indices e scores necessarios (array compacto)
                        top_scores_cpu = top_scores_gpu.cpu().float().numpy()
                        top_indices_cpu = top_indices_gpu.cpu().numpy()

                t_busca = time.perf_counter()
                # Processamento dos candidatos na CPU: poucas linhas por processo (~10 positivos x
                # ~400 candidatos), lidas da matriz do .npy.
                for i_local, proc in enumerate(batch_procs):
                    stats["processos_avaliados"] += 1
                    pos_indices = indices_por_processo[proc.id_global]

                    cands_idx = top_indices_cpu[i_local]
                    cands_score_q = top_scores_cpu[i_local]

                    # Filtra candidatos na faixa [limiar_min, limiar_max]
                    candidatos_validos = []
                    for c_idx, score_q in zip(cands_idx, cands_score_q):
                        if limiar_min <= score_q <= limiar_max:
                            candidatos_validos.append((int(c_idx), float(score_q)))

                    # Similaridade consulta -> positivos, para todos os processos (calibrar a faixa)
                    pos_indices_array = np.array(pos_indices, dtype=np.int64)
                    pos_embs_f32 = np.asarray(corpus_embs_cpu[pos_indices_array], dtype=np.float32)
                    scores_pos_vec = pos_embs_f32 @ q_embs_cpu[i_local]
                    limpos = [s for s, i in zip(scores_pos_vec, pos_indices) if not chunks_corpus[i].contem_vazamento_ementa]
                    validos_q = cands_score_q[cands_score_q > _LIMITE_MASCARADO]
                    if len(validos_q):
                        amostras_scores["melhor_candidato"].append(float(validos_q[0]))
                        amostras_scores["candidato_50"].append(float(validos_q[min(49, len(validos_q) - 1)]))
                    if limpos:
                        amostras_scores["melhor_positivo"].append(float(max(limpos)))
                    amostras_scores["candidatos_na_faixa"].append(len(candidatos_validos))

                    if len(candidatos_validos) < n_negativos:
                        stats["chunks_sem_negativos_suficientes"] += len(pos_indices)
                        continue

                    cand_indices_array = np.array([idx for idx, _ in candidatos_validos], dtype=np.int64)
                    cand_scores_q_array = np.array([s for _, s in candidatos_validos], dtype=np.float32)
                    cand_embs_f32 = np.asarray(corpus_embs_cpu[cand_indices_array], dtype=np.float32)
                    # (n_pos, n_cand): so os vetores deste processo
                    sims_p_mat = pos_embs_f32 @ cand_embs_f32.T

                    for j_pos, pos_idx in enumerate(pos_indices):
                        ch_positivo = chunks_corpus[pos_idx]

                        # Chunk com a ementa copiada nunca e positivo, mesmo sendo o unico do processo:
                        # treinaria o atalho lexical que a avaliacao remove.
                        if ignorar_chunk_0_como_positivo and ch_positivo.contem_vazamento_ementa:
                            stats["chunks_vazados_ignorados_como_positivo"] += 1
                            continue

                        stats["chunks_positivos_avaliados"] += 1

                        sims_p_array = sims_p_mat[j_pos]

                        # Filtro relativo: o negativo fica abaixo de 95% da similaridade deste positivo
                        if max_relativo_positivo is not None:
                            teto = max_relativo_positivo * float(scores_pos_vec[j_pos])
                            permitidos = np.nonzero(cand_scores_q_array < teto)[0]
                        else:
                            permitidos = np.arange(len(cand_scores_q_array))
                        if len(permitidos) < n_negativos:
                            stats["chunks_sem_negativos_suficientes"] += 1
                            stats["tuplas_descartadas_filtro_relativo"] += 1
                            continue

                        # Ordenacao pelos negativos mais desafiadores (prioriza maior similaridade com o positivo)
                        ordem_local = np.lexsort((cand_scores_q_array[permitidos], sims_p_array[permitidos]))[::-1]
                        ordem = permitidos[ordem_local[:n_negativos]]

                        neg_indices_finais = cand_indices_array[ordem]
                        neg_scores_finais = [round(float(s), 4) for s in cand_scores_q_array[ordem]]
                        neg_textos = [textos_corpus[idx] for idx in neg_indices_finais]
                        neg_ids = [chunks_corpus[idx].chunk_id for idx in neg_indices_finais]

                        score_pos = round(float(scores_pos_vec[j_pos]), 4)
                        margem = round(score_pos - max(neg_scores_finais), 4)

                        registro = {
                            "id_global": proc.id_global,
                            "str_tribunal": proc.tribunal,
                            "str_numero_unico": proc.numero_processo,
                            "query": proc.ementa,
                            "positive": ch_positivo.texto,
                            "positive_chunk_id": ch_positivo.chunk_id,
                            "positive_chunk_idx": ch_positivo.chunk_idx,
                            "negatives": neg_textos,
                            "negative_chunk_ids": neg_ids,
                            "score_positive": score_pos,
                            "scores_negativos": neg_scores_finais,
                            "margin": margem,
                        }

                        f_out.write(json.dumps(registro, ensure_ascii=False) + "\n")
                        stats["tripletos_salvos"] += 1
                        stats["tripletos_novos"] += 1
                        soma_margens += margem
                        # Negativo mais parecido com a consulta que o proprio positivo: suspeita de falso negativo
                        if margem < 0:
                            stats["tuplas_com_negativo_acima_do_positivo"] += 1

                t_tuplas = time.perf_counter()
                tqdm.write(
                    f"Lote {start:,}: consultas {t_consultas - t_inicio:.1f}s | busca {t_busca - t_consultas:.1f}s"
                    f" | tuplas {t_tuplas - t_busca:.1f}s"
                )
                # Processo so entra no sidecar depois do lote inteiro gravado e com flush.
                _gravar_progresso(
                    caminho_progresso,
                    {
                        "n_processos_concluidos": start + len(batch_procs),
                        "ultimo_id_global": batch_procs[-1].id_global,
                        "tripletos_salvos": stats["tripletos_salvos"],
                        "byte_offset": _posicao_bytes(f_out),
                        "n_chunks_corpus": total_chunks,
                    },
                )
                barra.update(len(batch_procs))
                # Sem empty_cache: modelo e fatias ficam na GPU. O alocador reaproveita o cache.

    if stats["tripletos_novos"] > 0:
        stats["media_margem_tripletos"] = round(soma_margens / stats["tripletos_novos"], 4)
    stats["distribuicao_scores"] = _resumir_scores(amostras_scores, n_negativos, limiar_min, limiar_max)

    return stats


def _resumir_scores(amostras: dict[str, list], n_negativos: int, limiar_min: float, limiar_max: float) -> dict[str, Any]:
    """
    Percentis das similaridades vistas nesta execucao. Servem para calibrar a faixa
    [limiar_min, limiar_max]: ela foi escolhida no TJMA e muda de escala com outro tribunal.
    """
    resumo: dict[str, Any] = {"faixa": [limiar_min, limiar_max]}
    for nome in ("melhor_candidato", "candidato_50", "melhor_positivo"):
        valores = np.asarray(amostras.get(nome, []), dtype=np.float32)
        if len(valores):
            resumo[nome] = {f"p{q}": round(float(np.percentile(valores, q)), 4) for q in (5, 25, 50, 75, 95)}
    na_faixa = np.asarray(amostras.get("candidatos_na_faixa", []))
    if len(na_faixa):
        resumo["candidatos_na_faixa_mediana"] = float(np.median(na_faixa))
        resumo["processos_com_menos_de_n_na_faixa"] = round(float(np.mean(na_faixa < n_negativos)), 4)
    return resumo
