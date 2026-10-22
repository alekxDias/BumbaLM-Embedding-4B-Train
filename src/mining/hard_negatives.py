"""
Mineracao vetorial acelerada de Hard Negatives em GPU para aprendizado contrastivo.
"""

from __future__ import annotations

import gc
import inspect
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.data.schema import ChunkProcesso, ProcessoJudicial


DEFAULT_QUERY_INSTRUCTION = (
    "Instruct: Dado um trecho de um documento jurídico brasileiro, "
    "recupere o trecho mais relevante semanticamente.\nQuery: "
)

NOME_ARQUIVO_EMBEDDINGS = "corpus_embeddings_f16.npy"
_BLOCO_GRAVACAO_EMBEDDINGS = 100_000
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
    max_seq_len: int = 1024,
    device: Optional[str] = None,
    attn_implementation: str = "sdpa",
) -> SentenceTransformer:
    """
    Inicializa o modelo SentenceTransformer para inferencia vetorial e calculo de similaridades.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    dtype_modelo = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else (torch.float16 if device == "cuda" else torch.float32)

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


def _embeddings_compativeis(caminho: Path, n_chunks: int, model_id: str) -> bool:
    """Confere se o .npy e o sidecar descrevem este corpus e este modelo."""
    sidecar = _caminho_sidecar_embeddings(caminho)
    if not caminho.is_file() or not sidecar.is_file():
        return False
    try:
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
        if int(meta.get("n_chunks", -1)) != n_chunks or meta.get("model_id") != model_id:
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
) -> np.ndarray:
    """
    Reutiliza o .npy quando ele bate com o corpus. Senao, vetoriza e grava em float16.
    """
    destino = Path(caminho)
    n_chunks = len(textos)

    if not recomputar and _embeddings_compativeis(destino, n_chunks, model_id):
        print(f"Reutilizando embeddings: {destino}")
        print("Vetorizacao do corpus ignorada.")
        return carregar_embeddings_float16(destino)

    if recomputar:
        print("Recalculo de embeddings solicitado.")
    elif destino.is_file():
        print("Embeddings existentes nao batem com este corpus ou modelo. Vetorizando de novo...")
    else:
        print("Arquivo de embeddings ausente. Vetorizando o corpus...")

    print("Gerando embeddings de todos os chunks do corpus ...")
    embeddings = encodar_textos_corpus(
        modelo=modelo,
        textos=textos,
        batch_size=encode_batch_size,
        show_progress=True,
        dispositivos=dispositivos,
    )
    salvar_embeddings_float16(destino, embeddings, model_id=model_id)
    del embeddings
    gc.collect()
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
) -> None:
    # O indice global vira indice local da fatia. Chunk de outro processo, ou de outra placa, fica de fora.
    for i_local, proc in enumerate(batch_procs):
        locais = [
            idx - offset
            for idx in indices_por_processo[proc.id_global]
            if offset <= idx < offset + n_local
        ]
        if locais:
            sims[i_local, locais] = -1e9


def _topk_numa_fatia(
    q_embs: torch.Tensor,
    matriz: torch.Tensor,
    offset: int,
    batch_procs: list[ProcessoJudicial],
    indices_por_processo: dict[str, list[int]],
    k: int,
    dtype_tensor: torch.dtype,
) -> tuple[np.ndarray, np.ndarray]:
    # As ementas nascem numa placa so. A copia para a fatia e pequena: 256 vetores.
    q_local = q_embs.to(device=matriz.device, dtype=dtype_tensor)
    sims = torch.matmul(q_local, matriz.T)
    _mascarar_mesmo_processo(sims, offset, matriz.shape[0], batch_procs, indices_por_processo)
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
    # O sinal negativo coloca os mais similares primeiro. -1e9, o mascarado, vai para o fim.
    ordem = np.argsort(-scores, axis=1)[:, :k_final]
    return np.take_along_axis(scores, ordem, axis=1), np.take_along_axis(indices, ordem, axis=1)


def _coletar_linhas_do_corpus(
    indices: np.ndarray,
    shards: list[tuple[int, torch.Tensor]] | None,
    corpus_embs_gpu: torch.Tensor | None,
    dispositivo: torch.device,
    dtype_tensor: torch.dtype,
) -> torch.Tensor:
    """
    Copia so as linhas pedidas a partir da matriz que ja esta na GPU.
    Nao volta no .npy mapeado em disco.
    """
    idx = torch.as_tensor(np.ascontiguousarray(indices, dtype=np.int64), dtype=torch.long)
    if idx.ndim != 1:
        idx = idx.reshape(-1)
    if shards is None:
        if corpus_embs_gpu is None:
            raise RuntimeError("Matriz do corpus ausente na GPU.")
        dim = corpus_embs_gpu.shape[1]
        fonte = corpus_embs_gpu
    else:
        dim = shards[0][1].shape[1]
        fonte = None
    if idx.numel() == 0:
        return torch.empty((0, dim), device=dispositivo, dtype=dtype_tensor)

    # Uma placa: a matriz inteira ja esta em corpus_embs_gpu.
    if fonte is not None:
        return fonte.index_select(0, idx.to(fonte.device)).to(device=dispositivo, dtype=dtype_tensor)

    # Duas placas: cada indice global cai numa fatia. O local e indice menos o offset.
    saida = torch.empty((idx.shape[0], dim), device=dispositivo, dtype=dtype_tensor)
    idx_cpu = idx.cpu()
    for offset, matriz in shards:
        n_local = matriz.shape[0]
        no_shard = (idx_cpu >= offset) & (idx_cpu < offset + n_local)
        if not bool(no_shard.any()):
            continue
        posicoes = torch.nonzero(no_shard, as_tuple=False).squeeze(1)
        locais = (idx_cpu[no_shard] - offset).to(device=matriz.device)
        linhas = matriz.index_select(0, locais).to(device=dispositivo, dtype=dtype_tensor)
        saida[posicoes.to(dispositivo)] = linhas
    return saida


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
    """
    saida = Path(arquivo_saida)
    saida.parent.mkdir(parents=True, exist_ok=True)

    device = modelo.device
    dtype_tensor = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else (torch.float16 if device.type == "cuda" else torch.float32)

    # 1. Indexacao rapida de processos
    indices_por_processo: dict[str, list[int]] = defaultdict(list)
    for idx, ch in enumerate(chunks_corpus):
        indices_por_processo[ch.id_global_processo].append(idx)

    processos_validos = [p for p in processos if p.id_global in indices_por_processo]
    textos_corpus = [ch.texto for ch in chunks_corpus]
    total_chunks = len(chunks_corpus)

    print(f"Total de chunks a vetorizar: {total_chunks:,}")
    print(f"Total de processos para consulta: {len(processos_validos):,}")

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
    )

    # 2b. Retomada: corta linha incompleta e devolve de qual processo seguir.
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
        "media_margem_tripletos": 0.0,
    }

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

                    # Duas placas: top-k em cada metade e juncao. Uma placa: a matriz inteira, como antes.
                    if shards is not None:
                        partes_scores = []
                        partes_indices = []
                        for offset, matriz in shards:
                            scores_fatia, indices_fatia = _topk_numa_fatia(
                                q_embs=q_embs_gpu,
                                matriz=matriz,
                                offset=offset,
                                batch_procs=batch_procs,
                                indices_por_processo=indices_por_processo,
                                k=k_candidatos_gpu,
                                dtype_tensor=dtype_tensor,
                            )
                            partes_scores.append(scores_fatia)
                            partes_indices.append(indices_fatia)
                        top_scores_cpu, top_indices_cpu = _juntar_topk(
                            partes_scores, partes_indices, k_candidatos_gpu
                        )
                    else:
                        # Multiplicacao de matrizes em lote na GPU
                        sims_batch_gpu = torch.matmul(q_embs_gpu, corpus_embs_gpu.T)

                        # Mascara chunks do mesmo processo judicial diretamente na GPU
                        for i_local, proc in enumerate(batch_procs):
                            pos_idx_list = indices_por_processo[proc.id_global]
                            sims_batch_gpu[i_local, pos_idx_list] = -1e9

                        # Top-K direto na GPU: extrai apenas os candidatos mais proximos
                        top_scores_gpu, top_indices_gpu = torch.topk(sims_batch_gpu, k=k_candidatos_gpu, dim=-1)

                        # Transfere para CPU apenas os indices e scores necessarios (array compacto)
                        top_scores_cpu = top_scores_gpu.cpu().float().numpy()
                        top_indices_cpu = top_indices_gpu.cpu().numpy()

                # Processamento dos candidatos. Vetores vêm da GPU, nao do .npy.
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

                    if len(candidatos_validos) < n_negativos:
                        stats["chunks_sem_negativos_suficientes"] += len(pos_indices)
                        continue

                    cand_indices_array = np.array([idx for idx, _ in candidatos_validos], dtype=np.int64)
                    cand_scores_q_array = np.array([s for _, s in candidatos_validos], dtype=np.float32)
                    pos_indices_array = np.array(pos_indices, dtype=np.int64)
                    cand_embs_gpu = _coletar_linhas_do_corpus(
                        cand_indices_array, shards, corpus_embs_gpu, device, dtype_tensor
                    )
                    pos_embs_gpu = _coletar_linhas_do_corpus(
                        pos_indices_array, shards, corpus_embs_gpu, device, dtype_tensor
                    )
                    cand_embs_f32 = cand_embs_gpu.float()
                    pos_embs_f32 = pos_embs_gpu.float()
                    # (n_pos, n_cand) e (n_pos,): so os vetores deste processo, ja na GPU.
                    sims_p_mat = torch.matmul(pos_embs_f32, cand_embs_f32.T).cpu().numpy()
                    scores_pos_vec = torch.matmul(
                        pos_embs_f32, q_embs_gpu[i_local].float()
                    ).cpu().numpy()

                    for j_pos, pos_idx in enumerate(pos_indices):
                        ch_positivo = chunks_corpus[pos_idx]

                        # Filtro do Chunk 0: evita usar cabeçalho se houver outros chunks limpos
                        if (
                            ignorar_chunk_0_como_positivo
                            and ch_positivo.contem_vazamento_ementa
                            and len(pos_indices) > 1
                        ):
                            continue

                        stats["chunks_positivos_avaliados"] += 1

                        sims_p_array = sims_p_mat[j_pos]

                        # Ordenacao pelos negativos mais desafiadores (prioriza maior similaridade com o positivo)
                        ordem = np.lexsort((cand_scores_q_array, sims_p_array))[::-1][:n_negativos]

                        if len(ordem) < n_negativos:
                            stats["chunks_sem_negativos_suficientes"] += 1
                            continue

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

    return stats
