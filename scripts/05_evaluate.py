#!/usr/bin/env python3
"""
Pipeline BumbaLM - Etapa 5: Avaliacao e Comparativo entre Modelos.

Executa:
1. Leitura das configuracoes do projeto e splits de avaliacao ('test_processos.jsonl' e 'test_corpus_chunks.jsonl').
2. Descoberta das variantes treinadas em experiments/<nome_experimento>/<variante>/melhor_checkpoint.json
   (dificeis, aleatorios, nenhum) ou leitura dos modelos passados em --modelo.
3. Avaliacao opcional do modelo base zero-shot (--compare_baseline).
4. Execucao da avaliacao de Information Retrieval (IR) com o mesmo corpus para todos os modelos.
5. Emissao de relatorios consolidados (JSON e TXT) com nDCG@10, MRR@10, Acc@1, 5, 10 e MAP@100.

Os modelos sao carregados um por vez, na precisao da GPU (fp16 na V100), e liberados em seguida.

Uso:
    python scripts/05_evaluate.py --compare_baseline
    python scripts/05_evaluate.py --compare_baseline --split both
    python scripts/05_evaluate.py --modelo dificeis=experiments/x/dificeis/checkpoint-1200 --compare_baseline
    python scripts/05_evaluate.py --model_path experiments/x/best_model_4b
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
from pathlib import Path
from typing import Any, Optional

# Adiciona a raiz do repositorio ao PYTHONPATH
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import torch
from sentence_transformers import SentenceTransformer

from src.evaluation.ir_evaluator import executar_avaliacao_completa
from src.hardware import dtype_para_dispositivo

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("bumbalm_evaluate")

# Ordem das colunas no relatorio. Variantes fora da lista vem depois, em ordem alfabetica.
ORDEM_VARIANTES = ["baseline", "nenhum", "aleatorios", "dificeis"]
METRICAS_RELATORIO = ["ndcg@10", "mrr@10", "map@100", "accuracy@1", "accuracy@5", "accuracy@10",
                      "precision@10", "recall@10"]


def carregar_yaml(caminho: Path) -> dict[str, Any]:
    """Le arquivo YAML com seguranca."""
    if not caminho.is_file():
        raise FileNotFoundError(f"Arquivo de configuracao nao encontrado: {caminho}")
    try:
        import yaml
        with caminho.open(encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        return {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config_base", type=Path, default=BASE_DIR / "configs" / "config_base.yaml")
    parser.add_argument("--config_train", type=Path, default=BASE_DIR / "configs" / "config_train.yaml")
    parser.add_argument(
        "--experimento_dir",
        type=Path,
        default=None,
        help="Pasta com as variantes treinadas. Padrao: experiments/<nome_experimento>.",
    )
    parser.add_argument(
        "--modelo",
        action="append",
        default=[],
        metavar="NOME=CAMINHO",
        help="Modelo extra a avaliar. Pode repetir. Se usado, a descoberta automatica e desligada.",
    )
    parser.add_argument(
        "--model_path",
        type=Path,
        default=None,
        help="Atalho para um unico modelo treinado (equivale a --modelo treinado=CAMINHO).",
    )
    parser.add_argument(
        "--compare_baseline",
        action="store_true",
        help="Avalia tambem o modelo base zero-shot.",
    )
    parser.add_argument(
        "--baseline_id",
        type=str,
        default=None,
        help="Modelo base zero-shot. Padrao: modelo.model_id do config_train.yaml.",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["test", "val", "both"],
        default="test",
        help="Particao a ser avaliada: test, val ou both. Padrao: test.",
    )
    parser.add_argument("--data_dir", type=Path, default=None, help="Diretorio com os dados processados.")
    parser.add_argument("--output_dir", type=Path, default=None, help="Diretorio dos relatorios de avaliacao.")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size da inferencia. Padrao: avaliacao.eval_batch_size do config_train.yaml (32 na V100).",
    )
    parser.add_argument(
        "--precisao",
        type=str,
        choices=["auto", "fp16", "bf16", "fp32"],
        default=None,
        help="Precisao dos pesos na inferencia. Padrao: avaliacao.precisao do YAML ou auto (fp16 na V100).",
    )
    return parser.parse_args()


def _absoluto(caminho: Path | str) -> Path:
    p = Path(caminho)
    return p if p.is_absolute() else BASE_DIR / p


def descobrir_variantes(dir_experimento: Path) -> dict[str, str]:
    """Le melhor_checkpoint.json de cada subpasta. Devolve {variante: caminho do modelo}."""
    modelos: dict[str, str] = {}
    if not dir_experimento.is_dir():
        return modelos
    for registro in sorted(dir_experimento.glob("*/melhor_checkpoint.json")):
        dados = json.loads(registro.read_text(encoding="utf-8"))
        caminho = dados.get("modelo_para_avaliar") or dados.get("modelo_final")
        if caminho and Path(caminho).exists():
            modelos[registro.parent.name] = caminho
        else:
            log.warning("Modelo de %s nao encontrado em disco: %s", registro.parent.name, caminho)
    # Pasta antiga, de antes das variantes: best_model_4b direto no experimento.
    if not modelos and (dir_experimento / "best_model_4b").is_dir():
        modelos["treinado"] = str(dir_experimento / "best_model_4b")
    return modelos


def carregar_modelo_avaliacao(caminho: str, max_seq_length: int, precisao: str = "auto") -> SentenceTransformer:
    """Carrega na precisao da GPU com padding a esquerda, exigido pelo last-token pooling."""
    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = dtype_para_dispositivo(dispositivo, precisao)
    modelo = SentenceTransformer(
        caminho,
        trust_remote_code=True,
        device=dispositivo,
        model_kwargs={"torch_dtype": dtype},
        tokenizer_kwargs={"padding_side": "left"},
    )
    # Mesmo limite do treino para todos. O base no Hub viria com 32k tokens.
    modelo.max_seq_length = max_seq_length
    return modelo


def _ordenar_nomes(nomes: list[str]) -> list[str]:
    conhecidos = [n for n in ORDEM_VARIANTES if n in nomes]
    return conhecidos + sorted(n for n in nomes if n not in ORDEM_VARIANTES)


def gerar_relatorio_comparativo(
    metricas_por_modelo: dict[str, dict[str, float]],
    nome_split: str,
    caminho_saida: Path,
) -> None:
    """Tabela com uma coluna por modelo. Com baseline, acrescenta o ganho absoluto sobre ele."""
    nomes = _ordenar_nomes(list(metricas_por_modelo))
    base = metricas_por_modelo.get("baseline")
    largura = max(12, *(len(n) for n in nomes)) + 2

    with caminho_saida.open("w", encoding="utf-8") as f:
        f.write("=" * 72 + "\n")
        f.write(f"RELATORIO COMPARATIVO DE RECUPERACAO JURIDICA (SPLIT: {nome_split.upper()})\n")
        f.write("=" * 72 + "\n\n")
        f.write(f"{'Metrica':<14}" + "".join(f"{n:>{largura}}" for n in nomes) + "\n")
        f.write("-" * (14 + largura * len(nomes)) + "\n")
        for met in METRICAS_RELATORIO:
            if not any(met in m for m in metricas_por_modelo.values()):
                continue
            linha = f"{met:<14}"
            for nome in nomes:
                valor = metricas_por_modelo[nome].get(met)
                linha += f"{valor:>{largura}.4f}" if valor is not None else f"{'-':>{largura}}"
            f.write(linha + "\n")

        if base:
            f.write("\nGanho absoluto sobre o baseline\n")
            outros = [n for n in nomes if n != "baseline"]
            f.write(f"{'Metrica':<14}" + "".join(f"{n:>{largura}}" for n in outros) + "\n")
            f.write("-" * (14 + largura * len(outros)) + "\n")
            for met in METRICAS_RELATORIO:
                if met not in base:
                    continue
                linha = f"{met:<14}"
                for nome in outros:
                    valor = metricas_por_modelo[nome].get(met)
                    linha += f"{valor - base[met]:>+{largura}.4f}" if valor is not None else f"{'-':>{largura}}"
                f.write(linha + "\n")
        f.write("\n" + "=" * 72 + "\n")


def main() -> None:
    args = parse_args()

    cfg_base = carregar_yaml(args.config_base)
    cfg_train = carregar_yaml(args.config_train)
    modelo_cfg = cfg_train.get("modelo", {})

    dir_processados = _absoluto(args.data_dir or cfg_base.get("dados", {}).get("diretorio_processados", "data/processed"))
    exp_dir_base = Path(cfg_base.get("saidas", {}).get("diretorio_experimentos", "experiments"))
    exp_nome = cfg_base.get("saidas", {}).get("nome_experimento", "exp_001_h200_4b")
    dir_experimento = _absoluto(args.experimento_dir or (exp_dir_base / exp_nome))

    # Modelos a avaliar: explicitos tem prioridade sobre a descoberta automatica
    modelos: dict[str, str] = {}
    for item in args.modelo:
        if "=" not in item:
            raise ValueError(f"--modelo espera NOME=CAMINHO, recebeu: {item}")
        nome, caminho = item.split("=", 1)
        modelos[nome.strip()] = str(_absoluto(caminho.strip()))
    if args.model_path:
        modelos["treinado"] = str(_absoluto(args.model_path))
    if not modelos:
        modelos = descobrir_variantes(dir_experimento)

    baseline_id = args.baseline_id or modelo_cfg.get("model_id", "Qwen/Qwen3-Embedding-4B")
    if args.compare_baseline:
        modelos = {"baseline": baseline_id, **modelos}

    if not modelos:
        raise FileNotFoundError(
            f"Nenhum modelo para avaliar em {dir_experimento}.\n"
            "Rode 'python scripts/04_train.py --negativos <variante>' antes, passe --modelo ou use --compare_baseline."
        )

    dir_saida = _absoluto(args.output_dir or (dir_experimento / "avaliacoes"))
    dir_saida.mkdir(parents=True, exist_ok=True)

    query_instruction = modelo_cfg.get(
        "query_instruction",
        "Instruct: Dado um trecho de um documento jurídico brasileiro, recupere o trecho mais relevante semanticamente.\nQuery: ",
    )
    max_seq_length = int(modelo_cfg.get("max_seq_length", 768))
    precisao = args.precisao or str(cfg_train.get("avaliacao", {}).get("precisao", "auto"))
    if args.batch_size is None:
        args.batch_size = int(cfg_train.get("avaliacao", {}).get("eval_batch_size", 32))
    splits_para_avaliar = ["test"] if args.split == "test" else (["val"] if args.split == "val" else ["val", "test"])

    print("=" * 64)
    print("BumbaLM - Etapa 5: Avaliacao In-Domain de Recuperacao Juridica")
    print("=" * 64)
    for nome, caminho in modelos.items():
        print(f"Modelo {nome:<12}: {caminho}")
    print(f"Splits Alvo          : {', '.join(splits_para_avaliar)}")
    print(f"Diretorio de Dados   : {dir_processados}")
    print(f"Diretorio de Saida   : {dir_saida}")
    print(f"Batch Size Avaliacao : {args.batch_size} | Max tokens: {max_seq_length} | Precisao: {precisao}")
    print("-" * 64)

    # Arquivos de cada split, resolvidos uma vez
    arquivos_split: dict[str, tuple[Path, Path]] = {}
    for split_nome in splits_para_avaliar:
        nome_pt = "teste" if split_nome == "test" else "validacao"
        arq_processos = dir_processados / f"{split_nome}_processos.jsonl"
        if not arq_processos.is_file():
            arq_processos = dir_processados / f"{nome_pt}_processos.jsonl"
        arq_chunks = dir_processados / f"{split_nome}_corpus_chunks.jsonl"
        if not arq_chunks.is_file():
            arq_chunks = dir_processados / f"{nome_pt}_corpus_chunks.jsonl"
        if arq_processos.is_file() and arq_chunks.is_file():
            arquivos_split[split_nome] = (arq_processos, arq_chunks)
        else:
            log.warning("Arquivos para o split '%s' nao encontrados em %s. Pulando...", split_nome, dir_processados)

    # Um modelo por vez na GPU: dois 4B em fp16 nao cabem com folga em 32 GB
    resultados: dict[str, dict[str, Any]] = {split: {} for split in arquivos_split}
    for nome, caminho in modelos.items():
        print(f"\nCarregando modelo '{nome}': {caminho} ...")
        modelo = carregar_modelo_avaliacao(caminho, max_seq_length, precisao)
        for split_nome, (arq_processos, arq_chunks) in arquivos_split.items():
            print(f"\n--- Avaliando '{nome}' no split {split_nome.upper()} ---")
            resultados[split_nome][nome] = executar_avaliacao_completa(
                model=modelo,
                caminho_processos=arq_processos,
                caminho_chunks=arq_chunks,
                output_dir=dir_saida / split_nome / nome,
                nome_avaliacao=f"{split_nome}_{nome}",
                query_instruction=query_instruction,
                excluir_chunks_contaminados=True,
                batch_size=args.batch_size,
                avaliar_por_tribunal=True,
            )
        del modelo
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    relatorio_consolidado_geral: dict[str, Any] = {"modelos": modelos, "splits": {}}
    for split_nome, por_modelo in resultados.items():
        caminho_relatorio_txt = dir_saida / f"comparativo_{split_nome}.txt"
        gerar_relatorio_comparativo(
            metricas_por_modelo={n: r.get("metricas_global", {}) for n, r in por_modelo.items()},
            nome_split=split_nome,
            caminho_saida=caminho_relatorio_txt,
        )
        relatorio_consolidado_geral["splits"][split_nome] = por_modelo

        print("\n" + "=" * 64)
        print(f"RESUMO DOS RESULTADOS GLOBAIS - SPLIT: {split_nome.upper()}")
        print("=" * 64)
        print(caminho_relatorio_txt.read_text(encoding="utf-8"))

    caminho_consolidado_json = dir_saida / "avaliacao_consolidada.json"
    with caminho_consolidado_json.open("w", encoding="utf-8") as f:
        json.dump(relatorio_consolidado_geral, f, indent=2, ensure_ascii=False)

    print("=" * 64)
    print("Avaliacao finalizada com sucesso!")
    print(f"Relatorios e metricas gravados em: {dir_saida}")
    print("=" * 64)


if __name__ == "__main__":
    main()
