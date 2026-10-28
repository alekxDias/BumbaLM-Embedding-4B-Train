#!/usr/bin/env python3
"""
Orquestrador Master do Pipeline BumbaLM-Embedding-4B.

Permite executar o pipeline completo de ponta a ponta ou etapas especificas:
  - split : 01_prepare_and_split.py (Limpeza e divisao estratificada em Train/Val/Test)
  - chunk : 02_generate_chunks.py (Fragmentacao em janelas de tokens e auditoria de Chunk 0)
  - mine  : 03_mine_negatives.py (Mineracao acelerada em GPU de hard negatives)
  - pairs : 03b_build_pairs.py (Pares ementa+chunk sem hard negatives; so in-batch)
  - train : 04_train.py (Fine-tuning contrastivo com telemetria cientifica)
  - eval  : 05_evaluate.py (Avaliacao e comparativo com baseline)

1. Execucao completa de ponta a ponta:
   python scripts/run_pipeline.py --all

2. Execucao completa definindo arquivo bruto de entrada e comparacao com baseline:
   python scripts/run_pipeline.py --all --input data/raw/acordaos_tribunais.jsonl --compare_baseline

3. Execucao completa otimizada para GPU H200 (141 GB):
   python scripts/run_pipeline.py --all --batch_size 96 --epochs 1 --lr 2e-5 --compare_baseline

4. Apenas a preparacao dos dados (limpeza profunda, splits 80/10/10 e chunking com auditoria):
   python scripts/run_pipeline.py --steps split,chunk

5. Apenas a mineracao vetorial de hard negatives na GPU (Etapa 3):
   python scripts/run_pipeline.py --steps mine --batch_size 128

6. Treinamento e avaliacao combinados (quando os dados e tripletos ja foram gerados):
   python scripts/run_pipeline.py --steps train,eval --batch_size 96 --epochs 1 --compare_baseline

7. Apenas treinamento contrastivo com parametros customizados:
   python scripts/run_pipeline.py --steps train --batch_size 96 --epochs 2 --lr 1.5e-5

7b. Treino nas duas GPUs (DDP, lote de 64 por placa):
   python scripts/run_pipeline.py --steps train --gpus 0,1

7c. Mineracao nas duas GPUs (vetorizacao e busca repartidas; nao usa torchrun):
   python scripts/run_pipeline.py --steps mine --gpus 0,1

7d. TJMA sem hard negatives (pares + treino in-batch + eval vs baseline):
   python scripts/run_pipeline.py --steps pairs,train,eval --n_negativos 0 --gpus 0 --output_dir experiments/exp_tjma_sem_hn --compare_baseline

7e. TJMA com hard negatives (mine + treino + eval vs baseline):
   python scripts/run_pipeline.py --steps mine,train,eval --gpus 0 --output_dir experiments/exp_tjma_com_hn --compare_baseline

8. Apenas avaliacao do modelo salvo contra o modelo baseline zero-shot no conjunto de teste:
   python scripts/run_pipeline.py --steps eval --compare_baseline

9. Apenas avaliacao no split de validacao ou em ambos os splits (val e test):
   python scripts/run_pipeline.py --steps eval --eval_split val
   python scripts/run_pipeline.py --steps eval --eval_split both --compare_baseline
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

ETAPAS_DISPONIVEIS = {
    "split": BASE_DIR / "scripts" / "01_prepare_and_split.py",
    "chunk": BASE_DIR / "scripts" / "02_generate_chunks.py",
    "mine": BASE_DIR / "scripts" / "03_mine_negatives.py",
    "pairs": BASE_DIR / "scripts" / "03b_build_pairs.py",
    "train": BASE_DIR / "scripts" / "04_train.py",
    "eval": BASE_DIR / "scripts" / "05_evaluate.py",
}

ORDEM_PADRAO = ["split", "chunk", "mine", "train", "eval"]
ETAPAS_VALIDAS = list(ETAPAS_DISPONIVEIS.keys())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Executa todas as 5 etapas sequencialmente.",
    )
    parser.add_argument(
        "--steps",
        type=str,
        default=None,
        help="Lista de etapas separadas por virgula. Opcoes: split, chunk, mine, pairs, train, eval.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Caminho do arquivo JSONL bruto com os processos (para a etapa split).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Sobrescreve o batch size para treino ou inferencia.",
    )
    parser.add_argument(
        "--mini_batch_size",
        type=int,
        default=None,
        help="Sobrescreve o mini_batch_size do GradCache para a etapa train.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Sobrescreve a quantidade de epocas para a etapa train.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Sobrescreve o learning rate para a etapa train.",
    )
    parser.add_argument(
        "--compare_baseline",
        action="store_true",
        help="Se informado, a etapa eval tambem avalia o modelo zero-shot para comparacao.",
    )
    parser.add_argument(
        "--eval_split",
        type=str,
        choices=["test", "val", "both"],
        default="test",
        help="Particao a ser avaliada na etapa eval (padrao: test).",
    )
    parser.add_argument(
        "--max_por_tribunal",
        type=int,
        default=None,
        help="Limita o numero de registros por tribunal na etapa split (ex: 500 para teste).",
    )
    parser.add_argument(
        "--smoke_test",
        action="store_true",
        help="Modo de execucao rapida ponta a ponta (500 reg/tribunal, batch menor) para validar o pipeline.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Retoma treinamento a partir de checkpoint ('auto' ou caminho).",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="GPUs das etapas mine e train. Ex.: '1' ou '0,1'.",
    )
    parser.add_argument(
        "--n_negativos",
        type=int,
        default=None,
        help="Hard negatives por instancia na etapa train. Use 0 para pares so com in-batch.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Diretorio do experimento (train) e/ou pasta de relatorios (eval).",
    )
    parser.add_argument(
        "--triplets_file",
        type=Path,
        default=None,
        help="JSONL de treino para a etapa train (triplets ou pairs).",
    )
    parser.add_argument(
        "--model_path",
        type=Path,
        default=None,
        help="Caminho do modelo treinado para a etapa eval.",
    )
    return parser.parse_args()


def executar_comando(comando: list[str], nome_etapa: str) -> bool:
    """Executa um subcomando monitorando tempo e saida."""
    print("\n" + "=" * 64)
    print(f"INICIANDO ETAPA: [{nome_etapa.upper()}]")
    print(f"Comando: {' '.join(comando)}")
    print("=" * 64 + "\n")

    inicio = time.time()
    resultado = subprocess.run(comando)
    tempo_decorrido = time.time() - inicio

    if resultado.returncode != 0:
        print("\n" + "!" * 64)
        print(f"ERRO: A etapa [{nome_etapa.upper()}] falhou com codigo de saida {resultado.returncode}.")
        print("!" * 64 + "\n")
        return False

    print("\n" + "-" * 64)
    print(f"ETAPA [{nome_etapa.upper()}] CONCLUIDA COM SUCESSO! (Tempo: {tempo_decorrido:.2f}s)")
    print("-" * 64 + "\n")
    return True


def main() -> None:
    args = parse_args()

    if args.smoke_test:
        if not args.steps:
            args.all = True
        if args.max_por_tribunal is None:
            args.max_por_tribunal = 500
        if args.epochs is None:
            args.epochs = 1
        print("=" * 64)
        print("MODO SMOKE TEST ATIVADO: Execucao rapida para validacao do pipeline.")
        print(f"Max reg/tribunal: {args.max_por_tribunal} | Hiperparametros: Originais do config_train.yaml")
        print("=" * 64)

    if not args.all and not args.steps:
        print("\nErro: Nenhuma etapa foi selecionada para execucao.")
        print("\nExemplos rapidos:")
        print("  - Teste rapido ponta a ponta: python scripts/run_pipeline.py --smoke_test")
        print("  - Executar todo o pipeline : python scripts/run_pipeline.py --all")
        print("  - Preparar dados e chunks  : python scripts/run_pipeline.py --steps split,chunk")
        print("  - Treinar e avaliar        : python scripts/run_pipeline.py --steps train,eval --compare_baseline")
        print("  - Sem HN (in-batch): python scripts/run_pipeline.py --steps pairs,train,eval --n_negativos 0 --compare_baseline")
        print("  - Apenas avaliar modelo    : python scripts/run_pipeline.py --steps eval --compare_baseline")
        print("\nPara ver todas as opcoes e exemplos detalhados: python scripts/run_pipeline.py --help\n")
        sys.exit(1)

    if args.all:
        etapas_selecionadas = list(ORDEM_PADRAO)
    else:
        etapas_selecionadas = [s.strip().lower() for s in args.steps.split(",") if s.strip()]
        for e in etapas_selecionadas:
            if e not in ETAPAS_DISPONIVEIS:
                print(f"Erro: Etapa desconhecida '{e}'. Opcoes validas: {', '.join(ETAPAS_VALIDAS)}")
                sys.exit(1)

    python_bin = sys.executable

    print("=" * 64)
    print("ORQUESTRADOR DO PIPELINE BUMBALM-EMBEDDING-4B")
    print("=" * 64)
    print(f"Interpretador Python : {python_bin}")
    print(f"Etapas Selecionadas  : {' -> '.join([e.upper() for e in etapas_selecionadas])}")
    print("=" * 64)

    tempo_inicio_total = time.time()

    for etapa in etapas_selecionadas:
        script_path = str(ETAPAS_DISPONIVEIS[etapa])
        cmd = [python_bin, script_path]

        if etapa == "split":
            if args.input:
                cmd.extend(["--input", str(args.input)])
            if args.max_por_tribunal:
                cmd.extend(["--max_por_tribunal", str(args.max_por_tribunal)])

        if etapa == "mine":
            if args.batch_size:
                cmd.extend(["--batch_size", str(args.batch_size)])
            # A mineracao reparte vetorizacao e busca. Nao usa o torchrun do treino.
            if args.gpus:
                cmd.extend(["--gpus", args.gpus])

        if etapa == "pairs":
            # Usa diretorio processado do config_base; sem flags obrigatorias.
            pass

        if etapa == "train":
            if args.batch_size:
                cmd.extend(["--batch_size", str(args.batch_size)])
            if args.mini_batch_size:
                cmd.extend(["--mini_batch_size", str(args.mini_batch_size)])
            if args.epochs:
                cmd.extend(["--epochs", str(args.epochs)])
            if args.lr:
                cmd.extend(["--lr", str(args.lr)])
            if args.resume_from_checkpoint:
                cmd.extend(["--resume_from_checkpoint", str(args.resume_from_checkpoint)])
            if args.gpus:
                cmd.extend(["--gpus", args.gpus])
            if args.n_negativos is not None:
                cmd.extend(["--n_negativos", str(args.n_negativos)])
            if args.triplets_file is not None:
                cmd.extend(["--triplets_file", str(args.triplets_file)])
            if args.output_dir is not None:
                cmd.extend(["--output_dir", str(args.output_dir)])

        if etapa == "eval":
            cmd.extend(["--split", args.eval_split])
            if args.compare_baseline:
                cmd.append("--compare_baseline")
            if args.batch_size:
                cmd.extend(["--batch_size", str(args.batch_size)])
            if args.model_path is not None:
                cmd.extend(["--model_path", str(args.model_path)])
            elif args.output_dir is not None:
                # Se train salvou em --output_dir, eval procura best_model_4b la.
                cmd.extend(["--model_path", str(Path(args.output_dir) / "best_model_4b")])
                cmd.extend(["--output_dir", str(Path(args.output_dir) / "avaliacoes")])

        sucesso = executar_comando(cmd, etapa)
        if not sucesso:
            print(f"Pipeline abortado devido a falha na etapa [{etapa}].")
            sys.exit(1)

    tempo_total = time.time() - tempo_inicio_total
    print("\n" + "=" * 64)
    print(f"PIPELINE COMPLETO EXECUTADO COM SUCESSO! Tempo total: {tempo_total:.2f}s")
    print("=" * 64)


if __name__ == "__main__":
    main()
