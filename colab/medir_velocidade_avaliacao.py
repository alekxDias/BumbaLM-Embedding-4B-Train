"""
Mede a velocidade da avaliacao numa amostra do teste e estima horas e unidades do Colab por modelo.

Carrega o modelo do mesmo jeito que a Etapa 5 (scripts/05_evaluate.py), na precisao pedida, e
vetoriza os primeiros trechos do test_corpus_chunks.jsonl. Os trechos tem ate 768 tokens; as
consultas (ementas) costumam ser mais curtas, entao a estimativa tende a sobrar um pouco.

Uso (no Colab):
    python colab/medir_velocidade_avaliacao.py --modelo /content/modelos/Qwen3-Embedding-4B \
        --chunks /content/dados/processed_tjrs100k/test_corpus_chunks.jsonl --precisao tf32
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def carregar_etapa5():
    """Importa o scripts/05_evaluate.py (o nome comeca com digito, entao nao da para usar import)."""
    spec = importlib.util.spec_from_file_location("etapa5_avaliacao", REPO / "scripts" / "05_evaluate.py")
    modulo = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(modulo)
    return modulo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--modelo", required=True, help="Pasta do modelo (base ou best_model_4b).")
    parser.add_argument("--chunks", type=Path, required=True, help="test_corpus_chunks.jsonl")
    parser.add_argument("--precisao", default="tf32", choices=["auto", "fp16", "bf16", "fp32", "tf32"])
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--amostra", type=int, default=1024, help="Trechos vetorizados na medida (padrao: 1.024).")
    parser.add_argument("--max_seq_length", type=int, default=768)
    parser.add_argument("--textos_teste", type=int, default=9999 + 92101, help="Consultas + trechos limpos do teste.")
    parser.add_argument("--modelos", type=int, default=3, help="Modelos a avaliar (padrao: base, nenhum e dificeis).")
    parser.add_argument("--unidades_por_hora", type=float, default=8.88)
    args = parser.parse_args()

    import torch

    from src.hardware import aplicar_precisao_matmul

    if not torch.cuda.is_available():
        raise SystemExit("Sem GPU: a medida so faz sentido na placa da avaliacao.")
    etapa5 = carregar_etapa5()
    # De novo depois do import: o modulo da Etapa 5 nao mexe nisso, mas fica explicito aqui
    modo = aplicar_precisao_matmul(args.precisao)

    textos: list[str] = []
    with args.chunks.open(encoding="utf-8") as f:
        for linha in f:
            if linha.strip():
                textos.append(json.loads(linha)["texto"])
            if len(textos) >= args.amostra:
                break

    inicio = time.time()
    modelo = etapa5.carregar_modelo_avaliacao(args.modelo, args.max_seq_length, args.precisao)
    dtype = next(modelo.parameters()).dtype
    print(f"GPU             : {torch.cuda.get_device_name(0)}")
    print(f"Modelo          : {args.modelo} (carregado em {time.time() - inicio:.0f} s)")
    print(f"Pesos           : {dtype} | matmul em fp32: {modo} | precisao pedida: {args.precisao}")

    # Aquecimento: o primeiro lote inclui compilacao de kernels e alocacao de memoria
    modelo.encode(textos[: args.batch_size], batch_size=args.batch_size, show_progress_bar=False)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    inicio = time.time()
    modelo.encode(textos, batch_size=args.batch_size, show_progress_bar=False)
    torch.cuda.synchronize()
    segundos = time.time() - inicio

    por_texto = segundos / len(textos)
    horas_modelo = por_texto * args.textos_teste / 3600
    horas_total = horas_modelo * args.modelos
    pico_gb = torch.cuda.max_memory_allocated() / 1024**3
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"Amostra         : {len(textos):,} trechos em {segundos:.1f} s ({por_texto * 1000:.1f} ms por texto)")
    print(f"Memoria (pico)  : {pico_gb:.1f} GB de {total_gb:.0f} GB com lote de {args.batch_size}")
    print(f"Teste completo  : ~{horas_modelo:.2f} h por modelo ({args.textos_teste:,} textos)")
    print(f"{args.modelos} modelos      : ~{horas_total:.1f} h, ~{horas_total * args.unidades_por_hora:.0f} unidades "
          f"a {args.unidades_por_hora} por hora (fora a preparacao de cada sessao)")


if __name__ == "__main__":
    main()
