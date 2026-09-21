"""
Callback cientifico para telemetria de hardware (VRAM/RAM), tempo e metricas de treino.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Optional

import torch
from transformers import TrainerCallback


log = logging.getLogger("bumbalm_telemetry")


class ScientificMetricsCallback(TrainerCallback):
    """
    Coleta e persiste telemetria detalhada de hardware e metricas de otimizacao por step.
    Exporta relatorios em formato JSON estruturado e CSV tabular para geracao de graficos.
    """

    def __init__(self, output_dir: Path | str, nome_modelo: str = "Qwen/Qwen3-Embedding-4B"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.nome_modelo = nome_modelo
        self.start_time = time.time()
        self.metrics_history: list[dict[str, Any]] = []

        self.json_path = self.output_dir / "relatorio_treinamento.json"
        self.csv_path = self.output_dir / "metricas_treino_val.csv"

    def _obter_ram_gb(self) -> float:
        """Retorna a memoria RAM residente (RSS) utilizada pelo processo em GB."""
        try:
            import psutil
            return round(psutil.Process(os.getpid()).memory_info().rss / (1024 ** 3), 2)
        except Exception:
            return 0.0

    def _obter_vram_gb(self) -> tuple[float, float]:
        """Retorna (VRAM_alocada_gb, VRAM_reservada_gb) na GPU atual."""
        if torch.cuda.is_available():
            alocada = round(torch.cuda.max_memory_allocated() / (1024 ** 3), 2)
            reservada = round(torch.cuda.max_memory_reserved() / (1024 ** 3), 2)
            return alocada, reservada
        return 0.0, 0.0

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Disparado a cada registro de log durante o treinamento."""
        if not logs:
            return

        vram_aloc, vram_res = self._obter_vram_gb()
        ram_gb = self._obter_ram_gb()
        tempo_decorrido = round(time.time() - self.start_time, 2)

        registro = {
            "step": state.global_step,
            "epoch": round(state.epoch, 4) if state.epoch else 0.0,
            "tempo_segundos": tempo_decorrido,
            "vram_alocada_gb": vram_aloc,
            "vram_reservada_gb": vram_res,
            "ram_gb": ram_gb,
        }

        for k, v in logs.items():
            if isinstance(v, (int, float)):
                registro[k] = round(v, 6)

        self.metrics_history.append(registro)

    def on_train_end(self, args, state, control, **kwargs):
        """Consolida as metricas e exporta os arquivos ao finalizar o treinamento."""
        tempo_total_seg = round(time.time() - self.start_time, 2)
        vram_pico_aloc, vram_pico_res = self._obter_vram_gb()
        ram_pico = self._obter_ram_gb()

        horas = int(tempo_total_seg // 3600)
        minutos = int((tempo_total_seg % 3600) // 60)
        segundos = int(tempo_total_seg % 60)
        tempo_formatado = f"{horas}h {minutos}m {segundos}s"

        resumo = {
            "modelo": self.nome_modelo,
            "total_steps": state.global_step,
            "tempo_total_segundos": tempo_total_seg,
            "tempo_total_formatado": tempo_formatado,
            "vram_pico_alocada_gb": vram_pico_aloc,
            "vram_pico_reservada_gb": vram_pico_res,
            "ram_pico_gb": ram_pico,
            "historico_metricas": self.metrics_history,
        }

        # 1. Exporta resumo executivo em JSON
        with self.json_path.open("w", encoding="utf-8") as f:
            json.dump(resumo, f, indent=2, ensure_ascii=False)

        # 2. Exporta tabela consolidada em CSV para analise e graficos
        try:
            if self.metrics_history:
                todas_chaves = set()
                for item in self.metrics_history:
                    todas_chaves.update(item.keys())
                colunas = sorted(list(todas_chaves))

                with self.csv_path.open("w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=colunas)
                    writer.writeheader()
                    writer.writerows(self.metrics_history)
        except Exception as e:
            log.warning(f"Aviso ao persistir arquivo CSV de metricas: {e}")

        print("\n" + "=" * 64)
        print("Resumo Cientifico da Execucao:")
        print(f"  - Total de Steps Executados : {state.global_step:,}")
        print(f"  - Tempo Total Decorrido     : {tempo_formatado}")
        print(f"  - Pico de VRAM Alocada      : {vram_pico_aloc} GB")
        print(f"  - Pico de VRAM Reservada    : {vram_pico_res} GB")
        print(f"  - Pico de Memoria RAM       : {ram_pico} GB")
        print(f"Relatorio JSON salvo em       : {self.json_path}")
        print(f"Metricas CSV salvas em        : {self.csv_path}")
        print("=" * 64)
