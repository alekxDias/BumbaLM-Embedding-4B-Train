"""
Subpacote de treinamento, arquitetura e telemetria do BumbaLM.
"""

from src.training.telemetry import ScientificMetricsCallback
from src.training.model_builder import construir_modelo_embedding
from src.training.trainer import carregar_dataset_tripletos, configurar_treinador

__all__ = [
    "ScientificMetricsCallback",
    "construir_modelo_embedding",
    "carregar_dataset_tripletos",
    "configurar_treinador",
]
