"""
Subpacote de avaliacao, auditoria e benchmarks.
"""

from src.evaluation.leakage_audit import (
    classificar_contaminacao,
    auditar_e_marcar_chunks,
    gerar_relatorio_auditoria,
)

__all__ = [
    "classificar_contaminacao",
    "auditar_e_marcar_chunks",
    "gerar_relatorio_auditoria",
]
