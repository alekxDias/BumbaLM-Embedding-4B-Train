"""
Subpacote de avaliacao, auditoria e benchmarks.
"""

from src.evaluation.ir_evaluator import (
    criar_ir_evaluator,
    executar_avaliacao_completa,
    montar_estruturas_ir,
)
from src.evaluation.leakage_audit import (
    auditar_e_marcar_chunks,
    classificar_contaminacao,
    gerar_relatorio_auditoria,
)

__all__ = [
    "classificar_contaminacao",
    "auditar_e_marcar_chunks",
    "gerar_relatorio_auditoria",
    "montar_estruturas_ir",
    "criar_ir_evaluator",
    "executar_avaliacao_completa",
]

