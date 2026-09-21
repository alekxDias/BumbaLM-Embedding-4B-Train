"""
Esquema de dados e definicao das estruturas de processo e chunks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional


def normalizar_identificador(texto: str) -> str:
    """
    Remove caracteres nao numericos de um numero de processo CNJ.
    Exemplo: '0800123-45.2020.8.10.0001' -> '08001234520208100001'
    """
    return re.sub(r"\D+", "", str(texto or "").strip())


def normalizar_sigla_tribunal(tribunal: str) -> str:
    """
    Padroniza a sigla do tribunal em letras maiusculas e sem espacos.
    Se nao informado ou vazio, retorna string vazia.
    """
    return str(tribunal or "").strip().upper()


@dataclass
class ProcessoJudicial:
    """
    Representa um processo judicial integro e validado.
    """
    tribunal: str
    numero_processo: str
    ementa: str
    inteiro_teor: str
    metadados: dict[str, Any] = field(default_factory=dict)
    
    # Identificador exclusivo: {tribunal}_{numero} se houver tribunal, senao apenas {numero}
    id_global: str = field(init=False)

    def __post_init__(self):
        self.tribunal = normalizar_sigla_tribunal(self.tribunal)
        num_norm = normalizar_identificador(self.numero_processo)
        if self.tribunal:
            self.id_global = f"{self.tribunal}_{num_norm}" if num_norm else f"{self.tribunal}_sem_numero"
        else:
            self.id_global = num_norm if num_norm else "sem_numero"
        self.ementa = str(self.ementa or "").strip()
        self.inteiro_teor = str(self.inteiro_teor or "").strip()

    @property
    def id(self) -> str:
        return self.id_global

    def to_dict(self) -> dict[str, Any]:
        """Converte o registro para formato serializavel em JSONL."""
        return {
            "id_global": self.id_global,
            "str_tribunal": self.tribunal,
            "str_numero_unico": self.numero_processo,
            "str_ementa": self.ementa,
            "str_inteiro_teor": self.inteiro_teor,
            "metadados": self.metadados,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProcessoJudicial:
        """Instancia ProcessoJudicial a partir de dicionario JSON."""
        return cls(
            tribunal=data.get("str_tribunal", data.get("tribunal", "")),
            numero_processo=data.get("str_numero_unico", data.get("numero_processo", "")),
            ementa=data.get("str_ementa", data.get("ementa", "")),
            inteiro_teor=data.get("str_inteiro_teor", data.get("inteiro_teor", "")),
            metadados=data.get("metadados", {}),
        )


@dataclass
class ChunkProcesso:
    """
    Representa um fragmento (chunk) do inteiro teor de uma decisao.
    """
    chunk_id: str             # Exemplo: 'TJMA_08001234520208100001__0'
    id_global_processo: str   # Exemplo: 'TJMA_08001234520208100001'
    tribunal: str
    numero_processo: str
    chunk_idx: int
    n_tokens: int
    texto: str
    contem_vazamento_ementa: bool = False
    motivo_vazamento: Optional[str] = None

    @property
    def processo_id(self) -> str:
        return self.id_global_processo

    @property
    def is_ementa_leak(self) -> bool:
        return self.contem_vazamento_ementa

    @property
    def motivo_leak(self) -> Optional[str]:
        return self.motivo_vazamento

    def to_dict(self) -> dict[str, Any]:
        """Converte o chunk para formato serializavel em JSONL."""
        return {
            "chunk_id": self.chunk_id,
            "id_global_processo": self.id_global_processo,
            "str_tribunal": self.tribunal,
            "str_numero_unico": self.numero_processo,
            "chunk_idx": self.chunk_idx,
            "n_tokens": self.n_tokens,
            "texto": self.texto,
            "contem_vazamento_ementa": self.contem_vazamento_ementa,
            "motivo_vazamento": self.motivo_vazamento,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ChunkProcesso:
        """Instancia ChunkProcesso a partir de dicionario JSON."""
        return cls(
            chunk_id=data.get("chunk_id", ""),
            id_global_processo=data.get("id_global_processo", data.get("processo_id", "")),
            tribunal=data.get("str_tribunal", data.get("tribunal", "")),
            numero_processo=data.get("str_numero_unico", data.get("numero_processo", "")),
            chunk_idx=int(data.get("chunk_idx", data.get("chunk_index", 0))),
            n_tokens=int(data.get("n_tokens", data.get("num_tokens", 0))),
            texto=data.get("texto", ""),
            contem_vazamento_ementa=bool(data.get("contem_vazamento_ementa", data.get("is_ementa_leak", False))),
            motivo_vazamento=data.get("motivo_vazamento", data.get("motivo_leak")),
        )
