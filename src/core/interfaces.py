"""
interfaces.py — Contratos del sistema (Principio I y D de SOLID)
Define QUÉ debe hacer cada componente, sin decir CÓMO.
"""
from typing import Protocol
from .models import DocumentChunk, AnswerResponse

class IIndexer(Protocol):
    """Contrato para cualquier indexador de documentos."""
    def index(self) -> int: ...

class IRetriever(Protocol):
    """Contrato para cualquier recuperador de información."""
    def retrieve(self, query: str) -> list[DocumentChunk]: ...
    def ask(self, query: str) -> AnswerResponse: ...