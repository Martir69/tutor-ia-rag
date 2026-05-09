"""
models.py — Entidades del dominio
Define las estructuras de datos puras del sistema.
Sin dependencias externas (Principio D de SOLID)
"""
from dataclasses import dataclass, field


@dataclass
class DocumentChunk:
    """Fragmento de documento indexado con métricas de relevancia."""
    id: str
    text: str
    source: str
    chunk_index: int
    distance: float = 0.0

    @property
    def confidence(self) -> float:
        """Relevancia 0-100%. Convierte distancia coseno (0=idéntico, 2=opuesto)."""
        return round(max(0.0, (1.0 - self.distance / 2.0) * 100), 1)


@dataclass
class Query:
    """Pregunta del estudiante."""
    text: str


@dataclass
class AnswerResponse:
    """Respuesta generada por el tutor con trazabilidad de fuentes."""
    answer: str
    sources: list[str]
    chunks_used: int
    chunk_details: list[dict] = field(default_factory=list)

    def formatted(self) -> str:
        if self.chunk_details:
            bars = "\n".join(
                f"  • {d['source']} — **{d['confidence']:.0f}%**"
                for d in sorted(self.chunk_details, key=lambda x: -x["confidence"])
            )
            footer = (
                f"📊 **Confianza RAG por fuente:**\n{bars}\n"
                f"🔍 **Fragmentos consultados:** {self.chunks_used}"
            )
        else:
            fuentes = ", ".join(self.sources) if self.sources else "ninguna"
            footer = f"📚 **Fuentes:** {fuentes} | 🔍 **Fragmentos:** {self.chunks_used}"
        return f"{self.answer}\n\n---\n{footer}"
