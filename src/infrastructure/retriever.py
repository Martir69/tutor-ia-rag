"""
retriever.py — Búsqueda Semántica y Generación de Respuestas
Responsabilidad ÚNICA: Buscar en ChromaDB → construir prompt → llamar Ollama
(Principio S de SOLID — separado de indexer.py)
"""
import logging
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
import ollama

from src.core.models import DocumentChunk, AnswerResponse

log = logging.getLogger(__name__)

# ── Configuración ──────────────────────────────────────────
CHROMA_DIR  = Path(__file__).parent.parent.parent / "chroma_db"
COLLECTION  = "tutor_ia"
TOP_K       = 4
LLM_MODEL   = "phi3:mini"
MAX_TOKENS  = 512


class RAGRetriever:
    """
    Recupera fragmentos relevantes y genera respuesta con Ollama.
    Intercambiable con otro retriever gracias a IRetriever
    (Principio L de SOLID)
    """

    def __init__(self) -> None:
        self.client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.ef     = embedding_functions.DefaultEmbeddingFunction()
        try:
            self.collection = self.client.get_collection(
                name=COLLECTION,
                embedding_function=self.ef,
            )
            log.info(f" ChromaDB: {self.collection.count()} fragmentos")
        except Exception:
            raise RuntimeError(
                " Base vacía. Ejecuta primero: "
                "python3 -m src.infrastructure.indexer"
            )

    # ── Búsqueda semántica ─────────────────────────────────
    def retrieve(self, query: str) -> list[DocumentChunk]:
        """Encuentra los TOP_K fragmentos más similares a la query."""
        results = self.collection.query(
            query_texts=[query],
            n_results=min(TOP_K, self.collection.count()),
        )
        chunks = []
        for i, text in enumerate(results["documents"][0]):
            chunks.append(DocumentChunk(
                id          = f"chunk_{i}",
                text        = text,
                source      = results["metadatas"][0][i].get("source", "?"),
                chunk_index = i,
                distance    = round(results["distances"][0][i], 4),
            ))
        return chunks

    # ── Construcción del prompt aumentado ──────────────────
    def _build_prompt(self, query: str, chunks: list[DocumentChunk]) -> str:
        context = "\n\n---\n\n".join(
            f"[Fuente: {c.source}]\n{c.text}" for c in chunks
        )
        return f"""Eres un Tutor Inteligente de Inteligencia Artificial.
Responde ÚNICAMENTE basándote en el contexto proporcionado.
Si la información no está en el contexto, dilo claramente.
Responde en español, de forma clara y pedagógica.

### CONTEXTO DEL CURSO:
{context}

### PREGUNTA DEL ESTUDIANTE:
{query}

### RESPUESTA DEL TUTOR:"""

    # ── Generación con Ollama ──────────────────────────────
    def _generate(self, prompt: str) -> str:
        """Llama a Ollama localmente y retorna la respuesta."""
        try:
            response = ollama.generate(
                model   = LLM_MODEL,
                prompt  = prompt,
                options = {
                    "num_predict": MAX_TOKENS,
                    "temperature": 0.3,
                },
            )
            return response["response"].strip()
        except Exception as e:
            return f" Error con Ollama: {e}\nEjecuta: ollama serve"

    # ── Pipeline RAG completo ──────────────────────────────
    def ask(self, query: str) -> AnswerResponse:
        """
        Pipeline completo:
        query → retrieve → prompt → generate → AnswerResponse
        """
        if not query.strip():
            return AnswerResponse(
                answer       = "Por favor escribe una pregunta.",
                sources      = [],
                chunks_used  = 0,
            )

        log.info(f" Query: {query}")
        chunks        = self.retrieve(query)
        prompt        = self._build_prompt(query, chunks)
        answer        = self._generate(prompt)
        sources       = list(dict.fromkeys(c.source for c in chunks))
        chunk_details = [{"source": c.source, "confidence": c.confidence} for c in chunks]
        return AnswerResponse(
            answer        = answer,
            sources       = sources,
            chunks_used   = len(chunks),
            chunk_details = chunk_details,
        )