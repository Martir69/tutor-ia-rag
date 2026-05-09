"""
indexer.py — Ingestión y Vectorización de Documentos
Responsabilidad ÚNICA: Leer docs → fragmentar → guardar en ChromaDB
Soporta PDFs con texto y PDFs escaneados (OCR automático)
(Principio S de SOLID)
"""
import logging
from pathlib import Path
from typing import List, Tuple

import fitz  # PyMuPDF
import chromadb
from chromadb.utils import embedding_functions

log = logging.getLogger(__name__)

# ── Configuración ──────────────────────────────────────────
DOCS_DIR   = Path(__file__).parent.parent.parent / "docs"
CHROMA_DIR = Path(__file__).parent.parent.parent / "chroma_db"
COLLECTION = "tutor_ia"
CHUNK_SIZE = 500
OVERLAP    = 50
OCR_MIN_CHARS = 50  # si un PDF tiene menos de esto, usamos OCR


class DocumentIndexer:
    """
    Lee PDFs (texto + OCR), TXT, MD y PL.
    Detecta automáticamente si un PDF necesita OCR.
    (Principio O de SOLID — extensible sin modificar index())
    """

    def __init__(self) -> None:
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.ef     = embedding_functions.DefaultEmbeddingFunction()
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION,
            embedding_function=self.ef,
        )

    # ── OCR ────────────────────────────────────────────────
    def _pdf_ocr(self, path: Path) -> str:
        """Extrae texto de PDF escaneado usando OCR."""
        try:
            from pdf2image import convert_from_path
            import pytesseract
            log.info(f"🔍 OCR en: {path.name}")
            images = convert_from_path(str(path), dpi=200)
            texts  = []
            for i, img in enumerate(images):
                text = pytesseract.image_to_string(img, lang="spa+eng")
                texts.append(text)
                log.info(f"   Página {i+1}/{len(images)} procesada")
            return "\n".join(texts)
        except Exception as e:
            log.warning(f"OCR falló en {path.name}: {e}")
            return ""

    # ── Parsers ────────────────────────────────────────────
    def _parse_pdf(self, path: Path) -> str:
        """Intenta texto directo; si no alcanza, usa OCR."""
        try:
            doc  = fitz.open(str(path))
            text = "\n".join(page.get_text() for page in doc)
            if len(text.strip()) >= OCR_MIN_CHARS:
                return text
            log.info(f"⚠️  {path.name} parece escaneado, usando OCR...")
            return self._pdf_ocr(path)
        except Exception as e:
            log.warning(f"No se pudo leer {path.name}: {e}")
            return ""

    def _parse_txt(self, path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            log.warning(f"No se pudo leer {path.name}: {e}")
            return ""

    def _parse_pl(self, path: Path) -> str:
        return self._parse_txt(path)

    # ── Carga ──────────────────────────────────────────────
    def _load_documents(self) -> List[Tuple[str, str]]:
        docs    = []
        parsers = {
            ".pdf": self._parse_pdf,
            ".txt": self._parse_txt,
            ".md":  self._parse_txt,
            ".pl":  self._parse_pl,
        }
        for path in sorted(DOCS_DIR.iterdir()):
            parser = parsers.get(path.suffix.lower())
            if not parser:
                continue
            text = parser(path)
            if text.strip():
                docs.append((text, path.name))
                log.info(f"✅ Listo: {path.name} ({len(text):,} chars)")
            else:
                log.warning(f"⚠️  Vacío: {path.name}")
        return docs

    # ── Chunking ───────────────────────────────────────────
    def _chunk(self, text: str, source: str) -> List[dict]:
        chunks, start, idx = [], 0, 0
        while start < len(text):
            end   = min(start + CHUNK_SIZE, len(text))
            piece = text[start:end].strip()
            if piece:
                chunks.append({
                    "id":      f"{source}__c{idx}",
                    "text":    piece,
                    "source":  source,
                    "chunk_n": idx,
                })
                idx += 1
            start += CHUNK_SIZE - OVERLAP
        return chunks

    # ── Pipeline principal ─────────────────────────────────
    def index(self) -> int:
        log.info("🚀 Iniciando indexación...")
        documents = self._load_documents()

        if not documents:
            log.warning("⚠️  No hay documentos en docs/")
            return 0

        if self.collection.count() > 0:
            log.info("🔄 Limpiando índice anterior...")
            self.client.delete_collection(COLLECTION)
            self.collection = self.client.get_or_create_collection(
                name=COLLECTION,
                embedding_function=self.ef,
            )

        all_chunks = []
        for text, source in documents:
            all_chunks.extend(self._chunk(text, source))

        for i in range(0, len(all_chunks), 100):
            batch = all_chunks[i:i + 100]
            self.collection.add(
                ids       = [c["id"]     for c in batch],
                documents = [c["text"]   for c in batch],
                metadatas = [{"source":  c["source"],
                              "chunk_n": c["chunk_n"]} for c in batch],
            )
            log.info(f"📥 Lote {i//100 + 1}: {len(batch)} fragmentos")

        log.info(f"✅ Total indexado: {len(all_chunks)} fragmentos")
        return len(all_chunks)


# ── Ejecución directa ──────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    indexer = DocumentIndexer()
    total   = indexer.index()
    print(f"\n🎉 {total} fragmentos indexados exitosamente.")