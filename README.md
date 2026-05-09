# Tutor Inteligente de IA

> Sistema RAG académico **100% local** para el curso de Inteligencia Artificial  
> Universidad Mariano Gálvez de Guatemala — Ingeniería en Ciencias y Sistemas

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://python.org)
[![Gradio](https://img.shields.io/badge/Gradio-4.44-FF7C00)](https://gradio.app)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)](https://docker.com)
[![Ollama](https://img.shields.io/badge/Ollama-phi3:mini-black)](https://ollama.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5-7C6DFA)](https://trychroma.com)
[![License](https://img.shields.io/badge/License-MIT-22C55E)](LICENSE)
[![Documentación](https://img.shields.io/badge/Documentaci%C3%B3n-Serie%20II-7C6DFA)](DOCUMENTACION.md)

> 📄 **Documentación académica completa (Serie II — stack, arquitectura, SOLID, prompts):** [DOCUMENTACION.md](DOCUMENTACION.md)

---

## ¿Qué es?

Asistente conversacional que responde preguntas del curso de IA basándose **únicamente** en los PDFs del curso.
Cada respuesta muestra el **% de confianza RAG** por fuente, demostrando exactamente qué tan relevante fue cada documento en la respuesta.

Sin OpenAI. Sin nube. Sin internet. Todo corre local.

---

## Un solo comando

```bash
git clone https://github.com/Martir69/tutor-ia-rag.git
cd tutor-ia-rag
docker compose up --build
```

Abre **http://localhost:7860** — listo.

---

## Arquitectura

```
┌─────────────────────────────────────────────────────────────────┐
│                    TUTOR INTELIGENTE DE IA                      │
│                                                                  │
│  ┌──────────────┐    ┌─────────────────────────────────────┐   │
│  │  src/core/   │    │        src/infrastructure/          │   │
│  │              │    │                                     │   │
│  │  models.py   │◄───│  indexer.py          retriever.py  │   │
│  │  (Entidades) │    │  PDF→ChromaDB        DB→Ollama→Resp │   │
│  │              │    │                                     │   │
│  │ interfaces.py│◄───│  IIndexer            IRetriever    │   │
│  │  (Contratos) │    │  (protocolo)         (protocolo)    │   │
│  └──────────────┘    └──────────────┬──────────────────────┘   │
│         ▲                           │                           │
│         │            ┌──────────────▼──────────────────────┐   │
│         └────────────│       src/interface/app.py          │   │
│                      │       Gradio UI (puerto 7860)       │   │
│                      └─────────────────────────────────────┘   │
│                                                                  │
│  ┌─────────────────┐    ┌─────────────────────────────────┐    │
│  │  ChromaDB local │    │     Ollama local (phi3:mini)    │    │
│  │  (vectores)     │    │     LLM 3.8B, sin GPU          │    │
│  └─────────────────┘    └─────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

**Flujo de una pregunta:**
```
Pregunta → ChromaDB (búsqueda semántica) → TOP-4 chunks relevantes
        → Prompt aumentado → Ollama phi3:mini → Respuesta + Confianza RAG
```

---

## Stack tecnológico

| Tecnología | Versión | Rol |
|---|---|---|
| Python | 3.12 | Lenguaje base |
| ChromaDB | 0.5.23 | Vector store local |
| Ollama | 0.3.3 | LLM runner sin GPU |
| phi3:mini | 3.8B Q4 | Modelo LLM |
| PyMuPDF | 1.24.11 | PDF parser + fallback OCR |
| Gradio | 4.44.1 | Web UI |
| Docker | 29+ | Deploy un-comando |

> Las justificaciones técnicas de cada elección están en [DOCUMENTACION.md → Apartado 1](DOCUMENTACION.md#apartado-1--stack-tecnológico-15-pts).

---

## Elemento creativo — Confianza RAG

Cada respuesta muestra el **% de relevancia** de cada fuente consultada:

```
📊 Confianza RAG por fuente:
  • AI-Semana7.pdf — 87%
  • Inteligencia_Artificial_Fundamentos_y_Agentes.pdf — 72%
🔍 Fragmentos consultados: 4
```

La confianza convierte la distancia coseno de ChromaDB a porcentaje:

```python
confidence = max(0, (1 - distance / 2) * 100)  # 0=idéntico → 100%, 2=opuesto → 0%
```

---

## Estructura del proyecto

```
tutor-ia-rag/
├── docs/                       ← PDFs del curso (material indexado)
├── src/
│   ├── core/
│   │   ├── models.py           ← DocumentChunk, Query, AnswerResponse
│   │   └── interfaces.py       ← IIndexer, IRetriever (protocolos)
│   ├── infrastructure/
│   │   ├── indexer.py          ← PDFs → ChromaDB (con OCR automático)
│   │   └── retriever.py        ← ChromaDB → Ollama → AnswerResponse
│   └── interface/
│       └── app.py              ← Gradio UI
├── scripts/
│   ├── entrypoint.sh           ← Espera Ollama → indexa → lanza UI
│   └── setup.sh                ← Setup local sin Docker
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

---

## Opciones de ejecución

### Opción 1 — Docker (recomendado para evaluación)

**Requisitos:** [Docker Desktop](https://www.docker.com/products/docker-desktop/) instalado y corriendo.  
**Puerto necesario libre:** `7860` (UI). Si algo ya lo usa: `GRADIO_SERVER_PORT=7861 docker compose up --build`

```bash
docker compose up --build
```

El contenedor hace todo automáticamente: espera Ollama, descarga phi3:mini si no existe, indexa los docs, lanza la UI.

### Opción 2 — Local con venv

```bash
# Requisitos: Python 3.12, Ollama corriendo, tesseract y poppler instalados
pip install -r requirements.txt
python3 -m src.infrastructure.indexer   # indexar una vez
python3 -m src.interface.app            # lanzar UI
```

---

## Agregar nuevos documentos

```bash
cp nuevo_material.pdf docs/
# Luego clic en "Indexar Documentos" en la UI, o:
python3 -m src.infrastructure.indexer
```

El indexer detecta automáticamente si el PDF está escaneado y aplica OCR.

---

## Preguntas de demostración

1. *"¿Qué es el algoritmo A* y qué garantiza sobre el costo de la solución?"*
2. *"¿Cuál es la diferencia entre un agente reactivo y un agente deliberativo BDI?"*
3. *"¿Cuánto cuesta una GPU H100?"* — demuestra que el sistema dice "no está en el contexto"

---

## Autor

**Martir Alexander Vasquez**  
📧 mvasquezc19@miumg.edu.gt  
🎓 Universidad Mariano Gálvez de Guatemala  
💻 Ingeniería en Ciencias y Sistemas — Laboratorio de IA  
🐙 [github.com/Martir69/tutor-ia-rag](https://github.com/Martir69/tutor-ia-rag)
