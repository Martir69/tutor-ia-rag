# Serie II — Documentación Técnica: Tutor Inteligente de IA

**Autor:** Martir Alexander Vasquez — mvasquezc19@miumg.edu.gt  
**Universidad:** Mariano Gálvez de Guatemala  
**Curso:** Laboratorio de Inteligencia Artificial  
**Repositorio:** https://github.com/Martir69/tutor-ia-rag

---

## Apartado 1 — Stack Tecnológico (1.5 pts)

### Decisión de diseño: implementación directa sin orquestadores

El sistema **no utiliza LangChain ni LlamaIndex** de forma deliberada. Ambos frameworks introducen abstracciones que, para un MVP de esta escala, añaden complejidad sin beneficio real: dependencias transitivas inestables, debugging opaco y acoplamiento a versiones específicas. En cambio, se construyó el pipeline RAG directamente sobre las APIs nativas de ChromaDB y Ollama, logrando el mismo resultado con código auditable y sin capas intermedias.

| Capa | Herramienta Elegida | Alternativa Descartada | Justificación técnica |
|---|---|---|---|
| **Lenguaje base** | Python 3.12 | Python 3.10/3.11 | Soporte nativo de `dataclass`, `Protocol`, `match` y tipado mejorado. Versión LTS activa. |
| **Pipeline RAG** | Implementación directa (sin framework) | LangChain / LlamaIndex | LangChain añade ~40 dependencias transitivas y abstracciones que complican el debugging. Para un pipeline de 3 pasos (retrieve → prompt → generate), es overhead puro. |
| **Modelo de embeddings** | `all-MiniLM-L6-v2` vía ChromaDB `DefaultEmbeddingFunction` | `text-embedding-3-small` (OpenAI) | Corre 100% local, sin costo de API, latencia <50ms, 384 dimensiones suficientes para corpus académico de ~150k tokens. |
| **Vector Store** | ChromaDB 0.5.23 (local, persistente) | Pinecone (nube) | Pinecone requiere cuenta y API key. ChromaDB persiste en disco con SQLite + HNSW, cero configuración, ideal para entorno académico sin internet. |
| **LLM generativo** | `phi3:mini` 3.8B Q4 vía Ollama | GPT-4o / Mistral API | 100% local (GDPR, sin costo), 2.2GB RAM, viable en CPU. Para responder preguntas del temario con contexto RAG, 3.8B parámetros es suficiente. GPT-4o tiene costo por token y requiere internet. |
| **Parseo de documentos** | PyMuPDF 1.24.11 + pytesseract (fallback OCR) | Unstructured | PyMuPDF es más liviano y maduro. `Unstructured` añade 15+ dependencias. El fallback OCR manual con `pdf2image` + `tesseract` da control total sobre el proceso. |
| **Interfaz web** | Gradio 4.44.1 | FastAPI + React / Streamlit | Gradio produce UI de chat funcional en ~50 líneas. Para un tutor académico, la prioridad es la demo rápida sobre el control de UI. Streamlit descartado por ausencia de componente chat nativo. |

### Cómo se integran

```
docs/ (PDFs) → PyMuPDF/OCR → chunks de texto
                                    ↓
                          all-MiniLM-L6-v2 (embeddings)
                                    ↓
                          ChromaDB (persistencia HNSW)
                                    ↓ (query semántica)
                          TOP-4 chunks relevantes
                                    ↓
                          Prompt aumentado (RAG)
                                    ↓
                          phi3:mini vía Ollama
                                    ↓
                          Respuesta + confianza RAG → Gradio UI
```

---

## Apartado 2 — Fase sdd-init: Ingestión del Repositorio (1.5 pts)

### Formatos ingeridos y parsers

| Formato | Parser | Lógica |
|---|---|---|
| `.pdf` (con texto digital) | `PyMuPDF` (`fitz.open`) | Extracción directa de texto por página |
| `.pdf` (escaneado / imagen) | `pdf2image` + `pytesseract` (OCR) | Detecta automáticamente si `len(text) < 50 chars` y aplica OCR |
| `.txt` | `Path.read_text(encoding="utf-8")` | Lectura directa |
| `.md` | Mismo que `.txt` | Texto plano |
| `.pl` (Prolog) | Mismo que `.txt` | Reglas Prolog tratadas como texto estructurado |

La detección de PDFs escaneados es automática: si `PyMuPDF` extrae menos de 50 caracteres de una página, el sistema infiere que es una imagen y activa OCR con soporte de idioma `spa+eng`.

### Estrategia de chunking

```python
CHUNK_SIZE = 500   # caracteres por fragmento
OVERLAP    = 50    # solapamiento entre fragmentos consecutivos
```

**Justificación del tamaño:** 500 caracteres equivale a ~3-4 oraciones. Fragmentos más grandes diluyen la especificidad semántica del embedding; más pequeños pierden contexto gramatical. El solapamiento de 50 evita que conceptos que caen en el límite de dos fragmentos se pierdan en la recuperación.

### Modelo de embeddings

`all-MiniLM-L6-v2` (384 dimensiones) incluido en `chromadb.utils.embedding_functions.DefaultEmbeddingFunction`. Corre localmente, sin latencia de red, y tiene rendimiento documentado en benchmarks BEIR para corpus técnicos en inglés y español.

### Vector store

ChromaDB con índice HNSW (Hierarchical Navigable Small World). Persiste en `chroma_db/` mediante SQLite como metadata store y archivos binarios como índice vectorial. La colección se llama `tutor_ia`.

### Prompt para el modelo de exploración inicial (sdd-init)

```markdown
# sdd-init: Exploración del Repositorio del Curso

Eres un agente de análisis de conocimiento académico con ventana de contexto extendida.
Se te proporciona el contenido completo de los documentos del curso de Inteligencia
Artificial (Guía Didáctica, reglas Prolog, tablas de verdad, PDFs de unidades).

## Tu tarea
1. Identifica los TEMAS PRINCIPALES cubiertos en los documentos.
2. Detecta CONCEPTOS TÉCNICOS que requieren definición precisa (algoritmos, fórmulas, estructuras).
3. Identifica RELACIONES ENTRE CONCEPTOS (p. ej. A* es una extensión de BFS con heurística).
4. Señala GAPS o zonas donde el material es ambiguo o incompleto.
5. Propone una TAXONOMÍA de temas para organizar el índice del tutor.

## Documentos proporcionados
[CONTENIDO COMPLETO DE LOS PDFs — hasta 128k tokens]

## Formato de salida esperado
### Temas identificados
- Tema: <nombre> | Documentos fuente: <lista> | Densidad de cobertura: Alta/Media/Baja

### Conceptos clave
- Concepto: <nombre> | Definición encontrada: <sí/no> | Relaciones con: <otros conceptos>

### Recomendaciones para chunking
- <observaciones sobre granularidad óptima por tipo de contenido>
```

### Por qué se necesita un modelo de gran ventana de contexto en sdd-init

La fase `sdd-init` es de **exploración y comprensión global**: el agente debe leer TODO el corpus antes de proponer cómo estructurarlo. Si el modelo tiene una ventana de 4k–8k tokens, no puede ver más de 2-3 documentos a la vez y pierde relaciones entre conceptos distribuidos en distintos PDFs.

Un modelo como `kimi-k2.5` con 128k tokens puede ingerir los 9 PDFs del curso (~185MB, ~150k tokens de texto) **en un solo prompt**, lo que permite:

1. Detectar que el concepto "agente BDI" aparece en tres documentos con definiciones complementarias (no redundantes).
2. Identificar que las tablas de verdad en formato Prolog en `tablas.pl` corresponden a los ejemplos del capítulo 4 del PDF.
3. Proponer un chunking diferenciado por tipo (párrafos para teoría, funciones completas para código Prolog).

En la **fase de inferencia** (responder preguntas), el contexto ya está comprimido en el vector store, por lo que phi3:mini con 4k tokens es suficiente. La ventana grande es crítica solo en la exploración inicial.

---

## Apartado 3 — Fases sdd-propose & sdd-spec (2.0 pts)

### 3A — Arquitectura Limpia implementada

```
┌─────────────────────────────────────────────────────────────┐
│  CAPA DOMINIO  (src/core/)                                  │
│  Sin dependencias externas — entidades y contratos puros    │
│                                                             │
│  models.py     → DocumentChunk, Query, AnswerResponse       │
│  interfaces.py → Protocol IIndexer, Protocol IRetriever    │
└─────────────────────────────┬───────────────────────────────┘
                              │ depende de
┌─────────────────────────────▼───────────────────────────────┐
│  CAPA INFRAESTRUCTURA  (src/infrastructure/)                │
│  Implementaciones concretas que satisfacen los protocolos   │
│                                                             │
│  indexer.py   → DocumentIndexer implements IIndexer         │
│                 (PyMuPDF + OCR + ChromaDB)                  │
│  retriever.py → RAGRetriever implements IRetriever          │
│                 (ChromaDB query + Ollama generate)          │
└─────────────────────────────┬───────────────────────────────┘
                              │ depende de
┌─────────────────────────────▼───────────────────────────────┐
│  CAPA INTERFAZ  (src/interface/)                            │
│  Solo UI — no sabe que ChromaDB existe                      │
│                                                             │
│  app.py → Gradio Blocks                                     │
│           Importa IRetriever, nunca RAGRetriever directamente│
└─────────────────────────────────────────────────────────────┘
```

**Regla de dependencias:** las flechas apuntan siempre hacia el centro (dominio). La capa de infraestructura implementa los contratos del dominio; la interfaz solo habla con el dominio a través de los protocolos.

### 3B — Prompt para sdd-spec

```markdown
# Spec Request: Tutor Inteligente de IA — RAG MVP

## Contexto
Sistema conversacional académico que responde preguntas del curso de Inteligencia
Artificial de la Universidad Mariano Gálvez, basándose exclusivamente en los PDFs
del curso. Opera 100% local (sin APIs externas). Stack: Python 3.12, ChromaDB,
Ollama phi3:mini, Gradio.

## Requisitos Funcionales

- RF-01: El sistema debe indexar documentos en formato PDF, TXT, MD y PL desde
  el directorio `docs/` y almacenar sus embeddings en ChromaDB.
- RF-02: El sistema debe detectar automáticamente PDFs escaneados y aplicar OCR.
- RF-03: El sistema debe responder preguntas en lenguaje natural consultando
  exclusivamente el corpus indexado.
- RF-04: Cada respuesta debe incluir las fuentes consultadas y el porcentaje de
  confianza RAG por fuente.
- RF-05: La UI debe permitir re-indexar documentos sin reiniciar el servidor.
- RF-06: El sistema debe negarse a responder con información que no esté en el
  corpus (no alucinar fuera del contexto).

## Criterios de Aceptación

**Escenario 1 — Pregunta cubierta por el material:**
Dado que el corpus contiene el documento `AI-Semana7.pdf` con contenido sobre A*,
Cuando el estudiante pregunta "¿Qué es el algoritmo A*?",
Entonces el sistema responde con una explicación coherente, cita `AI-Semana7.pdf`
como fuente y muestra un porcentaje de confianza ≥ 50%.

**Escenario 2 — Pregunta fuera del corpus:**
Dado que ningún documento del corpus menciona el precio de hardware,
Cuando el estudiante pregunta "¿Cuánto cuesta una GPU H100?",
Entonces el sistema responde indicando explícitamente que la información
no se encuentra en el material del curso.

**Escenario 3 — Indexación exitosa:**
Dado que existen 9 documentos PDF en `docs/`,
Cuando el usuario ejecuta `python3 -m src.infrastructure.indexer`,
Entonces el sistema reporta el número de fragmentos indexados y
ChromaDB persiste la colección `tutor_ia` en disco.

**Escenario 4 — Re-indexación desde UI:**
Dado que el usuario agrega un nuevo PDF a `docs/`,
Cuando hace clic en "Indexar Documentos" en la UI,
Entonces el sistema re-indexa todos los documentos y actualiza el contador
de fragmentos en el panel de estado sin reiniciar el servidor.

## Restricciones No Funcionales

- Latencia máxima de respuesta: < 120 segundos en CPU sin GPU (phi3:mini en CPU).
- Precisión mínima del RAG: el sistema debe recuperar al menos 1 fragmento
  con distancia coseno < 1.0 para preguntas directas del temario.
- Privacidad: ninguna consulta o documento sale de la máquina del usuario.
- Portabilidad: debe ejecutarse con `docker compose up --build` en cualquier
  máquina con Docker instalado.
```

### Justificación del modelo para sdd-spec

`devstral-medium` (Mistral) es adecuado para esta fase porque está entrenado específicamente en código y especificaciones técnicas. La generación de criterios de aceptación en formato Given/When/Then requiere comprensión precisa de contratos funcionales, no creatividad narrativa. Un modelo de propósito general puede generar criterios vagos o inconsistentes con la arquitectura propuesta; un modelo especializado en desarrollo de software produce criterios directamente ejecutables como tests.

---

## Apartado 4 — Fase sdd-apply: Generación del Pipeline RAG (2.0 pts)

### 4A — Prompt para sdd-apply

```markdown
## sdd-apply Prompt: RAG Indexer + Retriever

Eres un agente de implementación. Genera código Python puro para el pipeline
RAG de un tutor académico. Arquitectura limpia obligatoria: las clases deben
implementar los protocolos IIndexer e IRetriever definidos en src/core/interfaces.py.

### Módulo 1: src/infrastructure/indexer.py

Implementa la clase `DocumentIndexer(IIndexer)`:

- Lee documentos desde: `Path(__file__).parent.parent.parent / "docs"`
- Formatos soportados: `.pdf`, `.txt`, `.md`, `.pl`
- Para PDFs: intenta extracción con PyMuPDF; si el texto extraído tiene
  menos de 50 caracteres, aplica OCR con pdf2image + pytesseract (lang="spa+eng")
- Estrategia de chunking: chunk_size=500 caracteres, overlap=50 caracteres,
  fragmentación por ventana deslizante
- Modelo de embeddings: chromadb.utils.embedding_functions.DefaultEmbeddingFunction()
  (all-MiniLM-L6-v2, local, sin API)
- Persiste en ChromaDB colección `tutor_ia`, directorio `chroma_db/`
- Si la colección ya existe, elimínala y re-indexa (re-index completo)
- Inserción en lotes de 100 fragmentos para eficiencia de memoria

### Módulo 2: src/infrastructure/retriever.py

Implementa la clase `RAGRetriever(IRetriever)`:

- Al inicializarse, carga la colección `tutor_ia` de ChromaDB;
  si no existe, lanza RuntimeError con instrucción de indexar primero
- `retrieve(query: str) -> list[DocumentChunk]`: consulta ChromaDB con
  `n_results=4`, retorna DocumentChunk con campo `distance` (distancia coseno)
- `_build_prompt(query, chunks)`: construye prompt RAG en español con el
  contexto recuperado y la instrucción de responder SOLO desde el contexto
- `_generate(prompt)`: llama a `ollama.generate(model="phi3:mini",
  options={"num_predict": 512, "temperature": 0.3})`
- `ask(query: str) -> AnswerResponse`: pipeline completo;
  incluye `chunk_details` con source y confidence por fragmento
- confidence se calcula como: `max(0, (1 - distance / 2) * 100)`

### Restricciones obligatorias
- Sin LangChain ni LlamaIndex — solo ChromaDB y Ollama directamente
- Type hints en todos los métodos públicos
- Manejo de errores con try/except en parseo de PDFs y llamada a Ollama
- Logging con `logging.getLogger(__name__)`, nivel INFO
- El método `index()` retorna `int` (total de fragmentos indexados)
- El método `ask()` nunca lanza excepción al usuario final;
  errores de Ollama se retornan como texto en `AnswerResponse.answer`
```

### 4B — Justificación del modelo y decisiones técnicas

**Por qué `codestral-latest` (o equivalente) para sdd-apply:**

La fase de implementación requiere que el modelo genere código correcto en la primera pasada, siguiendo protocolos definidos, con imports precisos y tipos coherentes. `codestral-latest` está entrenado específicamente en código Python de producción, conoce la API de ChromaDB y puede respetar una arquitectura prescrita sin "inventar" abstracciones innecesarias. Un modelo de propósito general tiende a introducir LangChain o wrappers adicionales aunque el prompt diga lo contrario.

**Decisiones técnicas justificadas en el código real:**

| Decisión | Valor | Por qué |
|---|---|---|
| `chunk_size` | 500 chars | ~3 oraciones: granularidad semántica sin perder contexto gramatical |
| `overlap` | 50 chars | Evita conceptos partidos en el límite de dos fragmentos |
| `TOP_K` | 4 fragmentos | Balance entre contexto suficiente (4 × 500 = 2000 chars) y ventana de phi3:mini (4096 tokens) |
| `temperature` | 0.3 | Respuestas deterministas y factuales; se penaliza la creatividad en un tutor académico |
| `num_predict` | 512 tokens | Respuestas concisas; previene que el modelo divague fuera del contexto |
| Distancia coseno → % | `(1 - d/2) × 100` | Distancia coseno en ChromaDB ∈ [0, 2]; fórmula normaliza a [0%, 100%] |

---

## Apartado 5 — Fase sdd-verify: Auditoría del Código (1.5 pts)

### Prompt para sdd-verify

```markdown
## sdd-verify Prompt: Senior Code Reviewer — Sistema RAG Académico

Eres un revisor senior de sistemas de IA con expertise en RAG, seguridad de LLMs
y arquitectura limpia en Python. Audita el código de `indexer.py` y `retriever.py`
con máximo rigor técnico.

### Checklist de Revisión

**1. Calidad del código**
   - ¿Todos los métodos públicos tienen type hints completos?
   - ¿Se siguen las convenciones PEP 8 (nombres, espaciado, longitud de línea)?
   - ¿El manejo de errores con try/except es específico (no `except Exception` desnudo)?
   - ¿Los logs son informativos y no incluyen datos sensibles?

**2. Alucinaciones del RAG**
   - ¿El prompt del retriever instruye EXPLÍCITAMENTE al LLM a responder solo
     desde el contexto proporcionado?
   - ¿Existe una instrucción de fallback cuando el contexto no contiene la respuesta?
   - ¿Se valida que `chunks` no esté vacío antes de construir el prompt?
   - ¿La respuesta del LLM puede incluir información que no esté en los fragmentos
     recuperados? Identifica la línea exacta donde esto podría ocurrir.

**3. Prompt Injection**
   - ¿Puede un usuario malicioso enviar una query que sobrescriba las instrucciones
     del sistema? Por ejemplo: "Ignora las instrucciones anteriores y..."
   - ¿El texto de los documentos indexados podría contener instrucciones maliciosas
     que el LLM ejecute? (indirect prompt injection)
   - ¿Se sanitiza o se limita la longitud de la query del usuario antes de
     incorporarla al prompt?

**4. Principios SOLID**
   - ¿`indexer.py` tiene exactamente UNA responsabilidad? ¿O mezcla parseo,
     chunking y persistencia en una sola clase?
   - ¿`retriever.py` depende de una abstracción (IRetriever) o de la
     implementación concreta de ChromaDB?
   - ¿Agregar un nuevo formato de documento (p. ej. .docx) requiere modificar
     `index()` o solo agregar un método `_parse_docx()`?

**5. Separación indexer / retriever**
   - ¿`retriever.py` importa algo de `indexer.py`? (si sí, es un acoplamiento indebido)
   - ¿Pueden ejecutarse `indexer.py` y `retriever.py` de forma completamente
     independiente?
   - ¿La colección de ChromaDB es el único punto de comunicación entre ambos?

### Para cada hallazgo reporta

- **Severidad:** CRÍTICO / ADVERTENCIA / SUGERENCIA
- **Archivo y línea afectada**
- **Descripción del problema**
- **Corrección propuesta con ejemplo de código**

### Ejemplo de hallazgo esperado (formato)

```
ADVERTENCIA — retriever.py:70
Problema: El prompt no limita la longitud de la query del usuario. Una query
de 10,000 caracteres podría superar la ventana de contexto de phi3:mini.
Corrección:
  query = query.strip()[:500]  # limitar antes de construir el prompt
```
```

### Justificación del modelo de razonamiento para sdd-verify

La auditoría de seguridad en sistemas RAG requiere razonamiento de múltiples pasos: el revisor debe (1) leer el código, (2) simular el comportamiento del LLM con distintas inputs maliciosas, (3) razonar sobre flujos de ejecución indirectos (p. ej. un PDF que contiene instrucciones para el LLM). Un modelo como `claude-3-7-sonnet-thinking` o `gpt-o3` con cadena de pensamiento visible puede:

- Construir el flujo de datos mentalmente y detectar dónde una inyección puede "escapar"
- Generar casos de prueba adversariales que un modelo sin thinking descartaría
- Distinguir entre un bug de seguridad real y un falso positivo

Los modelos sin capacidad de razonamiento extendido tienden a reportar hallazgos superficiales (PEP 8, nombres de variables) y pasan por alto vulnerabilidades estructurales como el indirect prompt injection.

---

## Apartado 6 — Buenas Prácticas: SOLID (1.5 pts)

### Principio S — Single Responsibility

**Regla:** cada módulo tiene exactamente una razón para cambiar.

| Módulo | Única responsabilidad | Si cambia... |
|---|---|---|
| `indexer.py` | Convertir documentos en embeddings almacenados | Solo cambia si cambia el vector store o la estrategia de chunking |
| `retriever.py` | Convertir una query en una respuesta usando el LLM | Solo cambia si cambia el modelo LLM o la lógica de recuperación |
| `app.py` | Exponer el sistema al usuario | Solo cambia si cambia el framework de UI |
| `models.py` | Definir las entidades del dominio | Solo cambia si cambia la estructura de los datos |

**Ejemplo concreto:** `indexer.py` nunca llama a Ollama. `retriever.py` nunca lee un PDF. Si mañana se reemplaza ChromaDB por Pinecone, solo se modifica `retriever.py`, no `app.py`.

---

### Principio O — Open / Closed

**Regla:** abierto a extensión, cerrado a modificación.

El método `_load_documents()` en `indexer.py` usa un diccionario de parsers:

```python
parsers = {
    ".pdf": self._parse_pdf,
    ".txt": self._parse_txt,
    ".md":  self._parse_txt,
    ".pl":  self._parse_pl,
}
```

Para agregar soporte a `.docx`, se añade `_parse_docx()` y se registra en el diccionario:

```python
".docx": self._parse_docx,
```

El método `index()` no se toca. La extensión no modifica el flujo existente.

---

### Principio L — Liskov Substitution

**Regla:** cualquier implementación de un protocolo debe poder reemplazar a otra sin romper el sistema.

`RAGRetriever` implementa `IRetriever`. Si se creara `PineconeRetriever` que también implementa `IRetriever`, podría usarse en `app.py` sin ningún cambio:

```python
# app.py — hoy
from src.infrastructure.retriever import RAGRetriever
_retriever = RAGRetriever()

# app.py — mañana con Pinecone, misma interfaz
from src.infrastructure.pinecone_retriever import PineconeRetriever
_retriever = PineconeRetriever()
```

`app.py` llama `retriever.ask(query)` en ambos casos. El contrato de `IRetriever` garantiza que el tipo de retorno (`AnswerResponse`) es idéntico.

---

### Principio I — Interface Segregation

**Regla:** los clientes no deben depender de métodos que no usan.

Se definen dos protocolos separados en `interfaces.py`:

```python
class IIndexer(Protocol):
    def index(self) -> int: ...         # solo indexación

class IRetriever(Protocol):
    def retrieve(self, query: str) -> list[DocumentChunk]: ...
    def ask(self, query: str) -> AnswerResponse: ...    # solo consulta
```

No existe un `IDatabaseManager` que mezcle ambas responsabilidades. `app.py` depende de `IRetriever` y nunca ve `IIndexer`. El script de indexación (`indexer.py`) no conoce nada de `IRetriever`.

---

### Principio D — Dependency Inversion

**Regla:** los módulos de alto nivel dependen de abstracciones, no de implementaciones.

`app.py` (módulo de alto nivel) importa el protocolo, no la clase concreta:

```python
# Dependencia correcta (abstracción)
from src.core.interfaces import IRetriever

def get_retriever() -> IRetriever | None:
    return RAGRetriever()   # la construcción concreta ocurre aquí, aislada
```

`app.py` nunca importa `chromadb`, `ollama` ni `fitz`. Esos son detalles de infraestructura que pertenecen a la capa de implementación concreta.

**Beneficio demostrable:** si Ollama cambia su API en una versión futura, el único archivo que se modifica es `retriever.py`. `app.py`, `models.py` e `interfaces.py` permanecen intactos.

---

### Resumen SOLID — tabla de ejemplos del código real

| Principio | Aplicación en el Tutor RAG | Ejemplo concreto en el código |
|---|---|---|
| **S** | `indexer.py` solo indexa; `retriever.py` solo consulta | `retriever.py` no importa `fitz`; `indexer.py` no importa `ollama` |
| **O** | Agregar `.docx` sin tocar `index()` | Diccionario `parsers` en `_load_documents()` — solo agregar entrada |
| **L** | `PineconeRetriever` reemplazaría `RAGRetriever` sin cambiar `app.py` | Ambas implementan `IRetriever`; `app.py` llama `.ask()` en cualquiera |
| **I** | `IIndexer` e `IRetriever` son protocolos independientes | `app.py` importa solo `IRetriever`; nunca ve `IIndexer` |
| **D** | `app.py` nunca importa `chromadb` ni `ollama` | `from src.core.interfaces import IRetriever` en la capa de interfaz |
