"""
app.py — Interfaz Web del Tutor Inteligente
Responsabilidad ÚNICA: Exponer el sistema al usuario via Gradio
(Principio S de SOLID)
"""
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import gradio as gr

from src.infrastructure.indexer import DocumentIndexer
from src.infrastructure.retriever import RAGRetriever

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

DOCS_DIR = Path(__file__).parent.parent.parent / "docs"
_retriever: RAGRetriever | None = None


def get_retriever() -> RAGRetriever | None:
    global _retriever
    if _retriever is None:
        try:
            _retriever = RAGRetriever()
        except RuntimeError:
            return None
    return _retriever


def run_indexing() -> str:
    global _retriever
    try:
        indexer    = DocumentIndexer()
        total      = indexer.index()
        _retriever = None
        if total == 0:
            return "⚠️ No se encontraron documentos en docs/"
        return f"✅ {total} fragmentos indexados correctamente"
    except Exception as e:
        return f"❌ Error: {e}"


def get_status() -> str:
    n_docs = len([
        f for f in DOCS_DIR.iterdir()
        if f.suffix in (".pdf", ".txt", ".md", ".pl")
    ]) if DOCS_DIR.exists() else 0
    retriever = get_retriever()
    n_chunks  = retriever.collection.count() if retriever else 0
    estado    = "✅ Listo" if retriever else "⚠️ Sin indexar"
    lines = [
        f"📁 Documentos del curso: **{n_docs}**",
        f"🗄️  Fragmentos indexados: **{n_chunks}**",
        f"🤖 Modelo LLM: **phi3:mini** (Ollama local)",
        f"🔒 Modo: **100% Local — Sin internet**",
        f"⚡ Estado: **{estado}**",
    ]
    return "\n\n".join(lines)


def chat(message: str, history: list) -> tuple[str, list]:
    if not message.strip():
        return "", history
    retriever = get_retriever()
    if retriever is None:
        reply = "⚠️ Primero haz clic en **⚡ Indexar Documentos**"
    else:
        result = retriever.ask(message)
        reply  = result.formatted()
    history.append((message, reply))
    return "", history


CSS = """
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg:       #080810;
    --surface:  #12121f;
    --surface2: #1a1a2e;
    --border:   #2a2a45;
    --primary:  #7c6dfa;
    --accent:   #00d4aa;
    --text:     #e8e8f0;
    --muted:    #8888aa;
}

body, .gradio-container {
    background: var(--bg) !important;
    font-family: 'Syne', sans-serif !important;
    color: var(--text) !important;
}

#header {
    text-align: center;
    padding: 2.5rem 0 1.5rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.5rem;
}
#header h1 {
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, var(--primary), var(--accent));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
    letter-spacing: -1px;
}
#header p {
    color: var(--muted);
    margin-top: 0.5rem;
    font-size: 0.95rem;
    font-family: 'JetBrains Mono', monospace;
}

.gr-button-primary {
    background: linear-gradient(135deg, var(--primary), var(--accent)) !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
    color: white !important;
}
.gr-button-primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(124,109,250,0.4) !important;
}
.gr-button-secondary {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important;
}

.chatbot {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 16px !important;
}

input, textarea {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important;
}
input:focus, textarea:focus {
    border-color: var(--primary) !important;
    box-shadow: 0 0 0 3px rgba(124,109,250,0.2) !important;
}

.gr-panel, .gr-box, .gr-group {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 16px !important;
}

#status {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.85rem !important;
    background: var(--surface2) !important;
    border-left: 3px solid var(--accent) !important;
    padding: 1rem !important;
    border-radius: 8px !important;
}
"""

EJEMPLOS = [
    "¿Qué es el algoritmo A* y cómo funciona?",
    "Explica la diferencia entre BFS y DFS",
    "¿Qué es el aprendizaje supervisado?",
    "¿Cómo funciona la retropropagación?",
    "¿Qué es la Lógica de Predicados?",
    "¿Qué es un agente BDI?",
]

with gr.Blocks(css=CSS, title="Tutor IA — UMG") as demo:

    gr.HTML("""
    <div id="header">
        <h1>🎓 Tutor Inteligente de IA</h1>
        <p>Universidad Mariano Gálvez · RAG + Ollama · 100% Local</p>
    </div>
    """)

    with gr.Row():

        with gr.Column(scale=1, min_width=280):
            gr.Markdown("### ⚙️ Sistema")
            status_md = gr.Markdown(
                value=get_status(),
                elem_id="status",
            )
            gr.Button("🔄 Actualizar estado", variant="secondary", size="sm").click(
                fn=get_status,
                outputs=status_md,
            )

            gr.Markdown("### 📚 Conocimiento")
            btn_index    = gr.Button("⚡ Indexar Documentos", variant="primary")
            index_result = gr.Textbox(
                label="",
                interactive=False,
                lines=2,
                placeholder="Resultado aquí...",
            )
            btn_index.click(
                fn=run_indexing,
                outputs=index_result,
            ).then(
                fn=get_status,
                outputs=status_md,
            )

            gr.Markdown("### 💡 Ejemplos")
            btn_ejemplos = [
                gr.Button(e, size="sm", variant="secondary")
                for e in EJEMPLOS
            ]

        with gr.Column(scale=3):
            gr.Markdown("### 💬 Pregúntale al Tutor")
            chatbot = gr.Chatbot(
                value=[],
                height=520,
                show_label=False,
                bubble_full_width=False,
            )
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Escribe tu pregunta sobre IA...",
                    show_label=False,
                    scale=5,
                )
                send_btn = gr.Button("Enviar ➤", variant="primary", scale=1)

            send_btn.click(
                fn=chat,
                inputs=[msg, chatbot],
                outputs=[msg, chatbot],
            )
            msg.submit(
                fn=chat,
                inputs=[msg, chatbot],
                outputs=[msg, chatbot],
            )

            gr.Button("🗑️ Limpiar chat", variant="secondary", size="sm").click(
                fn=lambda: [],
                outputs=chatbot,
            )

    for btn, ejemplo in zip(btn_ejemplos, EJEMPLOS):
        btn.click(
            fn=lambda e=ejemplo: e,
            outputs=msg,
        ).then(
            fn=chat,
            inputs=[msg, chatbot],
            outputs=[msg, chatbot],
        )


if __name__ == "__main__":
    host = os.getenv("GRADIO_SERVER_NAME", "0.0.0.0")
    port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
    print(f"\n🚀 Iniciando Tutor Inteligente de IA...")
    print(f"📍 Abre: http://localhost:{port}\n")
    demo.launch(
        server_name=host,
        server_port=port,
        show_error=True,
        share=False,
        inbrowser=False,
        show_api=False,
    )
