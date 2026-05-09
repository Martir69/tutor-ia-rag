#!/bin/bash
set -e

OLLAMA_HOST="${OLLAMA_HOST:-http://ollama:11434}"
MODEL="${LLM_MODEL:-phi3:mini}"

# ── 1. Esperar Ollama ─────────────────────────────────────────────────────────
echo "⏳ Esperando Ollama en $OLLAMA_HOST..."
until curl -sf "$OLLAMA_HOST/api/tags" > /dev/null 2>&1; do
    sleep 3
done
echo "✅ Ollama disponible"

# ── 2. Descargar modelo si no existe ─────────────────────────────────────────
echo "🤖 Verificando modelo $MODEL..."
if ! curl -sf "$OLLAMA_HOST/api/tags" | grep -q "\"$MODEL\""; then
    echo "📥 Descargando $MODEL (puede tardar varios minutos en la primera ejecución)..."
    curl -s -X POST "$OLLAMA_HOST/api/pull" \
         -H "Content-Type: application/json" \
         -d "{\"name\":\"$MODEL\",\"stream\":false}" \
         --max-time 600
    echo ""
fi
echo "✅ Modelo $MODEL listo"

# ── 3. Indexar documentos ─────────────────────────────────────────────────────
echo "📚 Indexando documentos del curso..."
cd /app && python3 -m src.infrastructure.indexer
echo "✅ Indexación completada"

# ── 4. Lanzar UI ─────────────────────────────────────────────────────────────
echo "🚀 Iniciando Tutor Inteligente de IA en http://0.0.0.0:7860..."
exec python3 -m src.interface.app
