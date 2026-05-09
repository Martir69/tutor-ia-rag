#!/bin/bash
set -e

echo ""
echo "╔════════════════════════════════════════════╗"
echo "║     🎓 Tutor Inteligente de IA — Setup    ║"
echo "║     Universidad Mariano Gálvez             ║"
echo "╚════════════════════════════════════════════╝"
echo ""

# Verificar Ollama
if ! command -v ollama &>/dev/null; then
    echo "Instalando Ollama..."
    sudo apt install zstd -y
    curl -fsSL https://ollama.com/install.sh | sh
fi
echo "Ollama: $(ollama --version)"

# Descargar modelo
if ! ollama list | grep -q "phi3:mini"; then
    echo "📥 Descargando phi3:mini (~2.3GB)..."
    ollama pull phi3:mini
fi
echo "Modelo listo"

# Entorno virtual
python3 -m venv venv
source venv/bin/activate
pip install --quiet -r requirements.txt
echo " Dependencias instaladas"

echo ""
echo "╔════════════════════════════════════════════╗"
echo "║   Setup completo. Pasos siguientes:      ║"
echo "║                                            ║"
echo "║  source venv/bin/activate                  ║"
echo "║  python3 -m src.infrastructure.indexer    ║"
echo "║  python3 -m src.interface.app              ║"
echo "║  → http://localhost:7860                   ║"
echo "╚════════════════════════════════════════════╝"