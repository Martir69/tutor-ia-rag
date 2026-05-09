FROM python:3.12-slim

WORKDIR /app

# Sistema: OCR (tesseract + español) + PDF renderer (poppler) + curl para healthchecks
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-spa \
    tesseract-ocr-eng \
    poppler-utils \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Dependencias Python primero (layer cacheado)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Código fuente
COPY . .

# Directorios de datos (los volúmenes los montan desde el host)
RUN mkdir -p docs chroma_db scripts

# Permisos del entrypoint
RUN chmod +x scripts/entrypoint.sh 2>/dev/null || true

EXPOSE 7860

ENTRYPOINT ["scripts/entrypoint.sh"]
