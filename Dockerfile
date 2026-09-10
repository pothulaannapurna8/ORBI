FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libgdal-dev \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ backend/
COPY configs/ configs/
COPY scripts/ scripts/
COPY evaluation/ evaluation/

ENV USE_LOCAL_SQLITE_FALLBACK=true
ENV USE_LOCAL_QDRANT_STORAGE=true
ENV QDRANT_STORAGE_DIR=/tmp/qdrant_storage
ENV QDRANT_LOCAL_PATH=/tmp/qdrant_storage

EXPOSE 7860

CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-7860}"]