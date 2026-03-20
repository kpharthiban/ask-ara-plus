# ============================================================
# AskAra+ Backend — Production Dockerfile
# ============================================================

FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── CPU-only PyTorch (saves ~1.5GB vs CUDA version) ──
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# ── Python dependencies ──
COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

# ── Pre-download embedding model (avoids runtime download delay) ──
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2'); \
print('[Docker] Embedding model cached.')"

# ── Copy source + data ──
COPY backend/ ./backend/
COPY data/ ./data/
COPY start.sh ./start.sh
RUN chmod +x start.sh

EXPOSE 8000 8001

CMD ["./start.sh"]