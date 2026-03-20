# Use a slim Python image for ARM64 compatibility
FROM python:3.12-slim

# Install uv binary from official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Install system dependencies for ChromaDB and build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && rm -rf /var/lib/apt/lists/*

# Use uv to install dependencies into the system environment
# This is faster on a Pi and avoids venv complexity in Docker
COPY backend/requirements.txt ./
RUN uv pip install --system --no-cache -r requirements.txt

# Pre-download the embedding model (Crucial for Pi performance!)
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')"

# Copy source and data
COPY backend/ ./backend/
COPY data/ ./data/
COPY start.sh ./start.sh
RUN chmod +x start.sh

EXPOSE 8000 8001
CMD ["./start.sh"]