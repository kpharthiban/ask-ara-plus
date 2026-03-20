# 3.11 is the sweet spot for these versions
FROM python:3.11-slim-bookworm

# 1. Hardware Environment Variables (Keep these!)
ENV OPENBLAS_CORETYPE=ARMV8
ENV OMP_NUM_THREADS=1
ENV LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1
ENV TOKENIZERS_PARALLELISM=false

WORKDIR /app

# 2. Install native math backends
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libopenblas0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 3. CRITICAL: Pin EVERYTHING to ARM-safe versions
# sentence-transformers 3.0.1 is very stable with Torch 2.3.1
RUN pip install --no-cache-dir \
    "numpy==1.26.4" \
    "torch==2.3.1" \
    "transformers==4.40.0" \
    "sentence-transformers==3.0.1"

# 4. Install the rest of your requirements
COPY backend/requirements.txt ./
# Filter out the libraries we just manually pinned to avoid overwriting them
RUN grep -vE '^(torch|numpy|transformers|sentence-transformers)' requirements.txt > reqs_no_ml.txt && \
    pip install --no-cache-dir -r reqs_no_ml.txt

# 5. Hardware Verification (Should show OK)
RUN python -c "import numpy; import torch; print(f'PI CHECK: NumPy {numpy.__version__} & Torch {torch.__version__} are LIVE!')"

# 6. Pre-download the brain
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')"

# 7. Copy source + data
COPY backend/ ./backend/
COPY data/ ./data/
COPY start.sh ./
RUN chmod +x start.sh

EXPOSE 8000 8001
CMD ["./start.sh"]
