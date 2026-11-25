# ═══════════════════════════════════════════════════════════════════════════════
#  AI Financial Advisor — Hugging Face Spaces Dockerfile (All-in-One)
#  Runs Redis + FastAPI (port 8000) + Flask UI (port 7860) in one 16GB container.
# ═══════════════════════════════════════════════════════════════════════════════

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.hf_cache \
    ENVIRONMENT=production \
    DISABLE_PHOENIX=true \
    REDIS_HOST=127.0.0.1 \
    REDIS_PORT=6379 \
    API_BASE_URL=http://127.0.0.1:8000 \
    CHROMA_PERSIST_DIR=/app/chroma_db \
    AUTH_COOKIE_SECURE=true \
    AUTH_COOKIE_SAMESITE=none

# Install system dependencies + Redis server
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    redis-server \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install CPU-only PyTorch first (saves ~3GB)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir waitress

# Copy project source code & ChromaDB
COPY . .

# Make entrypoint executable and prepare writable directories
# (HF Spaces runs containers with UID 1000)
RUN sed -i 's/\r$//' /app/deploy/hf/start.sh && \
    chmod +x /app/deploy/hf/start.sh && \
    mkdir -p /app/data/uploads /app/data/collected_data /app/chroma_db /app/logs /app/.hf_cache /tmp/redis && \
    useradd -m -u 1000 user && \
    chown -R user:user /app /tmp/redis

USER user

EXPOSE 7860

CMD ["/app/deploy/hf/start.sh"]
