# ═══════════════════════════════════════════════════════════════════════════════
#  AI Financial Advisor — Optimized Production Dockerfile
#  Key optimization: CPU-only PyTorch (saves ~3GB and 10+ min build time)
# ═══════════════════════════════════════════════════════════════════════════════

FROM python:3.12-slim

# Prevent .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install system-level dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Step 1: Install CPU-only PyTorch FIRST (before requirements.txt) ──────────
# This prevents pip from pulling the massive NVIDIA/CUDA GPU packages (~3GB)
# The BGE-M3 embedding model works perfectly on CPU for inference.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu

# ── Step 2: Install all other dependencies ────────────────────────────────────
# Since torch is already installed (CPU), pip will skip downloading the GPU version
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir waitress

# ── Step 3: Copy project source code ─────────────────────────────────────────
COPY . .

# Ensure data directories exist
RUN mkdir -p /app/data/uploads /app/data/collected_data /app/chroma_db /app/logs

# Security: Run as non-root user
RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app
USER appuser

# Set HuggingFace cache to a predictable location for volume mounting
ENV HF_HOME=/app/.hf_cache

# Default command (overridden by docker-compose per service)
CMD ["python", "--version"]
