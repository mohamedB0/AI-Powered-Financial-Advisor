# ─── Stage 1: Base Python image with all dependencies ────────────────────────
FROM python:3.12-slim AS base

# Prevent Python from writing .pyc files and enable unbuffered stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install system-level dependencies required by some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (leverages Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir waitress gunicorn

# Copy the entire project
COPY . .

# Ensure the data directories exist
RUN mkdir -p /app/data/uploads /app/data/collected_data /app/chroma_db /app/logs

# ─── Stage 2: Production targets (selected via docker-compose `command`) ──────
# Default: do nothing (each service overrides the command)
CMD ["python", "--version"]
