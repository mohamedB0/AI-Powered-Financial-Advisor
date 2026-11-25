#!/usr/bin/env bash
set -e

echo "=== Starting AI Financial Advisor (Hugging Face Space) ==="

# 1. Start local Redis in background (sessions + Celery broker)
redis-server --daemonize yes --dir /tmp/redis --bind 127.0.0.1 --port 6379
echo "[1/3] Redis started on 127.0.0.1:6379"

# 2. Start FastAPI backend on internal port 8000
uvicorn api.main:app \
    --host 127.0.0.1 \
    --port 8000 \
    --workers 1 \
    --timeout-keep-alive 120 &
API_PID=$!
echo "[2/3] FastAPI starting on 127.0.0.1:8000 (PID $API_PID)"

# 3. Start Celery worker in background for uploads/tasks (optional, single-threaded)
celery -A tasks.celery_app:celery_app worker \
    --loglevel=info \
    -Q user_ingestion,scraping,data_fetch,ingestion,local_ingestion,default \
    -c 1 \
    --pool=solo &
echo "[2b/3] Celery worker started"

# 4. Start Flask UI immediately on port 7860 (required by HF Spaces health check)
echo "[3/3] Starting Flask UI on 0.0.0.0:7860..."
exec waitress-serve \
    --host 0.0.0.0 \
    --port 7860 \
    --threads 4 \
    ui.app:app
