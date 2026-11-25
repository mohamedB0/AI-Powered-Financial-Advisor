#!/usr/bin/env bash
set -e

PORT="${PORT:-10000}"

echo "=== Starting AI Financial Advisor (Render Cloud) on port $PORT ==="

# Start built-in Redis if no external REDIS_HOST is configured
if [ -z "${REDIS_HOST:-}" ] || [ "${REDIS_HOST}" = "127.0.0.1" ] || [ "${REDIS_HOST}" = "localhost" ]; then
    export REDIS_HOST=127.0.0.1
    export REDIS_PORT=6379
    redis-server --daemonize yes --dir /tmp/redis --bind 127.0.0.1 --port 6379 --maxmemory 48mb --maxmemory-policy allkeys-lru
    echo "[1/3] Built-in Redis started on 127.0.0.1:6379"
fi

# Start FastAPI backend on internal port 8000
uvicorn api.main:app \
    --host 127.0.0.1 \
    --port 8000 \
    --workers 1 \
    --timeout-keep-alive 120 &
echo "[2/3] FastAPI starting on 127.0.0.1:8000"

# Start Flask UI on Render's $PORT
echo "[3/3] Starting Flask UI on 0.0.0.0:$PORT"
exec waitress-serve \
    --host 0.0.0.0 \
    --port "$PORT" \
    --threads 4 \
    ui.app:app
