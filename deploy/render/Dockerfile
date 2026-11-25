# Lightweight Render.com Dockerfile (~200 MB RAM, fits 512 MB Free Tier)
# Uses OpenRouter BGE-M3 API + Supabase pgvector + built-in Redis + FastAPI + Flask UI

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ENVIRONMENT=production \
    DISABLE_PHOENIX=true \
    EMBEDDING_BACKEND=openrouter \
    VECTOR_BACKEND=pgvector \
    API_BASE_URL=http://127.0.0.1:8000 \
    AUTH_COOKIE_SECURE=true \
    AUTH_COOKIE_SAMESITE=lax

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev \
    curl \
    redis-server \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY deploy/render/requirements.txt ./requirements-cloud.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-cloud.txt

COPY . .

RUN sed -i 's/\r$//' /app/deploy/render/start.sh && \
    chmod +x /app/deploy/render/start.sh && \
    mkdir -p /app/data/uploads /app/logs /tmp/redis

EXPOSE 10000

CMD ["/app/deploy/render/start.sh"]
