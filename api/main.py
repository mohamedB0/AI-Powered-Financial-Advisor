"""
api/main.py
FastAPI application entry point with CORS, routers, and health check.

Run with the fastapi CLI (reads entrypoint from pyproject.toml):
    fastapi dev        # development — auto-reload
    fastapi run        # production

Or explicitly:
    fastapi dev api/main.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

# ── Arize Phoenix Observability ───────────────────────────────────────────────
import os

def _setup_phoenix_tracing():
    """Wire up Arize Phoenix as the OpenTelemetry trace backend.
    Instruments all LangChain / LangGraph calls automatically.
    Also enables manual spans from api/tracing.py for full pipeline visibility.
    Phoenix UI: http://localhost:6006
    """
    try:
        from opentelemetry.sdk import trace as trace_sdk
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from openinference.instrumentation.langchain import LangChainInstrumentor
        from opentelemetry import trace

        endpoint = os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006/v1/traces")

        # ⚡ Probe Phoenix once at startup. If it's not reachable, skip the
        # exporter entirely so we don't pay retry latency on every span.
        # (Set DISABLE_PHOENIX=true to force-skip even when reachable.)
        if os.getenv("DISABLE_PHOENIX", "false").lower() == "true":
            print("[Phoenix] Disabled via DISABLE_PHOENIX env. Skipping tracing setup.")
            return

        try:
            from urllib.parse import urlparse
            import socket
            parsed = urlparse(endpoint)
            host = parsed.hostname or "localhost"
            port = parsed.port or (443 if parsed.scheme == "https" else 6006)
            with socket.create_connection((host, port), timeout=1.0):
                pass
        except Exception as probe_err:
            print(f"[Phoenix] Collector at {endpoint} is unreachable ({probe_err}). "
                  f"Tracing disabled this run — start Phoenix to enable it.")
            return

        exporter = OTLPSpanExporter(endpoint=endpoint)

        # Resource attributes show in Phoenix's service/project filters
        resource = Resource.create({
            "service.name": "alpha-insight-api",
            "service.version": "1.0.0",
            "deployment.environment": os.getenv("ENVIRONMENT", "development"),
        })

        provider = trace_sdk.TracerProvider(resource=resource)
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)

        # Auto-instruments LangChain chains, LLM calls, retrievers, and LangGraph nodes
        LangChainInstrumentor().instrument()
        print(f"[Phoenix] Tracing active -> {endpoint}  |  UI: http://localhost:6006")
        print(f"[Phoenix] Service: alpha-insight-api | Manual pipeline spans enabled")
    except Exception as e:
        print(f"[Phoenix] Tracing setup failed (non-fatal): {e}")

_setup_phoenix_tracing()
# ──────────────────────────────────────────────────────────────────────────────

# ⚡ Silence the noisy "Failed to detach context" warnings emitted by
# OpenTelemetry when LangChain runs inside a thread pool. They log at
# ERROR via logger.exception(), so we need a filter (or CRITICAL+1).
# Functionally harmless — but they spam the logs on every parallel stage.
import logging as _otel_logging

class _DropDetachWarning(_otel_logging.Filter):
    def filter(self, record):
        msg = record.getMessage()
        if "Failed to detach context" in msg:
            return False
        return True

for _name in ("opentelemetry.context", "opentelemetry.context.contextvars_context",
              "opentelemetry"):
    _logger = _otel_logging.getLogger(_name)
    _logger.addFilter(_DropDetachWarning())
    _logger.setLevel(_otel_logging.CRITICAL)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Routers already carry their own prefix + tags
from api.routes.auth import router as auth_router
from api.routes.query import router as query_router
from api.routes.ingest import router as ingest_router
from api.routes.stocks import router as stocks_router
from api.routes.bvc import router as bvc_router
from api.routes.pdf_viewer import router as pdf_router
from api.routes.conversations import router as conversations_router
from api.routes.speech import router as speech_router
from api.routes.vision import router as vision_router

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    import logging
    logger = logging.getLogger(__name__)
    logger.info("Pre-loading the heavy BGE-M3 model into memory...")
    try:
        from ingestion.embeddings import Embedder
        _ = Embedder() # Trigger the lazy-load of the 2.2GB model
        logger.info("Model loaded successfully! Fast API is fully ready.")
    except Exception as e:
        logger.error(f"Failed to pre-load model: {e}")
        
    yield
    # Cleanup phase can be added here if needed

app = FastAPI(
    title="AI Financial Advisor API",
    description="Hybrid RAG backend for financial analysis — PFE 2026",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS: Use explicit origins from CORS_ORIGINS env var.
# Fallback to common dev origins if not set.
_cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:5000,http://127.0.0.1:5000,http://localhost:3000")
_allowed_origins = [o.strip() for o in _cors_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "Accept", "X-Requested-With"],
)

app.include_router(auth_router)
app.include_router(conversations_router)
app.include_router(query_router)
app.include_router(ingest_router)
app.include_router(stocks_router)
app.include_router(bvc_router)
app.include_router(pdf_router)
app.include_router(speech_router)
app.include_router(vision_router)


@app.get("/", tags=["Health"])
def root() -> dict:
    return {"status": "ok", "service": "AI Financial Advisor API", "version": "1.0.0"}


@app.get("/health", tags=["Health"])
def health() -> dict:
    """
    Returns system health including DB and vector store connectivity.
    """
    import datetime
    checks = {"api": "ok", "version": "1.0.0", "timestamp": datetime.datetime.utcnow().isoformat()}

    # Check PostgreSQL
    try:
        from storage.relational_db import RelationalDB
        db = RelationalDB()
        db.execute_sql("SELECT 1")
        checks["postgres"] = "ok"
    except Exception as e:
        checks["postgres"] = f"error: {str(e)[:80]}"

    # Check vector store (ChromaDB)
    try:
        from storage.chroma_store import ChromaStore
        cs = ChromaStore()
        checks["vector_store"] = f"ok ({cs.count()} vectors)"
    except Exception as e:
        checks["vector_store"] = f"error: {str(e)[:80]}"

    # Check Redis session store
    try:
        from api.auth import get_redis
        r = get_redis()
        r.ping()
        checks["redis_sessions"] = "ok"
    except Exception as e:
        checks["redis_sessions"] = f"error: {str(e)[:80]}"

    all_ok = all(v == "ok" or v.startswith("ok") for k, v in checks.items() if k not in ("api", "version", "timestamp"))
    checks["status"] = "healthy" if all_ok else "degraded"
    return checks
