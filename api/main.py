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

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Routers already carry their own prefix + tags
from api.routes.query import router as query_router
from api.routes.ingest import router as ingest_router
from api.routes.stocks import router as stocks_router
from api.routes.bvc import router as bvc_router
from api.routes.pdf_viewer import router as pdf_router

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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Restrict in production
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(query_router)
app.include_router(ingest_router)
app.include_router(stocks_router)
app.include_router(bvc_router)
app.include_router(pdf_router)


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

    all_ok = all(v == "ok" or v.startswith("ok") for k, v in checks.items() if k not in ("api", "version", "timestamp"))
    checks["status"] = "healthy" if all_ok else "degraded"
    return checks
