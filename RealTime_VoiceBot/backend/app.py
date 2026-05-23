"""
Real-Time Voice Assistant — FastAPI Entry Point
"""

import logging
import os

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import Response


import config  # loads .env and exposes typed constants

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

from websocket_handler import router as ws_router

app = FastAPI(title="ARIA Voice Assistant", version="1.0.0")

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # lock down to your domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── WebSocket routes ───────────────────────────────────────────────────────────
app.include_router(ws_router)

# ── Serve the frontend (optional — works when run as a single process) ─────────
FRONTEND_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "frontend")
)
if os.path.isdir(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

    @app.get("/")
    async def serve_index():
     return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

    @app.get("/style.css")
    async def serve_css():
     return FileResponse(os.path.join(FRONTEND_DIR, "style.css"))

    @app.get("/script.js")
    async def serve_js():
     return FileResponse(os.path.join(FRONTEND_DIR, "script.js"))

@app.get("/favicon.ico")
async def favicon():
    return Response(status_code=204)

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "whisper_backend": config.WHISPER_BACKEND,
        "llm_model": config.LLM_MODEL,
        "tts_backend": config.TTS_BACKEND,
    }


@app.on_event("startup")
async def on_startup():
    if not config.validate():
        logger.warning(
            "⚠  One or more config errors — server will start but some features may fail."
        )
    logger.info("ARIA running on http://%s:%s", config.HOST, config.PORT)


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host=config.HOST,
        port=config.PORT,
        reload=config.RELOAD,
        log_level="info",
    )
