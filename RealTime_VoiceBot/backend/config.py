"""
config.py — Centralised settings loader
────────────────────────────────────────
Loads .env, validates required keys, and exposes typed constants.
Import this at the top of every service instead of os.getenv() directly.
"""

import os
import sys
import logging
from pathlib import Path

# Load .env if python-dotenv is available (always in dev, optional in prod)
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).parent / ".env"
    load_dotenv(dotenv_path=_env_path, override=False)
except ImportError:
    pass  # dotenv not installed — env must be set externally

logger = logging.getLogger(__name__)

# ── OpenAI ─────────────────────────────────────────────────────────────────────
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

# ── Whisper ────────────────────────────────────────────────────────────────────
WHISPER_BACKEND: str    = os.getenv("WHISPER_BACKEND", "api").lower()   # api | local
WHISPER_LOCAL_MODEL: str = os.getenv("WHISPER_LOCAL_MODEL", "base")
WHISPER_LANGUAGE: str   = os.getenv("WHISPER_LANGUAGE", "en")

# ── LLM ────────────────────────────────────────────────────────────────────────
LLM_MODEL: str          = os.getenv("LLM_MODEL", "gpt-4o-mini")
LLM_MAX_TOKENS: int     = int(os.getenv("LLM_MAX_TOKENS", "512"))
LLM_TEMPERATURE: float  = float(os.getenv("LLM_TEMPERATURE", "0.7"))
LLM_SYSTEM_PROMPT: str  = os.getenv(
    "LLM_SYSTEM_PROMPT",
    (
        "You are ARIA, a helpful and concise voice assistant. "
        "Respond in plain, natural spoken language — no markdown, no bullet points, "
        "no lists. Keep answers brief unless the user explicitly asks for detail."
    ),
)

# ── TTS ─────────────────────────────────────────────────────────────────────────
TTS_BACKEND: str            = os.getenv("TTS_BACKEND", "openai").lower()  # elevenlabs | openai | pyttsx3
TTS_CHUNK_SIZE: int         = int(os.getenv("TTS_CHUNK_SIZE", "4096"))

ELEVENLABS_API_KEY: str     = os.getenv("ELEVENLABS_API_KEY", "")
ELEVENLABS_VOICE_ID: str    = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
ELEVENLABS_MODEL_ID: str    = os.getenv("ELEVENLABS_MODEL_ID", "eleven_turbo_v2")

OPENAI_TTS_VOICE: str       = os.getenv("OPENAI_TTS_VOICE", "alloy")
OPENAI_TTS_MODEL: str       = os.getenv("OPENAI_TTS_MODEL", "tts-1")

# ── Server ─────────────────────────────────────────────────────────────────────
HOST: str   = os.getenv("HOST", "0.0.0.0")
PORT: int   = int(os.getenv("PORT", "8000"))
RELOAD: bool = os.getenv("RELOAD", "false").lower() == "true"

# ── Conversation ───────────────────────────────────────────────────────────────
MAX_HISTORY_TURNS: int = int(os.getenv("MAX_HISTORY_TURNS", "20"))


# ── Startup validation ────────────────────────────────────────────────────────
def validate() -> bool:
    """
    Check that all required credentials are present.
    Returns True if valid, False (and prints warnings) if not.
    Call this from app startup.
    """
    errors: list[str] = []

    if not OPENAI_API_KEY:
        errors.append("OPENAI_API_KEY is not set (needed for Whisper API + LLM)")

    if TTS_BACKEND == "elevenlabs" and not ELEVENLABS_API_KEY:
        errors.append(
            "TTS_BACKEND=elevenlabs but ELEVENLABS_API_KEY is not set. "
            "Set the key or switch to TTS_BACKEND=openai"
        )

    if WHISPER_BACKEND not in ("api", "local"):
        errors.append(f"Unknown WHISPER_BACKEND '{WHISPER_BACKEND}'. Use 'api' or 'local'.")

    if TTS_BACKEND not in ("elevenlabs", "openai", "pyttsx3"):
        errors.append(f"Unknown TTS_BACKEND '{TTS_BACKEND}'. Use 'elevenlabs', 'openai', or 'pyttsx3'.")

    if errors:
        for err in errors:
            logger.error("CONFIG ERROR: %s", err)
        return False

    logger.info(
        "Config OK — Whisper=%s | LLM=%s | TTS=%s",
        WHISPER_BACKEND, LLM_MODEL, TTS_BACKEND,
    )
    return True
