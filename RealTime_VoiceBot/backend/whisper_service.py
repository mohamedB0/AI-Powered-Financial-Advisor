"""
Whisper Service
───────────────
Transcribes raw audio bytes using OpenAI's Whisper model.

Backends (WHISPER_BACKEND env var):
  • "api"    – OpenAI Whisper API  (default; requires OPENAI_API_KEY)
  • "local"  – openai-whisper package, runs on-device (no API cost)
"""

import asyncio
import io
import logging
import os
import tempfile
from functools import partial

import config

logger = logging.getLogger(__name__)


class WhisperService:
    def __init__(self):
        self._local_model = None
        if config.WHISPER_BACKEND == "local":
            self._load_local_model()

    # ── local model ────────────────────────────────────────────────────────────
    def _load_local_model(self):
        try:
            import whisper
            logger.info("Loading local Whisper model '%s'…", config.WHISPER_LOCAL_MODEL)
            self._local_model = whisper.load_model(config.WHISPER_LOCAL_MODEL)
            logger.info("Local Whisper model loaded ✓")
        except ImportError:
            raise RuntimeError(
                "WHISPER_BACKEND=local requires the 'openai-whisper' package.\n"
                "Install it with:  pip install openai-whisper"
            )

    # ── public API ─────────────────────────────────────────────────────────────
    async def transcribe(self, audio_bytes: bytes) -> str:
        """
        Transcribe *audio_bytes* (any format Whisper accepts) and return text.
        """
        if not audio_bytes:
            return ""

        if config.WHISPER_BACKEND == "local":
            return await self._transcribe_local(audio_bytes)
        return await self._transcribe_api(audio_bytes)

    # ── local backend ──────────────────────────────────────────────────────────
    async def _transcribe_local(self, audio_bytes: bytes) -> str:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, partial(self._transcribe_local_sync, audio_bytes)
        )

    def _transcribe_local_sync(self, audio_bytes: bytes) -> str:
        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as f:
            f.write(audio_bytes)
            tmp_path = f.name
        try:
            result = self._local_model.transcribe(
                tmp_path,
                language=config.WHISPER_LANGUAGE,
            )
            return result.get("text", "").strip()
        finally:
            os.unlink(tmp_path)

    # ── API backend ────────────────────────────────────────────────────────────
    async def _transcribe_api(self, audio_bytes: bytes) -> str:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=config.OPENAI_API_KEY)

        audio_file = io.BytesIO(audio_bytes)
        audio_file.name = "audio.webm"   # SDK requires a .name attribute

        response = await client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            language=config.WHISPER_LANGUAGE,
        )
        return response.text.strip()
