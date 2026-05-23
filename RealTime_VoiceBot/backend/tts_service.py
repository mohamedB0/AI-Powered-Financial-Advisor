"""
TTS Service
───────────
Converts text to speech and yields raw audio bytes.

Backends (TTS_BACKEND env var):
  • "openai"      – OpenAI TTS API   (default; good quality, requires OPENAI_API_KEY)
  • "elevenlabs"  – ElevenLabs stream (high quality, requires ELEVENLABS_API_KEY)
  • "pyttsx3"     – local pyttsx3    (offline fallback, lower quality)
"""

import asyncio
import logging
import os
import tempfile
from functools import partial
from typing import AsyncGenerator

import config

logger = logging.getLogger(__name__)


class TTSService:
    """Async TTS service; yields raw audio byte chunks."""

    async def synthesise(self, text: str) -> AsyncGenerator[bytes, None]:
        """Yield audio bytes for *text* using the configured backend."""
        if not text.strip():
            return

        if config.TTS_BACKEND == "elevenlabs":
            async for chunk in self._elevenlabs(text):
                yield chunk
        elif config.TTS_BACKEND == "openai":
            async for chunk in self._openai_tts(text):
                yield chunk
        else:
            async for chunk in self._pyttsx3(text):
                yield chunk

    # ── OpenAI TTS (default) ──────────────────────────────────────────────────
    async def _openai_tts(self, text: str) -> AsyncGenerator[bytes, None]:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=config.OPENAI_API_KEY)
        async with client.audio.speech.with_streaming_response.create(
            model=config.OPENAI_TTS_MODEL,
            voice=config.OPENAI_TTS_VOICE,
            input=text,
            response_format="mp3",
        ) as response:
            async for chunk in response.iter_bytes(config.TTS_CHUNK_SIZE):
                if chunk:
                    yield chunk

    # ── ElevenLabs ────────────────────────────────────────────────────────────
    async def _elevenlabs(self, text: str) -> AsyncGenerator[bytes, None]:
        import httpx

        url = (
            f"https://api.elevenlabs.io/v1/text-to-speech"
            f"/{config.ELEVENLABS_VOICE_ID}/stream"
        )
        headers = {
            "xi-api-key": config.ELEVENLABS_API_KEY,
            "Content-Type": "application/json",
            "Accept": "audio/mpeg",
        }
        payload = {
            "text": text,
            "model_id": config.ELEVENLABS_MODEL_ID,
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
        }

        async with httpx.AsyncClient(timeout=30) as client:
            async with client.stream("POST", url, json=payload, headers=headers) as resp:
                resp.raise_for_status()
                async for chunk in resp.aiter_bytes(config.TTS_CHUNK_SIZE):
                    if chunk:
                        yield chunk

    # ── pyttsx3 (offline fallback) ────────────────────────────────────────────
    async def _pyttsx3(self, text: str) -> AsyncGenerator[bytes, None]:
        loop = asyncio.get_event_loop()
        audio_bytes = await loop.run_in_executor(
            None, partial(self._pyttsx3_sync, text)
        )
        for i in range(0, len(audio_bytes), config.TTS_CHUNK_SIZE):
            yield audio_bytes[i : i + config.TTS_CHUNK_SIZE]

    @staticmethod
    def _pyttsx3_sync(text: str) -> bytes:
        try:
            import pyttsx3
        except ImportError:
            raise RuntimeError(
                "TTS_BACKEND=pyttsx3 requires pyttsx3.\n"
                "Install with:  pip install pyttsx3"
            )

        engine = pyttsx3.init()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_path = f.name
        try:
            engine.save_to_file(text, tmp_path)
            engine.runAndWait()
            with open(tmp_path, "rb") as f:
                return f.read()
        finally:
            os.unlink(tmp_path)
