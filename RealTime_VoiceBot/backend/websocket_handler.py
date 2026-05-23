"""
WebSocket Handler
─────────────────
Handles the real-time pipeline:
  1. Receive raw audio bytes from the browser (PCM / WebM chunks)
  2. Transcribe with Whisper
  3. Stream response tokens from the LLM
  4. Convert each sentence to speech via TTS and stream audio back
"""

import asyncio
import json
import logging

import config  # noqa: F401 — ensures .env is loaded before services init
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from whisper_service import WhisperService
from llm_service import LLMService
from tts_service import TTSService

logger = logging.getLogger(__name__)
router = APIRouter()

# Singletons (initialised once at import time to avoid cold-start per request)
whisper = WhisperService()
llm = LLMService()
tts = TTSService()


# ── helpers ───────────────────────────────────────────────────────────────────

def _split_into_sentences(text: str) -> list[str]:
    """Naïve sentence splitter for TTS chunking."""
    import re
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


async def _send_json(ws: WebSocket, payload: dict):
    await ws.send_text(json.dumps(payload))


# ── WebSocket endpoint ────────────────────────────────────────────────────────

@router.websocket("/ws/voice")
async def voice_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("Client connected")

    # Conversation history for the LLM (multi-turn)
    conversation: list[dict] = []

    try:
        while True:
            # ── Step 1: receive audio blob ─────────────────────────────────
            raw_message = await websocket.receive()

            if "bytes" in raw_message:
                audio_bytes: bytes = raw_message["bytes"]
            elif "text" in raw_message:
                data = json.loads(raw_message["text"])
                if data.get("type") == "ping":
                    await _send_json(websocket, {"type": "pong"})
                    continue
                # Control message — ignore unknown
                continue
            else:
                continue

            # ── Step 2: Speech → Text ──────────────────────────────────────
            await _send_json(websocket, {"type": "status", "message": "Transcribing…"})
            try:
                transcript = await whisper.transcribe(audio_bytes)
            except Exception as exc:
                logger.error("Whisper error: %s", exc)
                await _send_json(websocket, {"type": "error", "message": f"STT failed: {exc}"})
                continue

            if not transcript:
                await _send_json(websocket, {"type": "status", "message": "Could not understand audio"})
                continue

            await _send_json(websocket, {"type": "transcript", "text": transcript})
            logger.info("Transcript: %s", transcript)

            # Add user turn to conversation
            conversation.append({"role": "user", "content": transcript})

            # ── Step 3: LLM (streaming) ────────────────────────────────────
            await _send_json(websocket, {"type": "status", "message": "Thinking…"})

            full_reply = ""
            sentence_buffer = ""

            try:
                async for token in llm.stream(conversation):
                    full_reply += token
                    sentence_buffer += token

                    # Send token to frontend for live display
                    await _send_json(websocket, {"type": "token", "text": token})

                    # When we have a complete sentence, synthesise immediately
                    sentences = _split_into_sentences(sentence_buffer)
                    if len(sentences) > 1:
                        # All but the last fragment are "complete"
                        for sentence in sentences[:-1]:
                            await _synthesise_and_send(websocket, sentence)
                        sentence_buffer = sentences[-1]  # keep the tail

            except Exception as exc:
                logger.error("LLM error: %s", exc)
                await _send_json(websocket, {"type": "error", "message": f"LLM failed: {exc}"})
                continue

            # Flush any remaining sentence buffer
            if sentence_buffer.strip():
                await _synthesise_and_send(websocket, sentence_buffer.strip())

            # Add assistant turn to history
            conversation.append({"role": "assistant", "content": full_reply})
            logger.info("Reply: %s", full_reply)

            await _send_json(websocket, {"type": "done", "reply": full_reply})

    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as exc:
        logger.exception("Unexpected error: %s", exc)
        try:
            await _send_json(websocket, {"type": "error", "message": str(exc)})
        except Exception:
            pass


async def _synthesise_and_send(ws: WebSocket, text: str):
    """Convert text → audio and stream audio chunks back to the client."""
    try:
        async for chunk in tts.synthesise(text):
            # Send raw audio bytes; the browser will queue and play them
            await ws.send_bytes(chunk)
    except Exception as exc:
        logger.error("TTS error for '%s': %s", text[:40], exc)
        await _send_json(ws, {"type": "error", "message": f"TTS failed: {exc}"})
