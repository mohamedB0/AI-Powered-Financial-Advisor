"""
api/routes/speech.py
Speech-to-Text endpoint using Groq Whisper API.

POST /speech/transcribe — Upload audio, get text back.
"""

import io
import os
import logging

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from pydantic import BaseModel

from api.auth import get_current_user

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/speech", tags=["Speech"])


class TranscribeResponse(BaseModel):
    text: str


@router.post("/transcribe", response_model=TranscribeResponse)
async def transcribe_audio(
    audio: UploadFile = File(...),
    user: dict = Depends(get_current_user),
):
    """
    Transcribe an audio file (WebM/Opus from browser MediaRecorder) to text
    using Groq's Whisper API (free, fast).

    The frontend records the user's voice and sends it here. The returned text
    is then used as the chat input — sent through the normal RAG pipeline.
    """
    groq_key = os.getenv("GROQ_API_KEY", "")
    if not groq_key:
        raise HTTPException(
            status_code=503,
            detail="Speech-to-text is not configured (GROQ_API_KEY missing).",
        )

    # Read the uploaded audio
    audio_bytes = await audio.read()
    if not audio_bytes or len(audio_bytes) < 100:
        raise HTTPException(status_code=400, detail="Audio file is empty or too small.")

    try:
        from openai import OpenAI

        # Use Groq's OpenAI-compatible Whisper endpoint
        client = OpenAI(
            api_key=groq_key,
            base_url="https://api.groq.com/openai/v1",
        )

        audio_file = io.BytesIO(audio_bytes)
        audio_file.name = audio.filename or "recording.webm"

        response = client.audio.transcriptions.create(
            model="whisper-large-v3",
            file=audio_file,
        )

        text = response.text.strip()
        if not text:
            raise HTTPException(status_code=422, detail="Could not transcribe audio. Please try again.")

        logger.info("[STT] Transcribed: %s", text[:80])
        return TranscribeResponse(text=text)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("[STT] Whisper error: %s", e)
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")
