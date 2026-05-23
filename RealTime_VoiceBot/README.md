# ARIA — Real-Time AI Voice Assistant

A full-stack, real-time voice assistant that turns your speech into a conversation with an LLM and speaks the answer back — with sentence-level streaming so the first audio chunk arrives before the LLM has even finished generating.

```
Browser mic → WebSocket → Whisper STT → GPT-4o-mini → OpenAI/ElevenLabs TTS → Browser speaker
```

---

## Features

- **Real-time streaming** — tokens and audio chunks are piped back as they're generated; no waiting for a full response
- **Sentence-level TTS** — each sentence is synthesised independently so playback starts immediately
- **Multi-turn memory** — full conversation history sent to the LLM on every turn (configurable window)
- **Pluggable backends** — swap Whisper API ↔ local Whisper, OpenAI TTS ↔ ElevenLabs ↔ pyttsx3 with a single env var
- **Live transcript display** — see what you said and what ARIA is saying, token-by-token
- **Auto-reconnect** — WebSocket reconnects automatically on drop
- **Single-process serving** — the FastAPI backend also serves the frontend, so one port is all you need

---

## Project Structure

```
voice-assistant/
│
├── backend/
│   ├── app.py                # FastAPI entry point + static frontend serving
│   ├── config.py             # Centralised settings, dotenv loader, validation
│   ├── websocket_handler.py  # WebSocket endpoint — orchestrates the pipeline
│   ├── whisper_service.py    # Speech-to-Text (Whisper API or local)
│   ├── llm_service.py        # LLM streaming (OpenAI chat completions)
│   ├── tts_service.py        # Text-to-Speech (OpenAI / ElevenLabs / pyttsx3)
│   ├── requirements.txt
│   └── .env.example
│
├── frontend/
│   ├── index.html            # Single-page app
│   ├── style.css             # Terminal-noir dark theme
│   └── script.js             # WebSocket client, MediaRecorder, audio playback queue
│
├── Dockerfile
├── docker-compose.yml
├── .gitignore
└── README.md
```

---

## Quick Start

### Option A — Docker (recommended)

```bash
# 1. Clone / download the project
cd voice-assistant

# 2. Create your .env
cp backend/.env.example backend/.env
# Edit backend/.env and add your OPENAI_API_KEY at minimum

# 3. Build and run
docker compose up --build

# 4. Open http://localhost:8000 in your browser
```

### Option B — Local Python

**Requirements:** Python 3.11+, ffmpeg

```bash
# 1. Install ffmpeg (needed for audio decoding)
# macOS:   brew install ffmpeg
# Ubuntu:  sudo apt install ffmpeg
# Windows: https://ffmpeg.org/download.html

# 2. Set up the backend
cd voice-assistant/backend
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. Configure
cp .env.example .env
# Edit .env — add OPENAI_API_KEY at minimum

# 4. Run
python app.py

# 5. Open http://localhost:8000 in your browser or
#http://127.0.0.1:8000
```

---

## Configuration

All settings are driven by environment variables. Copy `backend/.env.example` to `backend/.env` and fill in values.

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | *(required)* | OpenAI key — used for Whisper API, LLM, and OpenAI TTS |
| `WHISPER_BACKEND` | `api` | `api` uses OpenAI Whisper API; `local` runs on-device |
| `WHISPER_LANGUAGE` | `en` | ISO 639-1 language code passed to Whisper |
| `WHISPER_LOCAL_MODEL` | `base` | Model size when `WHISPER_BACKEND=local`: `tiny` / `base` / `small` / `medium` / `large` |
| `LLM_MODEL` | `gpt-4o-mini` | Any OpenAI chat model |
| `LLM_MAX_TOKENS` | `512` | Maximum tokens per reply |
| `LLM_TEMPERATURE` | `0.7` | LLM creativity (0 = deterministic, 1 = creative) |
| `LLM_SYSTEM_PROMPT` | *(built-in)* | Override ARIA's persona |
| `MAX_HISTORY_TURNS` | `20` | Conversation turns kept in LLM context |
| `TTS_BACKEND` | `openai` | `openai` / `elevenlabs` / `pyttsx3` |
| `OPENAI_TTS_VOICE` | `alloy` | `alloy` / `echo` / `fable` / `onyx` / `nova` / `shimmer` |
| `OPENAI_TTS_MODEL` | `tts-1` | `tts-1` (fast) or `tts-1-hd` (higher quality) |
| `ELEVENLABS_API_KEY` | *(required for EL)* | ElevenLabs API key |
| `ELEVENLABS_VOICE_ID` | Rachel | ElevenLabs voice ID |
| `ELEVENLABS_MODEL_ID` | `eleven_turbo_v2` | ElevenLabs model |
| `HOST` | `0.0.0.0` | Bind address |
| `PORT` | `8000` | Bind port |
| `RELOAD` | `false` | Hot-reload (dev only) |

---

## Architecture Deep Dive

### Pipeline

```
┌─────────┐  audio bytes   ┌──────────────────┐
│ Browser │ ─────────────► │ websocket_handler │
│  (mic)  │                │                  │
│         │ ◄──── JSON ─── │  1. WhisperSvc   │  audio → transcript
│         │ ◄── tokens ─── │  2. LLMSvc       │  transcript → streamed tokens
│         │ ◄── mp3 bytes ─│  3. TTSSvc       │  sentence → audio chunks
└─────────┘                └──────────────────┘
```

### Sentence-level streaming

The handler accumulates LLM tokens into a buffer and splits on sentence boundaries (`.`, `!`, `?`). Each completed sentence is sent to TTS *while the LLM is still generating the next one*. This reduces perceived latency dramatically: the user hears the first sentence within ~1–2 seconds of speaking.

```
Token stream:  "The" "sky" "is" "blue." "It" "reflects" "…"
                                   ↑
                          TTS job fired here → audio streamed back
```

### Audio format

The browser records with `MediaRecorder` in `audio/webm;codecs=opus` (or the best supported format). ffmpeg (via the Whisper backend) decodes this before transcription. TTS audio is returned as MP3 chunks; the browser queues them in an `AudioContext` and plays sequentially.

---

## Switching Backends

### Use ElevenLabs for TTS
```env
TTS_BACKEND=elevenlabs
ELEVENLABS_API_KEY=your_key
ELEVENLABS_VOICE_ID=21m00Tcm4TlvDq8ikWAM
```

### Use local Whisper (no API cost for STT)
```bash
pip install openai-whisper torch
```
```env
WHISPER_BACKEND=local
WHISPER_LOCAL_MODEL=small   # small is a good balance of speed/accuracy
```

### Fully offline TTS fallback (no API needed)
```bash
pip install pyttsx3
```
```env
TTS_BACKEND=pyttsx3
```

---

## API Reference

### WebSocket — `ws://localhost:8000/ws/voice`

**Client → Server**

| Type | Description |
|---|---|
| Binary frame | Raw audio bytes (WebM/Opus from MediaRecorder) |
| `{"type":"ping"}` | Keep-alive ping |

**Server → Client**

| Type | Payload | Description |
|---|---|---|
| `status` | `{"message": "…"}` | Pipeline stage update |
| `transcript` | `{"text": "…"}` | User's transcribed speech |
| `token` | `{"text": "…"}` | LLM response token (streaming) |
| `done` | `{"reply": "…"}` | Full reply, signals end of turn |
| `error` | `{"message": "…"}` | Error description |
| `pong` | — | Response to ping |
| Binary frame | MP3 bytes | TTS audio chunk for playback |

### REST

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Liveness check + active config summary |
| GET | `/` | Serves `frontend/index.html` |
| GET | `/static/*` | Serves frontend assets |

---

## Development Tips

- **Hot reload:** Set `RELOAD=true` in `.env` and run `python app.py`
- **Cheaper during dev:** Use `LLM_MODEL=gpt-3.5-turbo` and `WHISPER_LOCAL_MODEL=tiny`
- **Check config on startup:** The `/health` endpoint shows which backends are active
- **Browser console:** All WebSocket events are logged for easy debugging
- **Audio not playing?** Check the browser console — `AudioContext` requires a user gesture (the orb click) before it can play audio; this is handled automatically in `script.js`

---

## Requirements

- Python 3.11+
- ffmpeg (for audio decoding in local Whisper mode, and recommended generally)
- A modern browser (Chrome / Edge / Firefox) with microphone access
- OpenAI API key (for Whisper API + LLM; optional if using local Whisper)

---

## License

MIT
