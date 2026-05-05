# AI Financial Advisor

A distributed, multilingual Hybrid RAG (Retrieval-Augmented Generation) platform for financial analysis of Moroccan public finance institutions. Combines vector search over scraped documents with SQL queries over structured financial data and real-time web search to deliver structured insights via natural language — in **French, Arabic, and English**.

**PFE 2026 - Final Year Project | Field: AI / FinTech | Author: M.Bouada - PFE2026a.lahrache**

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| **"Two-Brain" LLM Architecture** | Fast 8B model (Llama-3.1) for routing/orchestration + Heavy 72B model (Qwen-2.5) for high-quality generation |
| **Multilingual RAG** | Native support for French, Arabic, and English queries with language-anchored responses |
| **Focus Modes** | Chat tabs lock the AI into specific retrieval paths: *New Chat* (auto), *Company Analysis* (SQL), *Market Macro* (Vector/PDF) |
| **Investment Persona** | Users can select Balanced, Prudent, or Aggressive personas via the Settings panel — dynamically injected into the LLM system prompt |
| **In-UI PDF Viewer** | Clickable source citations open the original PDF page directly inside the chat interface |
| **Export Tools** | Copy, Download .md, and Export Full Report buttons on every AI response |
| **Live Settings Panel** | Adjust research depth, swap API keys, and change personas without restarting the server |
| **Jina AI Reranker** | Multilingual semantic reranking of retrieved chunks for precision and token efficiency |
| **CRAG Reflection Loop** | Automatic query rewriting and web search fallback when local context is insufficient |
| **Guardrail System** | Fast safety check blocks illegal investment advice and off-topic queries |
| **Background Ingestion** | Upload PDFs via the UI — Celery workers handle OCR, chunking, embedding, and storage asynchronously |
| **Docker-Ready** | Full `docker-compose.yml` with 7 production services |

---

## Architecture Overview

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│   Flask UI   │────▶│  FastAPI API  │────▶│  LangGraph Agent │
│  (Waitress)  │     │  (Uvicorn)   │     │   StateGraph     │
│  Port 5000   │     │  Port 8000   │     └────────┬────────┘
└─────────────┘     └──────────────┘              │
                                          ┌───────┼───────┐
                                          ▼       ▼       ▼
                                     ┌────────┐ ┌─────┐ ┌─────┐
                                     │ChromaDB│ │ SQL │ │ Web │
                                     │ BGE-M3 │ │     │ │     │
                                     └────────┘ └─────┘ └─────┘
                                          │
                                     ┌────┴────┐
                                     │Jina AI  │
                                     │Reranker │
                                     └─────────┘
```

---

## Quick Start

### Important Note on Data & PDFs
> **Note:** The `data/` directory (containing raw downloaded PDFs) is ignored by git to keep the repository lightweight. When you run the Celery workers, they will automatically scrape, download, and ingest the PDFs into ChromaDB.

### Option A: Local Development

#### 1. Setup environment
```bash
cd "Pfe 2026/financial-advisor"
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
# Edit .env with your API keys (see Configuration section)
```

#### 2. Start Infrastructure (Docker)
```bash
docker-compose up -d postgres redis
```

#### 3. Start the API
```bash
uvicorn api.main:app --reload
# Swagger UI: http://localhost:8000/docs
```

#### 4. Start the UI
```bash
python ui/app.py
# Open: http://localhost:5000/chat
```

#### 5. Start Celery Worker
```bash
python scripts/run_celery.py --worker
```

### Option B: Full Docker Production

```bash
# Build and launch all 7 services
docker-compose up -d --build

# Check status
docker-compose ps

# View logs
docker-compose logs -f api

# Stop everything
docker-compose down
```

#### Production Services

| Service | Server | Port | Role |
|---------|--------|------|------|
| **postgres** | PostgreSQL 16 | 5432 | Relational data storage |
| **redis** | Redis 7 | 6379 | Celery message broker |
| **api** | Uvicorn (4 workers) | 8000 | FastAPI backend |
| **ui** | Waitress (4 threads) | 5000 | Flask frontend |
| **celery-worker** | Celery (4 threads) | — | Background ingestion |
| **celery-beat** | Celery Beat | — | Scheduled tasks |
| **flower** | Flower | 5555 | Celery monitoring dashboard |

---

## Data Ingestion

### Via the UI (Recommended)
Click the **"Ingest Data"** button in the chat interface, select a PDF, and watch the live progress bar. The file is processed asynchronously by the Celery worker through: Upload → OCR → Chunking → Embedding → ChromaDB Storage.

### Via Config-Driven Scraping
Edit `config/scrapers/sites.json` and add a new entry:
```json
{
  "name": "bank_al_maghrib",
  "display_name": "Bank Al-Maghrib (BKAM)",
  "base_url": "https://www.bkam.ma",
  "listing_urls": ["https://www.bkam.ma/Publications-et-recherche/Rapports-annuels"],
  "pdf_link_pattern": ".pdf",
  "use_playwright": false,
  "enabled": true
}
```
Then: `python scripts/trigger_all.py --sites-only`

### Via Manual Drop
Place any PDF in `data/docs/` and run:
```bash
python scripts/trigger_all.py --local-only
```

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `POST /query/` | POST | Natural language financial question |
| `POST /query/stream` | POST | Streaming (SSE) version with token-by-token output |
| `POST /ingest/async` | POST | Upload a PDF for background ingestion |
| `GET /ingest/status/{task_id}` | GET | Poll ingestion progress |
| `GET /pdf/file?path=...` | GET | Serve an ingested PDF for in-UI preview |
| `GET /stocks/{ticker}` | GET | Stock price snapshot |
| `GET /stocks/{ticker}/history` | GET | Historical prices |
| `GET /bvc/prices` | GET | Live BVC (Casablanca Bourse) data |
| `GET /health` | GET | API health check |

### Query Example (with Focus Mode & Persona)
```json
POST /query/stream
{
  "question": "Quel est le taux d'inflation au Maroc en 2024 ?",
  "history": [],
  "persona": "Prudent",
  "top_k": 3,
  "focus_mode": "Market Macro"
}
```

---

## Technology Stack

| Component | Technology |
|-----------|-----------|
| Heavy LLM | Qwen-2.5-72B-Instruct via OpenRouter |
| Fast LLM | Llama-3.1-8B-Instruct via OpenRouter |
| Embeddings | BAAI/bge-m3 (1024 dims, multilingual) |
| Reranker | Jina AI Multilingual Reranker API |
| Vector DB | ChromaDB (local persistent) |
| Relational DB | PostgreSQL 16 |
| Task Queue | Celery + Redis |
| Orchestration | LangGraph StateGraph (CRAG pattern) |
| Web Search | Tavily API |
| Web Scraping | Playwright + BeautifulSoup |
| Backend API | FastAPI + Uvicorn |
| Frontend UI | Flask + Tailwind CSS |
| Monitoring | Celery Flower |
| Containerization | Docker + Docker Compose |

---

## Configuration

Key environment variables in `.env`:

```bash
# LLM (OpenRouter) — Two-Brain Architecture
OPENROUTER_API_KEY=sk-or-v1-...
OPENROUTER_MODEL=qwen/qwen-2.5-72b-instruct       # Heavy brain (generation)
FAST_LLM_MODEL=meta-llama/llama-3.1-8b-instruct    # Fast brain (routing/grading)

# Embeddings (Multilingual)
EMBEDDING_MODEL=BAAI/bge-m3
EMBEDDING_DIMENSION=1024

# ChromaDB
CHROMA_PERSIST_DIR=./chroma_db
CHROMA_COLLECTION=financial_data

# Reranker
JINA_API_KEY=jina_...

# Web Search
TAVILY_API_KEY=tvly-...

# PostgreSQL
DATABASE_URL=postgresql+psycopg2://postgres:root@localhost:5432/financial_advisor

# Redis / Celery
REDIS_HOST=localhost
REDIS_PORT=6379
USE_CELERY=true
```

---

## Project Structure

```
financial-advisor/
├── agents/                  # LangGraph agent, prompts, tools
│   ├── financial_agent.py   # Main RAG agent with CRAG reflection
│   ├── prompts.py           # System prompts & templates
│   └── tools.py             # Agent tools
├── api/                     # FastAPI backend
│   ├── main.py              # App entry point
│   └── routes/              # Query, Ingest, PDF Viewer, Dashboard
├── config/                  # Scraper configs (sites.json)
├── data/                    # Raw PDFs & uploaded documents
├── ingestion/               # Embedding & chunking logic
├── retrieval/               # Vector, SQL, Web search + Reranker
├── scripts/                 # CLI utilities (trigger, check, run)
├── storage/                 # ChromaDB store wrapper
├── tasks/                   # Celery task definitions
├── ui/                      # Flask frontend
│   ├── app.py               # Flask app
│   ├── templates/           # HTML (chat, pipeline, base)
│   └── static/              # CSS, JS, images
├── Dockerfile               # Production container image
├── docker-compose.yml       # 7-service orchestration
├── celery_config.py         # Celery configuration
├── requirements.txt         # Python dependencies
└── .env                     # Environment variables (not in git)
```

---

## Diagnostic Tools

```bash
# Check ChromaDB ingestion status
python scripts/check_chromadb.py

# Run API unit tests
pytest tests/test_api.py -v

# Monitor Celery tasks
python scripts/run_flower.py    # Open http://localhost:5555
```

---

## License

MIT License - PFE 2026
