# Alpha Insight — AI Financial Advisor

A distributed, multilingual **Hybrid RAG** (Retrieval-Augmented Generation) platform for financial analysis. Combines vector search over institutional documents with SQL queries over structured financial data and real-time web search to deliver structured insights via natural language — in **French, Arabic, and English**.

**PFE 2026** - Final Year Project | Field: AI / FinTech | Author: M.Bouada

---

## Key Features

| Feature | Description |
|---------|-------------|
| **Hybrid RAG Architecture** | Combines Vector (ChromaDB) + SQL (PostgreSQL) + Web (Tavily) retrieval, orchestrated by LangGraph |
| **CRAG Reflection Loop** | Auto-evaluates context quality, rewrites queries, and falls back to web search when needed |
| **Multilingual (FR/AR/EN)** | Native support with automatic language detection and enforced response language matching |
| **Google OAuth + Email Auth** | Dual authentication: Google Sign-In or classic email/password registration |
| **Voice Input (STT)** | Microphone button transcribes speech via Groq Whisper Large v3, fills chat input |
| **Live BVC Market Panel** | Real-time Casablanca stock exchange data with gainers/losers/most active tabs |
| **Collection Selector** | Users choose which knowledge base to query: All, Documents Only, or My Uploads Only |
| **Focus Modes** | Chat tabs: *New Chat* (auto), *Company Analysis* (SQL), *Market Macro* (Vector) |
| **Investment Personas** | Balanced, Prudent, or Aggressive — dynamically injected into LLM system prompt |
| **In-UI PDF Viewer** | Enhanced modal with zoom, page navigation, download, and fullscreen |
| **Jina Reranker v2** | Cross-encoder reranking (278M params, 108 languages, 8192 tokens) |
| **Guardrail System** | Fast keyword pre-screening + LLM safety check for off-topic/illegal queries |
| **Background Ingestion** | Async Celery workers with progress tracking (OCR → Chunk → Embed → Store) |
| **Toast Notifications** | Non-intrusive alerts for login, ingestion, errors, session expiry |
| **Responsive UI** | Mobile/tablet/desktop layouts with collapsible panels |
| **Message Timestamps** | Hover-to-reveal timestamps on all chat messages |
| **Help Guide** | Built-in usage guide accessible from the sidebar |
| **Export Tools** | Copy, Download .md, and Export Full Report (PDF) on every response |
| **Docker-Ready** | Full `docker-compose.yml` with production services |
| **Deployment Guide** | Nginx config + HTTPS (Let's Encrypt) + step-by-step VPS deployment |

---

## Architecture Overview

```
┌─────────────┐      ┌──────────────┐     ┌─────────────────────────────┐
│  Flask UI   │──▶   │  FastAPI API │──▶  │   LangGraph Agent (CRAG)    │
│ (Waitress)  │      │  (Uvicorn)   │     │                             │
│  Port 5000  │      │  Port 8000   │     │ Guardrail → Contextualize   │
└─────────────┘      └──────────────┘     │ → Route → Retrieve → Rerank │
                                          │ → Grade → (Rewrite + Web)   │
                                          │ → Generate (streaming)      │
                                          └──────────────┬──────────────┘
                                                         │
                                          ┌──────────────┼──────────────┐
                                          ▼              ▼              ▼
                                    ┌──────────┐  ┌──────────┐  ┌──────────┐
                                    │ ChromaDB │  │PostgreSQL│  │  Tavily  │
                                    │  BGE-M3  │  │   SQL    │  │   Web    │
                                    │ 12K+ vec │  │ 3.6K rows│  │  Search  │
                                    └──────────┘  └──────────┘  └──────────┘
                                          │
                                    ┌─────┴─────┐
                                    │  Jina AI  │
                                    │ Reranker  │
                                    └───────────┘
```

---

## Quick Start

### Prerequisites
- Python 3.12+
- Docker & Docker Compose
- PostgreSQL (or use Docker)
- Redis (or use Docker)

### 1. Clone and configure
```bash
git clone https://github.com/your-user/financial-advisor.git
cd financial-advisor
cp .env.example .env
# Edit .env with your API keys (GROQ_API_KEY, GOOGLE_CLIENT_ID, etc.)
```

### 2. Start infrastructure (Docker)
```bash
docker-compose up -d
```

### 3. Run database migrations
```bash
alembic upgrade head
```

### 4. Start the API (local dev)
```bash
uvicorn api.main:app --reload --port 8000
```

### 5. Start the UI
```bash
cd ui && flask run --port 5000
```

### 6. Open the app
- Chat UI: http://127.0.0.1:5000
- API Docs: http://127.0.0.1:8000/docs
- Flower: http://localhost:5555

---

## Authentication

The app supports two authentication methods:

- **Google OAuth** — Sign in with Google (requires `GOOGLE_CLIENT_ID` in `.env`)
- **Email/Password** — Classic registration and login with bcrypt-hashed passwords

Both methods issue a JWT session token used for all protected API calls.

---

## Data Sources

### Unstructured (ChromaDB — 12,000+ vectors)
| Source | Documents | Method |
|--------|-----------|--------|
| AMMC | ~109 PDFs (reports, guides, circulars) | Playwright scraping |
| Bank Al-Maghrib | ~47 PDFs (monetary reports, stability) | BeautifulSoup |
| HCP | ~80 PDFs (macro bulletins, budgets) | BeautifulSoup |
| Cour des Comptes | ~5 PDFs (audit reports) | BeautifulSoup |
| User Uploads | Variable | Chat UI upload |

### Structured (PostgreSQL — 3,656 rows)
| Table | Content | Source |
|-------|---------|--------|
| `stock_prices` (140 rows) | Daily OHLCV for 28 global tickers | yfinance |
| `financial_metrics` (470 rows) | P/E, EPS, revenue, market cap | yfinance |
| `macro_indicators` (3,046 rows) | VIX, S&P 500, Gold, Oil, FX, yields | yfinance |

### Live Data
| Source | Data | Method |
|--------|------|--------|
| BVC (Bourse de Casablanca) | Real-time stock prices, indices | REST scraping (30s refresh) |
| Tavily | Web search results | API (auto-triggered by CRAG) |

---

## Scripts

| Script | Description |
|--------|-------------|
| `scripts/ingest_missing.py` | Find and ingest PDFs not yet in ChromaDB |
| `scripts/populate_global_finance.py` | Populate PostgreSQL with global financial data |
| `scripts/populate_global_finance.py --update` | Daily update (last 5 days only) |
| `scripts/check_chromadb.py` | Show ChromaDB ingestion status |

---

## Models (Fine-tuning Notebooks)

| Notebook | Purpose | Dataset |
|----------|---------|---------|
| `models/speech_to_text_model.ipynb` | Fine-tune Whisper Tiny for AR/FR/EN | Google FLEURS |
| `models/text_to_sql_model.ipynb` | Fine-tune SQLCoder for financial SQL generation | Spider + custom |

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/auth/register` | POST | Create account (email/password) |
| `/auth/login` | POST | Login (email/password) |
| `/auth/google` | POST | Google OAuth login |
| `/auth/me` | GET | Get current user profile |
| `/query/stream` | POST | Streaming RAG query (SSE) |
| `/query/` | POST | Synchronous RAG query |
| `/ingest/async` | POST | Upload PDF for async ingestion |
| `/ingest/status/{task_id}` | GET | Poll ingestion progress |
| `/speech/transcribe` | POST | Speech-to-text (audio upload) |
| `/bvc/live` | GET | Live BVC stock prices |
| `/bvc/indices` | GET | BVC market indices |
| `/conversations/` | GET/POST | List/create conversations |
| `/conversations/{id}` | GET/PUT/DELETE | Manage conversation |
| `/pdf/file?path=...` | GET | Serve PDF for in-UI viewer |
| `/health` | GET | API health check |

---

## Technology Stack

| Layer | Technology |
|-------|-----------|
| **Primary LLM** | LLaMA 3.3-70B via Groq (free, ultra-fast) |
| **Fast LLM** | LLaMA 3.1-8B via Groq (routing, guardrail, rewriting) |
| **Fallback LLMs** | Qwen 2.5-72B, LLaMA 3.3-70B via OpenRouter |
| **Embeddings** | BAAI/BGE-M3 (1024d, 100+ languages, local) |
| **Reranker** | Jina Reranker v2 (cross-encoder, 278M, 108 langs) |
| **Vector DB** | ChromaDB (HNSW cosine, persistent) |
| **Relational DB** | PostgreSQL 16 |
| **Orchestration** | LangGraph StateGraph (CRAG pattern) |
| **STT** | Groq Whisper Large v3 (API) + fine-tuned Whisper Tiny |
| **Web Search** | Tavily API |
| **Backend** | FastAPI + Uvicorn (SSE streaming) |
| **Frontend** | Flask + Tailwind CSS |
| **Auth** | Google OAuth 2.0 + JWT + bcrypt |
| **Task Queue** | Celery + Redis |
| **Monitoring** | Arize Phoenix + Celery Flower |
| **Deployment** | Docker Compose + Nginx + Let's Encrypt |

---

## Deployment

See `deploy/DEPLOY.md` for full production deployment instructions including:
- VPS setup (Docker + Nginx + HTTPS)
- `deploy/nginx.conf` — ready-to-use reverse proxy config
- Google OAuth production setup
- Database migrations and backups

---

## Project Structure

```
financial-advisor/
├── agents/                  # LangGraph agent, prompts, tools
├── api/                     # FastAPI app + route handlers
│   └── routes/              # auth, query, ingest, speech, bvc, conversations
├── alembic/                 # Database migrations
├── config/                  # Scraper site configurations
├── data/                    # Collected documents + cache
├── deploy/                  # Nginx config + deployment guide
├── ingestion/               # Pipeline: loaders, chunkers, embeddings, dedup
├── models/                  # Fine-tuning notebooks (STT, Text-to-SQL)
├── retrieval/               # Vector search, BM25, reranker, query router, RRF
├── scripts/                 # Utility scripts (ingestion, data population)
├── storage/                 # ChromaDB + PostgreSQL wrappers
├── tasks/                   # Celery task definitions
├── ui/                      # Flask frontend (templates, static JS/CSS)
├── docker-compose.yml       # Production orchestration
├── Dockerfile               # Optimized build (CPU PyTorch)
├── requirements.txt         # Python dependencies
└── .env.example             # Environment template
```

---

## License

MIT License - PFE 2026
