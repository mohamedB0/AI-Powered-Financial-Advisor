# AI Financial Advisor — Project Progress Snapshot
> **PFE 2026** | Last updated: 2026-05-11 | Workspace: `c:\Users\lenovo\Desktop\Pfe 2026\financial-advisor`

---

## 🏗️ Architecture Overview

Two separate servers run side-by-side:

| Server | Tech | Port | Entry Point |
|--------|------|------|-------------|
| **FastAPI Backend** | FastAPI + Uvicorn | **8000** | `fastapi dev api/main.py` |
| **Flask Frontend UI** | Flask + Jinja2 | **5000** | `python ui/app.py` |

The Flask UI is a thin template renderer — all AI logic lives in the FastAPI backend. The frontend JavaScript hits `http://localhost:8000` for all data (queries, ingestion, BVC market data).

---

## ✅ Fully Implemented & Working

### 1. LangGraph RAG Agent (`agents/financial_agent.py`)
- **CRAG loop**: Guardrail → Contextualize → Retrieve → Grade → (Rewrite + Web Fallback if weak) → Generate
- **LLM stack**: Groq primary (`llama-3.3-70b-versatile`), OpenRouter fallbacks (`qwen-2.5-72b`, `llama-3.3-70b`, `llama-3.1-8b-free`)
- **Parallel execution**: Guardrail check + retrieval run concurrently via `ThreadPoolExecutor`
- **Guardrail**: Fast keyword pre-screening (~90% bypass LLM call), LLM-based check for edge cases
- **Contextualizer**: Pronoun/reference resolution using conversation history (fast-pass when no history)
- **Grader**: Heuristic-based (trusts Jina reranker score), avoids LLM grader over-aggression on French/Arabic
- **Rewriter**: Rewrites queries when context is weak, triggers web search fallback
- **Focus Modes**: "New Chat" (router decides), "Company Analysis" (forces SQL), "Market Macro" (forces Vector)
- **Persona system**: Balanced / Prudent / Aggressive — injected into system prompt

### 2. Retrieval System (`retrieval/`)
| File | Role |
|------|------|
| `query_router.py` | Semantic router → SQL / Vector / Web / Hybrid |
| `vector_search.py` | ChromaDB search (`financial_data` collection, BGE-M3 embeddings) |
| `sql_search.py` | PostgreSQL price history, top movers, revenue queries |
| `reranker.py` | Jina Reranker API — scores and re-orders vector results |
| `hybrid_merger.py` | Merges SQL + Vector + Web results into formatted context string |
| `web_search.py` | Tavily web search fallback |
| `semantic_cache.py` | ChromaDB-backed semantic similarity cache |
| `bvc_scraper.py` | **Live** Casablanca Bourse scraper (no auth required) |

### 3. FastAPI Routes (`api/routes/`)
| Route | Description |
|-------|-------------|
| `POST /query/stream` | **Main chat endpoint** — SSE token streaming, full CRAG pipeline |
| `POST /query/` | Non-streaming version |
| `GET /bvc/live` | Live BVC stock prices (113 instruments, proxied from casablanca-bourse.com) |
| `GET /bvc/indices` | Live index values (34 indices: MASI, MADEX, MASI 20, MASI ESG…) |
| `GET /bvc/prices` | Historical prices from PostgreSQL |
| `POST /bvc/refresh` | Trigger Celery BVC data fetch task |
| `POST /ingest/async` | Upload single doc → Celery task |
| `POST /ingest/batch` | Upload multiple docs → Celery tasks |
| `GET /ingest/status/{id}` | Poll Celery task progress |
| `GET /pdf/file` | Serve PDF files with path resolution for Windows/Docker |
| `GET /health` | System health check (PostgreSQL + ChromaDB) |

### 4. Chat UI (`ui/templates/chat.html` + `ui/static/js/main.js`)
- **Side-by-side layout**: Chat column (left) + Live Market Panel (right, 380px, toggleable)
- **Streaming chat**: SSE reader renders tokens as they arrive, with "Thinking…" spinner
- **Focus mode tabs**: New Chat / Company Analysis / Market Macro
- **Prompt suggestion buttons**: Click to fill input
- **Retrieved context accordion**: Collapsible, shows route label (Vector / SQL / Web Search…)
- **Keyword highlighting**: User query keywords highlighted in retrieved context blockquotes
- **PDF modal**: Click source links → inline PDF viewer modal
- **Export**: Copy to clipboard, Download .md, Export full session report
- **Settings modal**: Persona, RAG depth (top_k), OpenRouter API key override
- **File upload + ingestion progress**: Animated pipeline steps (Upload → OCR → Chunk → Embed → Store)
- **Upload resume**: `localStorage` persists active task ID across page navigation

### 5. Live Market Panel (BVC)
- **Auto-refresh**: Every 30 seconds
- **Indices strip**: Shows first 8 indices (MASI, MASI 20, MASI ESG…) with value + variation%
- **Tabs**: Tous (all) / Hausse (gainers, sorted ↓) / Baisse (losers, sorted ↑) / Actif (top-30 by volume)
- **Summary bar**: Total instruments | Gainers ↑ | Losers ↓ | Flat =
- **Manual refresh button** + last-update timestamp
- **Toggle button**: Hides/shows the 380px panel with CSS transition
- **Error state**: Shows retry button if BVC endpoint fails

### 6. Vector Database
- **Engine**: ChromaDB (local, persistent at `chroma_db/`)
- **Collection**: `financial_data`
- **Embeddings**: BGE-M3 (multilingual: French + Arabic + English)
- **Current size**: ~12,111 vectors
- **Sources ingested**: AMMC annual reports, Bank Al-Maghrib data, HCP macro indicators

### 7. PostgreSQL (`storage/relational_db.py`)
- Tables: `stock_prices`, `financial_metrics`, `macro_indicators`
- Connected and healthy (verified via `/health` endpoint)

### 8. Celery Workers (`tasks/`)
| Task file | Queue | Purpose |
|-----------|-------|---------|
| `document_ingestion_tasks.py` | `user_ingestion` | User-uploaded doc processing (OCR → Chunk → Embed → Store) |
| `bvc_fetch_tasks.py` | `bvc` | Periodic BVC data fetch to PostgreSQL |
| `data_fetch_tasks.py` | `ingestion` | Background financial data scraping |
| `config_driven_scraping_tasks.py` | `ingestion` | Config-driven web scraping pipeline |

---

## 🐛 Bugs Fixed in This Session (2026-05-11)

### Bug 1 — `const API_BASE` redeclaration crash
- **File**: `ui/templates/chat.html`
- **Problem**: `main.js` declares `const API_BASE` globally. The inline `<script>` in `chat.html` added for the market panel also declared `const API_BASE` at global scope → `SyntaxError` → entire JS engine crashed → chat + market panel both dead
- **Fix**: Removed the duplicate `const API_BASE` from `chat.html`. Added comment noting that `main.js` already owns it.

### Bug 2 — Chat messages invisible (empty-state never hidden)
- **File**: `ui/static/js/main.js`
- **Problem**: `sendQuestion()` called `addMessage()` which appended DOM nodes to `#chat-history` — but that div has `class="hidden"` by default. The old inline `submitMsg()` (removed in Bug 1 fix) was responsible for hiding `#empty-state` and removing `hidden` from `#chat-history`. After removing it, the API was responding correctly but nothing was visible.
- **Fix**: Added empty-state toggle at the top of `addMessage()`:
  ```js
  const emptyState = document.getElementById('empty-state');
  if (emptyState) emptyState.style.display = 'none';
  chatHistory.classList.remove('hidden');
  chatHistory.classList.add('flex');
  ```

### Bug 3 — Prompt buttons no longer filled input
- **File**: `ui/static/js/main.js`
- **Problem**: The `.prompt-btn` click handlers were in the removed inline block.
- **Fix**: Added `document.querySelectorAll('.prompt-btn').forEach(...)` inside `initChat()` in `main.js`.

### Bug 4 — Clear History didn't restore empty-state
- **File**: `ui/static/js/main.js`
- **Problem**: `clearBtn` listener only cleared `chatHistory.innerHTML` but left the empty-state hidden.
- **Fix**: Clear button now also re-hides `chatHistory`, removes `flex`, and restores `emptyState.style.display = ''`.

### Bug 5 — `get_market_summary()` dropped null-variation stocks
- **File**: `retrieval/bvc_scraper.py`
- **Problem**: `if s["variation_pct"] and float(...) == 0` evaluated to `False` for `None` values, so untraded stocks weren't counted in any bucket → `gainers + losers + stable < total_instruments`
- **Fix**: Changed to `if s["variation_pct"] is not None and float(...) == 0`; null stocks folded into the stable bucket.

---

## 📁 Key File Map

```
financial-advisor/
├── api/
│   ├── main.py                    # FastAPI app, router registration, BGE-M3 preload
│   └── routes/
│       ├── query.py               # /query + /query/stream (SSE)
│       ├── bvc.py                 # /bvc/live, /bvc/indices, /bvc/prices, /bvc/refresh
│       ├── ingest.py              # /ingest/async, /ingest/batch, /ingest/status
│       └── pdf_viewer.py          # /pdf/file
├── agents/
│   ├── financial_agent.py         # LangGraph CRAG agent (main brain)
│   ├── prompts.py                 # SYSTEM_PROMPT, QUERY_PROMPT, GRADER, REWRITE, etc.
│   └── tools.py                   # LangGraph tool definitions
├── retrieval/
│   ├── query_router.py            # Routes query to SQL/Vector/Web/Hybrid
│   ├── vector_search.py           # ChromaDB search
│   ├── sql_search.py              # PostgreSQL queries
│   ├── reranker.py                # Jina Reranker API
│   ├── hybrid_merger.py           # Context merging + formatting
│   ├── web_search.py              # Tavily search
│   ├── semantic_cache.py          # ChromaDB semantic similarity cache
│   └── bvc_scraper.py             # Live BVC scraper (casablanca-bourse.com)
├── ui/
│   ├── app.py                     # Flask app (port 5000)
│   ├── templates/
│   │   ├── base.html              # Sidebar, settings modal, Tailwind config
│   │   ├── chat.html              # Chat UI + Live Market Panel HTML/CSS
│   │   ├── dashboard.html         # Stock charts dashboard
│   │   └── pipeline.html          # Upload history page
│   └── static/js/
│       └── main.js                # ALL JS: initChat(), initDashboard(), initPipeline()
├── tasks/
│   ├── document_ingestion_tasks.py # User doc ingestion (OCR, chunking, embedding)
│   ├── bvc_fetch_tasks.py          # BVC Celery task
│   └── data_fetch_tasks.py        # Background data fetch tasks
├── ingestion/
│   └── embeddings.py              # BGE-M3 Embedder (2.2GB model)
├── storage/
│   ├── relational_db.py           # PostgreSQL wrapper
│   └── chroma_store.py            # ChromaDB wrapper
├── .env                           # GROQ_API_KEY, OPENROUTER_API_KEY, TAVILY_API_KEY, DB_URL
└── requirements.txt
```

---

## ⚠️ Known Limitations / TODOs

- **SQL disabled in streaming path**: `need_sql = False` is hardcoded in `retrieve_context()` (line 219 of `financial_agent.py`). Re-enable when SQL data is reliable.
- **BVC scraper uses undocumented API**: `casablanca-bourse.com/api/proxy/...` — could break if the site updates. No authentication needed currently.
- **No user authentication**: The system is single-user, no login/session management.
- **No conversation persistence**: Chat history lives in JS memory only — refreshing the page loses history.
- **Celery broker**: Uses Redis (must be running). SQLite used for result backend.
- **BGE-M3 model**: 2.2GB, pre-loaded at FastAPI startup via `lifespan()`. First boot is slow.
- **Dashboard page**: Still uses old `/api/dashboard/*` routes (separate from FastAPI BVC routes).

---

## 🔧 How to Start Everything

```powershell
# 1. FastAPI backend (port 8000)
fastapi dev api/main.py

# 2. Flask frontend (port 5000)
python ui/app.py

# 3. Celery worker for user doc ingestion
celery -A tasks.celery_app worker -Q user_ingestion -c 1 --loglevel=info

# 4. (Optional) Celery worker for BVC/data tasks
celery -A tasks.celery_app worker -Q bvc,ingestion -c 2 --loglevel=info
```

Access the UI at: **http://localhost:5000**

---

## 📊 Current System Health (as of 2026-05-11 ~17:20 GMT+1)

| Check | Status |
|-------|--------|
| FastAPI `/health` | ✅ Healthy |
| PostgreSQL | ✅ Connected |
| ChromaDB | ✅ 12,111 vectors |
| `/bvc/live` | ✅ 113 instruments live |
| `/bvc/indices` | ✅ 34 indices (MASI: 18,850 @ -0.53%) |
| `/query/stream` | ✅ Streaming, 359 tokens for test query |
| Chat UI | ✅ Fixed (empty-state toggle, JS crash resolved) |
