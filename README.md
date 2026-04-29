# AI Financial Advisor

A distributed Hybrid RAG (Retrieval-Augmented Generation) platform for financial analysis of Moroccan public finance institutions. Combines vector search over scraped documents with SQL queries over structured financial data to deliver structured insights via natural language.

**PFE 2026 - Final Year Project | Field: AI / FinTech | Author: M.Bouada - PFE2026a.lahrache**

---

## Architecture Overview

![Architecture Overview](architecture_diagram.png)

---

## Quick Start

### Important Note on Data & PDFs
> **Note:** The `data/` directory (containing raw downloaded PDFs) is ignored by git to keep the repository lightweight. You don't need to manually upload or push the PDFs! When you run the Celery workers locally (see step 3), they will automatically scrape, download, and save the PDFs into their associated folders under `data/`, and then ingest them into ChromaDB seamlessly.

### 1. Setup environment
```bash
cd "Pfe 2026/financial-advisor"
pip install -r requirements.txt
# Edit .env with your keys (see Configuration section)
```

### 2. Start the API
```bash
uvicorn api.main:app --reload
# Swagger UI: http://localhost:8000/docs
```

### 3. Start Celery Workers (3 separate terminals)
```bash
# Worker A — Web scraping (CDC, HCP, config-driven sites)
celery -A tasks.celery_app worker -Q scraping --pool=solo -n scraping_worker@%h --loglevel=INFO

# Worker B — Document ingestion into ChromaDB
celery -A tasks.celery_app worker -Q ingestion --pool=solo -n ingestion_worker@%h --loglevel=INFO

# Worker C — Local docs folder ingestion
celery -A tasks.celery_app worker -Q local_ingestion --pool=solo -n local_worker@%h --loglevel=INFO
```

### 4. Trigger the pipeline
```bash
python scripts/trigger_all.py              # Trigger all workers
python scripts/trigger_all.py --cdc-only   # CDC only
python scripts/trigger_all.py --hcp-only   # HCP only
python scripts/trigger_all.py --local-only # Local docs only
python scripts/trigger_all.py --sites-only # Config-driven sites (sites.json)
```

### 5. Monitor with Flower
```bash
python scripts/run_flower.py
# Open http://localhost:5555 — shows all 3 workers
```

---

## Adding a New Data Source

### Option 1 — Config-driven (recommended, no code needed)
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

### Option 2 — Drop files manually
Place any PDF or DOCX in `data/docs/` and run:
```bash
python scripts/trigger_all.py --local-only
```
Worker C will skip already-ingested files automatically.

### Option 3 — Write a dedicated scraping task
Copy `tasks/web_scraping_tasks.py`, adapt to the new site's HTML structure, register in `celery_config.py` and `tasks/__init__.py`.

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `POST /query/` | POST | Natural language financial question |
| `GET /stocks/{ticker}` | GET | Stock price snapshot |
| `GET /stocks/{ticker}/history` | GET | Historical prices |
| `GET /macro/{country}/{indicator}` | GET | Macro indicators |
| `GET /metrics/{company}` | GET | Financial metrics |

### Query example (PowerShell)
```powershell
Invoke-RestMethod -Uri "http://localhost:8000/query/" -Method POST `
  -ContentType "application/json" `
  -Body '{"question": "quel est le bilan de la cour des comptes 2023 ?", "top_k": 5}'
```

### Query response structure
```json
{
  "question": "...",
  "answer": "### Direct Answer\n...\n### Key Figures\n...\n### Evidence from Documents\n...\n### Limitations\n...",
  "context": "..."
}
```

---


## Technology Stack

| Component | Technology |
|-----------|-----------|
| LLM | meta-llama/llama-3.1-8b-instruct via OpenRouter |
| Embeddings | BAAI/bge-large-en-v1.5 (1024 dims) |
| Vector DB | ChromaDB (local persistent) |
| Relational DB | PostgreSQL |
| Task Queue | Celery + Redis (Upstash) |
| Orchestration | LangGraph StateGraph |
| Web Scraping | Playwright + BeautifulSoup |
| API | FastAPI + Swagger UI |
| Monitoring | Celery Flower |

---

## Configuration

Key environment variables in `.env`:

```bash
# LLM (OpenRouter)
OPENROUTER_API_KEY=sk-or-v1-...
OPENROUTER_MODEL=meta-llama/llama-3.1-8b-instruct

# ChromaDB
CHROMA_PERSIST_DIR=./chroma_db
CHROMA_COLLECTION=financial_data

# Embeddings
EMBEDDING_MODEL=BAAI/bge-large-en-v1.5
EMBEDDING_DIMENSION=1024

# PostgreSQL
DATABASE_URL=postgresql+psycopg2://postgres:root@localhost:5432/financial_advisor

# Redis / Celery (Upstash)
REDIS_HOST=your-redis-host.upstash.io
REDIS_PASSWORD=your_redis_password
USE_CELERY=true
```

---

## Diagnostic Tools

```bash
# Check ChromaDB ingestion status (works from any directory)
python scripts/check_chromadb.py

# Run API unit tests
pytest tests/test_api.py -v
```

---

## License

MIT License - PFE 2026
