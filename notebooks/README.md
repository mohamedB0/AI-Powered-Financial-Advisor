# Notebooks — Financial Advisor Pipelines

Interactive Jupyter notebooks for running and exploring each pipeline in the project.

## Notebooks

| # | Notebook | Description |
|---|----------|-------------|
| 1 | `01_document_ingestion.ipynb` | Ingest PDFs, DOCX, XLSX into ChromaDB (full RAG pipeline) |
| 2 | `02_bvc_market_data.ipynb` | Fetch live Moroccan stock market data from Casablanca Bourse |
| 3 | `03_global_stocks.ipynb` | Download global stock prices & financial metrics (Yahoo Finance) |
| 4 | `04_web_scraping.ipynb` | Config-driven scraping of Moroccan financial institutions |
| 5 | `05_rag_query.ipynb` | Run queries through the Hybrid RAG pipeline (Router → Retrieve → Generate) |
| 6 | `06_ocr_pipeline.ipynb` | Advanced OCR + Vision extraction from scanned PDFs |

## Prerequisites

1. Make sure you're in the project root:
   ```bash
   cd financial-advisor
   ```

2. Install Jupyter if not already installed:
   ```bash
   pip install jupyter ipykernel
   ```

3. Create a kernel for the project's virtual environment:
   ```bash
   python -m ipykernel install --user --name=financial-advisor
   ```

4. Ensure `.env` is configured (copy from `.env.example` if needed).

5. Start Jupyter:
   ```bash
   jupyter notebook notebooks/
   ```

## Memory Requirements

Some cells load the `BAAI/bge-m3` embedding model (2.2 GB, multilingual) into RAM.
These cells require a machine with at least 8 GB of free RAM.

Affected cells:
- `01_document_ingestion.ipynb` — sections 2-6 (ingestion + embedding)
- `05_rag_query.ipynb` — section 2 (vector search) and section 4 (full agent)
- `06_ocr_pipeline.ipynb` — section 4 (ingest OCR output into ChromaDB)

Notebooks that work without the embedding model (no extra RAM needed):
- `02_bvc_market_data.ipynb` — fully functional
- `03_global_stocks.ipynb` — fully functional
- `04_web_scraping.ipynb` — fully functional
- `05_rag_query.ipynb` — sections 1, 3, 5, 6 (router, SQL search, conversation, web search)
- `06_ocr_pipeline.ipynb` — sections 1-3 (PDF extraction + OCR)

If the kernel crashes, close memory-heavy applications and retry,
or run those notebooks on a machine with more RAM.

Each notebook loads `.env` from the project root. Key variables:

- `DATABASE_URL` — PostgreSQL connection string
- `GROQ_API_KEY` — Groq inference (primary LLM)
- `OPENROUTER_API_KEY` — OpenRouter fallback
- `CHROMA_COLLECTION` — Default ChromaDB collection
- `REDIS_HOST` / `REDIS_PORT` — For Celery tasks (optional in notebooks)
- `GEMINI_API_KEY` — For OCR Vision pipeline
