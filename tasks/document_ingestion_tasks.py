"""
tasks/document_ingestion_tasks.py
Worker B Task: Ingests financial documents (PDF/DOCX) into ChromaDB.
Universal ingestion pipeline — works for any scraped financial source.
Uses BGE Embedder (BAAI/bge-large-en-v1.5) for semantic embeddings.

Pipeline: Load → OCR (if scanned) → Chunk → BGE Embed → ChromaDB Upsert
Supports Celery progress updates so the UI can poll real-time status.
"""
import os
import sys
import time
import logging
from pathlib import Path

from tasks.celery_app import celery_app

logger = logging.getLogger(__name__)


# ── Pretty terminal progress bar ──────────────────────────────────────────────
# Renders a clean, animated progress bar in the Celery worker console so the
# ingestion of each user document is clear and satisfying to watch.

# Enable ANSI escape sequences on Windows consoles (cmd / PowerShell, Win10+).
if os.name == "nt":
    os.system("")

_TTY = sys.stdout.isatty()


def _c(code: str) -> str:
    """Return an ANSI code only when writing to a real terminal."""
    return code if _TTY else ""


_RESET, _BOLD, _DIM = "\033[0m", "\033[1m", "\033[2m"
_GREEN, _CYAN, _RED = "\033[92m", "\033[96m", "\033[91m"

_STEP_LABELS = {
    "loading":   "Loading",
    "ocr":       "OCR Vision",
    "chunking":  "Chunking",
    "embedding": "Embedding",
    "storing":   "Storing",
    "done":      "Complete",
}


def _progress_header(filename: str) -> None:
    line = "═" * 64
    sys.stdout.write(f"\n{_c(_CYAN)}{line}{_c(_RESET)}\n")
    sys.stdout.write(f"  {_c(_BOLD)}INGESTING{_c(_RESET)}  {filename}\n")
    sys.stdout.write(f"{_c(_CYAN)}{line}{_c(_RESET)}\n")
    sys.stdout.flush()


def _render_bar(label: str, percent: int, detail: str) -> str:
    width = 26
    filled = int(round(width * percent / 100))
    bar = (
        f"{_c(_GREEN)}{'█' * filled}{_c(_RESET)}"
        f"{_c(_DIM)}{'░' * (width - filled)}{_c(_RESET)}"
    )
    detail = (detail or "")[:44]
    return (
        f"  {_c(_BOLD)}{label:<11}{_c(_RESET)}"
        f"{_c(_DIM)}▕{_c(_RESET)}{bar}{_c(_DIM)}▏{_c(_RESET)}"
        f" {_c(_BOLD)}{percent:3d}%{_c(_RESET)}  {_c(_DIM)}{detail}{_c(_RESET)}"
    )


def _emit_progress(task, step: str, detail: str, percent: int, state: dict) -> None:
    """Push progress to Celery (for UI polling) AND animate a terminal bar."""
    # 1. Keep the existing UI contract intact — the web client polls this.
    task.update_state(state="PROGRESS", meta={
        "step": step, "detail": detail, "percent": percent,
    })

    # 2. Animate a satisfying bar in the worker console.
    label = _STEP_LABELS.get(step, step.title())
    start = state.get("pct", 0)
    try:
        if _TTY and percent > start:
            stride = max(1, (percent - start) // 14)
            for p in range(start, percent, stride):
                sys.stdout.write("\r" + _render_bar(label, p, detail))
                sys.stdout.flush()
                time.sleep(0.012)
        # Commit the final state of this stage on its own line.
        sys.stdout.write("\r" + _render_bar(label, percent, detail) + "\n")
        sys.stdout.flush()
    except Exception:
        pass
    state["pct"] = percent


def _progress_footer(filename: str, chunks: int, stored: int, total: int, ocr_used: bool) -> None:
    line = "═" * 64
    badge = f"{_c(_DIM)} (OCR+Vision){_c(_RESET)}" if ocr_used else ""
    sys.stdout.write(
        f"  {_c(_GREEN)}{_c(_BOLD)}✓ DONE{_c(_RESET)}  {filename}{badge} — "
        f"{_c(_BOLD)}{chunks}{_c(_RESET)} chunks · "
        f"{_c(_BOLD)}{stored}{_c(_RESET)} vectors stored "
        f"{_c(_DIM)}({total} total in collection){_c(_RESET)}\n"
    )
    sys.stdout.write(f"{_c(_CYAN)}{line}{_c(_RESET)}\n\n")
    sys.stdout.flush()


def _progress_error(filename: str, msg: str) -> None:
    sys.stdout.write(
        f"  {_c(_RED)}{_c(_BOLD)}✗ FAILED{_c(_RESET)}  {filename} — "
        f"{_c(_RED)}{msg}{_c(_RESET)}\n\n"
    )
    sys.stdout.flush()


@celery_app.task(
    name="tasks.document_ingestion_tasks.ingest_document",
    bind=True,  # gives access to self for progress updates
)
def ingest_document(self, file_path: str, use_ocr: bool = False, collection_name: str = None):
    """
    Worker B Task: Receives a document path from any scraping worker or UI upload.
    Runs the full ingestion pipeline:
      PDF → OCR/Vision (if enabled) → Markdown → Chunk → BGE Embed → ChromaDB Upsert

    Progress updates are pushed via self.update_state() so the UI can poll them.
    """
    from ingestion.embeddings import Embedder
    from storage.chroma_store import ChromaStore

    path = Path(file_path)

    logger.info("=" * 70)
    logger.info("INGESTION WORKER: Received %s", path.name)

    # Per-task progress state (tracks last rendered percent for smooth animation)
    _pstate: dict = {"pct": 0}
    _progress_header(path.name)

    if not path.exists():
        logger.error("INGESTION WORKER: File not found at %s", path)
        _progress_error(path.name, "File not found")
        return {"status": "error", "detail": f"File not found: {path}"}

    # Filter: only process PDFs and DOCX
    if path.suffix.lower() not in (".pdf", ".docx"):
        logger.info(
            "INGESTION WORKER: Skipping %s (not a PDF/DOCX — financial docs only)", path.name
        )
        _progress_error(path.name, "Skipped (not a PDF/DOCX file)")
        return {"status": "skipped", "detail": "Not a PDF/DOCX file"}

    # ── 0. Progress: Starting ────────────────────────────────────────────────
    _emit_progress(self, "loading", f"Loading {path.name}...", 5, _pstate)

    # ── 1. Load text records (with OCR if enabled) ───────────────────────────
    records = []
    ocr_used = False

    if path.suffix.lower() == ".pdf" and use_ocr:
        try:
            _emit_progress(self, "ocr",
                           f"Running Advanced OCR + Vision extraction on {path.name}...",
                           10, _pstate)

            from OCR_Processing.pdf_extractor import AdvancedPDFPipeline
            pipeline = AdvancedPDFPipeline()
            md_path = pipeline.process_pdf(str(path))

            # Read the generated markdown as a single text record
            md_text = Path(md_path).read_text(encoding="utf-8")
            if md_text.strip():
                records = [(md_text, {"source_file": path.name, "ocr": "true"})]
                ocr_used = True
                logger.info("INGESTION WORKER: OCR pipeline produced %d chars from %s", len(md_text), path.name)

        except Exception as e:
            logger.warning("INGESTION WORKER: OCR failed for %s: %s. Falling back to basic loader.", path.name, e)

    # Fallback: basic loader if OCR wasn't used or failed
    if not records:
        try:
            _emit_progress(self, "loading",
                           f"Extracting text from {path.name} (basic loader)...",
                           15, _pstate)

            if path.suffix.lower() == ".pdf":
                from ingestion.loaders.pdf_loader import PDFLoader
                records = PDFLoader().load(str(path))
            elif path.suffix.lower() == ".docx":
                from ingestion.loaders.docx_loader import DocxLoader
                records = DocxLoader().load(str(path))
        except Exception as e:
            logger.error("INGESTION WORKER: Loader failed for %s: %s", path.name, e)
            _progress_error(path.name, f"Loader failed: {e}")
            return {"status": "error", "detail": f"Loader failed: {e}"}

    if not records:
        logger.warning("INGESTION WORKER: No text content extracted from %s", path.name)
        _progress_error(path.name, "No text content extracted")
        return {"status": "error", "detail": "No text content extracted"}

    logger.info("INGESTION WORKER: Extracted %d text records from %s", len(records), path.name)

    # ── 2. Chunk ─────────────────────────────────────────────────────────────
    _emit_progress(self, "chunking",
                   f"Chunking {len(records)} records...", 40, _pstate)

    try:
        # Use RecursiveChunker for user uploads — it keeps table headers with
        # their data rows (semantic chunker splits tables into isolated lines).
        from ingestion.chunkers.recursive import RecursiveChunker
        chunker = RecursiveChunker(chunk_size=800, chunk_overlap=100)
        all_chunks = []
        for text, meta in records:
            meta["source_file"] = path.name
            meta["source"] = "financial_scraping_pipeline"
            meta["file_path"] = str(path.absolute())
            if ocr_used:
                meta["extraction_method"] = "ocr_vision"
            chunks = chunker.chunk(text, meta)
            all_chunks.extend(chunks)
    except Exception as e:
        logger.error("INGESTION WORKER: Chunking failed for %s: %s", path.name, e)
        _progress_error(path.name, f"Chunking failed: {e}")
        return {"status": "error", "detail": f"Chunking failed: {e}"}

    if not all_chunks:
        logger.warning("INGESTION WORKER: No chunks produced from %s", path.name)
        _progress_error(path.name, "No chunks produced")
        return {"status": "error", "detail": "No chunks produced"}

    logger.info("INGESTION WORKER: %d chunks created from %s", len(all_chunks), path.name)

    # ── 3. Embed with BGE ────────────────────────────────────────────────────
    _emit_progress(self, "embedding",
                   f"Generating BGE embeddings for {len(all_chunks)} chunks...",
                   65, _pstate)

    try:
        embedder = Embedder()
        texts = [c[0] for c in all_chunks]
        metadatas = [c[1] for c in all_chunks]
        embeddings = embedder.embed(texts)
    except Exception as e:
        logger.error("INGESTION WORKER: Embedding failed for %s: %s", path.name, e)
        _progress_error(path.name, f"Embedding failed: {e}")
        return {"status": "error", "detail": f"Embedding failed: {e}"}

    logger.info("INGESTION WORKER: BGE embeddings generated (%d vectors)", len(embeddings))

    # ── 4. Upsert into ChromaDB ───────────────────────────────────────────────
    _emit_progress(self, "storing",
                   f"Upserting {len(texts)} vectors into ChromaDB...", 85, _pstate)

    try:
        target_collection = collection_name or None  # None = default from env
        store = ChromaStore(collection_name=target_collection) if target_collection else ChromaStore()
        stored = store.upsert(texts=texts, embeddings=embeddings, metadatas=metadatas)
    except Exception as e:
        logger.error("INGESTION WORKER: ChromaDB upsert failed for %s: %s", path.name, e)
        _progress_error(path.name, f"ChromaDB upsert failed: {e}")
        return {"status": "error", "detail": f"ChromaDB upsert failed: {e}"}

    # ── 5. Done ──────────────────────────────────────────────────────────────
    _emit_progress(self, "done", f"Successfully ingested {path.name}!", 100, _pstate)

    total_docs = store.count()
    _progress_footer(path.name, len(all_chunks), stored, total_docs, ocr_used)

    logger.info(
        "INGESTION WORKER: Successfully committed %d chunks from %s → ChromaDB (%d total docs)",
        stored,
        path.name,
        total_docs,
    )
    logger.info("=" * 70)

    return {
        "status": "success",
        "filename": path.name,
        "chunks": len(all_chunks),
        "stored": stored,
        "total_docs": total_docs,
        "ocr_used": ocr_used,
    }
