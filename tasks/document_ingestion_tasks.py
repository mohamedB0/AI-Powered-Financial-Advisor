"""
tasks/document_ingestion_tasks.py
Worker B Task: Ingests financial documents (PDF/DOCX) into ChromaDB.
Universal ingestion pipeline — works for any scraped financial source.
Uses BGE Embedder (BAAI/bge-large-en-v1.5) for semantic embeddings.
NOTE: Uses ChromaDB exclusively.
"""
import logging
from pathlib import Path

from tasks.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(name="tasks.document_ingestion_tasks.ingest_document")
def ingest_document(file_path: str):
    """
    Worker B Task: Receives a document path from any scraping worker.
    Runs the full ingestion pipeline: load -> chunk -> BGE embed -> ChromaDB upsert.
    Only processes PDFs and DOCX files — skips .txt plaintext fallback files.
    """
    from ingestion.embeddings import Embedder
    from ingestion.chunkers.semantic import SemanticChunker
    from storage.chroma_store import ChromaStore

    path = Path(file_path)

    logger.info("=" * 70)
    logger.info("INGESTION WORKER: Received %s", path.name)

    if not path.exists():
        logger.error("INGESTION WORKER: File not found at %s", path)
        return

    # Filter: only process PDFs and DOCX
    if path.suffix.lower() not in (".pdf", ".docx"):
        logger.info(
            "INGESTION WORKER: Skipping %s (not a PDF/DOCX — financial docs only)", path.name
        )
        return

    # ── 1. Load text records ──────────────────────────────────────────────────
    try:
        records = []
        if path.suffix.lower() == ".pdf":
            from ingestion.loaders.pdf_loader import PDFLoader
            records = PDFLoader().load(str(path))
        elif path.suffix.lower() == ".docx":
            from ingestion.loaders.docx_loader import DocxLoader
            records = DocxLoader().load(str(path))
    except Exception as e:
        logger.error("INGESTION WORKER: Loader failed for %s: %s", path.name, e)
        return

    if not records:
        logger.warning("INGESTION WORKER: No text content extracted from %s", path.name)
        return

    logger.info("INGESTION WORKER: Extracted %d text records from %s", len(records), path.name)

    # ── 2. Chunk ─────────────────────────────────────────────────────────────
    try:
        chunker = SemanticChunker()  # BAAI/bge-large-en-v1.5 for topic-boundary splitting
        all_chunks = []
        for text, meta in records:
            meta["source_file"] = path.name
            meta["source"] = "financial_scraping_pipeline"
            chunks = chunker.chunk(text, meta)
            all_chunks.extend(chunks)
    except Exception as e:
        logger.error("INGESTION WORKER: Chunking failed for %s: %s", path.name, e)
        return

    if not all_chunks:
        logger.warning("INGESTION WORKER: No chunks produced from %s", path.name)
        return

    logger.info("INGESTION WORKER: %d chunks created from %s", len(all_chunks), path.name)

    # ── 3. Embed with BGE ────────────────────────────────────────────────────
    try:
        embedder = Embedder()
        texts = [c[0] for c in all_chunks]
        metadatas = [c[1] for c in all_chunks]
        embeddings = embedder.embed(texts)
    except Exception as e:
        logger.error("INGESTION WORKER: Embedding failed for %s: %s", path.name, e)
        return

    logger.info("INGESTION WORKER: BGE embeddings generated (%d vectors)", len(embeddings))

    # ── 4. Upsert into ChromaDB ───────────────────────────────────────────────
    try:
        store = ChromaStore()
        stored = store.upsert(texts=texts, embeddings=embeddings, metadatas=metadatas)
    except Exception as e:
        logger.error("INGESTION WORKER: ChromaDB upsert failed for %s: %s", path.name, e)
        return

    logger.info(
        "INGESTION WORKER: Successfully committed %d chunks from %s -> ChromaDB (%d total docs)",
        stored,
        path.name,
        store.count(),
    )
    logger.info("=" * 70)
