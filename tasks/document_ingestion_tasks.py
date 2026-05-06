"""
tasks/document_ingestion_tasks.py
Worker B Task: Ingests financial documents (PDF/DOCX) into ChromaDB.
Universal ingestion pipeline — works for any scraped financial source.
Uses BGE Embedder (BAAI/bge-large-en-v1.5) for semantic embeddings.

Pipeline: Load → OCR (if scanned) → Chunk → BGE Embed → ChromaDB Upsert
Supports Celery progress updates so the UI can poll real-time status.
"""
import logging
from pathlib import Path

from tasks.celery_app import celery_app

logger = logging.getLogger(__name__)


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
    from ingestion.chunkers.semantic import SemanticChunker
    from storage.chroma_store import ChromaStore

    path = Path(file_path)

    logger.info("=" * 70)
    logger.info("INGESTION WORKER: Received %s", path.name)

    if not path.exists():
        logger.error("INGESTION WORKER: File not found at %s", path)
        return {"status": "error", "detail": f"File not found: {path}"}

    # Filter: only process PDFs and DOCX
    if path.suffix.lower() not in (".pdf", ".docx"):
        logger.info(
            "INGESTION WORKER: Skipping %s (not a PDF/DOCX — financial docs only)", path.name
        )
        return {"status": "skipped", "detail": "Not a PDF/DOCX file"}

    # ── 0. Progress: Starting ────────────────────────────────────────────────
    self.update_state(state="PROGRESS", meta={
        "step": "loading",
        "detail": f"Loading {path.name}...",
        "percent": 5,
    })

    # ── 1. Load text records (with OCR if enabled) ───────────────────────────
    records = []
    ocr_used = False

    if path.suffix.lower() == ".pdf" and use_ocr:
        try:
            self.update_state(state="PROGRESS", meta={
                "step": "ocr",
                "detail": f"Running Advanced OCR + Vision extraction on {path.name}...",
                "percent": 10,
            })

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
            self.update_state(state="PROGRESS", meta={
                "step": "loading",
                "detail": f"Extracting text from {path.name} (basic loader)...",
                "percent": 15,
            })

            if path.suffix.lower() == ".pdf":
                from ingestion.loaders.pdf_loader import PDFLoader
                records = PDFLoader().load(str(path))
            elif path.suffix.lower() == ".docx":
                from ingestion.loaders.docx_loader import DocxLoader
                records = DocxLoader().load(str(path))
        except Exception as e:
            logger.error("INGESTION WORKER: Loader failed for %s: %s", path.name, e)
            return {"status": "error", "detail": f"Loader failed: {e}"}

    if not records:
        logger.warning("INGESTION WORKER: No text content extracted from %s", path.name)
        return {"status": "error", "detail": "No text content extracted"}

    logger.info("INGESTION WORKER: Extracted %d text records from %s", len(records), path.name)

    # ── 2. Chunk ─────────────────────────────────────────────────────────────
    self.update_state(state="PROGRESS", meta={
        "step": "chunking",
        "detail": f"Semantic chunking {len(records)} records...",
        "percent": 40,
    })

    try:
        chunker = SemanticChunker()
        all_chunks = []
        for text, meta in records:
            meta["source_file"] = path.name
            meta["source"] = "financial_scraping_pipeline"
            if ocr_used:
                meta["extraction_method"] = "ocr_vision"
            chunks = chunker.chunk(text, meta)
            all_chunks.extend(chunks)
    except Exception as e:
        logger.error("INGESTION WORKER: Chunking failed for %s: %s", path.name, e)
        return {"status": "error", "detail": f"Chunking failed: {e}"}

    if not all_chunks:
        logger.warning("INGESTION WORKER: No chunks produced from %s", path.name)
        return {"status": "error", "detail": "No chunks produced"}

    logger.info("INGESTION WORKER: %d chunks created from %s", len(all_chunks), path.name)

    # ── 3. Embed with BGE ────────────────────────────────────────────────────
    self.update_state(state="PROGRESS", meta={
        "step": "embedding",
        "detail": f"Generating BGE embeddings for {len(all_chunks)} chunks...",
        "percent": 65,
    })

    try:
        embedder = Embedder()
        texts = [c[0] for c in all_chunks]
        metadatas = [c[1] for c in all_chunks]
        embeddings = embedder.embed(texts)
    except Exception as e:
        logger.error("INGESTION WORKER: Embedding failed for %s: %s", path.name, e)
        return {"status": "error", "detail": f"Embedding failed: {e}"}

    logger.info("INGESTION WORKER: BGE embeddings generated (%d vectors)", len(embeddings))

    # ── 4. Upsert into ChromaDB ───────────────────────────────────────────────
    self.update_state(state="PROGRESS", meta={
        "step": "storing",
        "detail": f"Upserting {len(texts)} vectors into ChromaDB...",
        "percent": 85,
    })

    try:
        target_collection = collection_name or None  # None = default from env
        store = ChromaStore(collection_name=target_collection) if target_collection else ChromaStore()
        stored = store.upsert(texts=texts, embeddings=embeddings, metadatas=metadatas)
    except Exception as e:
        logger.error("INGESTION WORKER: ChromaDB upsert failed for %s: %s", path.name, e)
        return {"status": "error", "detail": f"ChromaDB upsert failed: {e}"}

    # ── 5. Done ──────────────────────────────────────────────────────────────
    self.update_state(state="PROGRESS", meta={
        "step": "done",
        "detail": f"Successfully ingested {path.name}!",
        "percent": 100,
    })

    logger.info(
        "INGESTION WORKER: Successfully committed %d chunks from %s → ChromaDB (%d total docs)",
        stored,
        path.name,
        store.count(),
    )
    logger.info("=" * 70)

    return {
        "status": "success",
        "filename": path.name,
        "chunks": len(all_chunks),
        "stored": stored,
        "total_docs": store.count(),
        "ocr_used": ocr_used,
    }
