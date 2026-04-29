"""
tasks/local_ingestion_tasks.py
Worker C Task: Scans a local docs folder and ingests new PDF/DOCX files into ChromaDB.

- Skips files already ingested (checks ChromaDB by filename hash).
- Routes new files to the existing `ingest_document` task on the ingestion queue.

Queue: local_ingestion
"""
import hashlib
import logging
from pathlib import Path

from tasks.celery_app import celery_app

logger = logging.getLogger(__name__)

DOCS_DIR = Path("data/docs")
SUPPORTED_EXTENSIONS = {".pdf", ".docx"}


def _already_ingested(filename: str) -> bool:
    """
    Check ChromaDB to see if this file has already been ingested.
    Uses a deterministic ID derived from the filename so re-runs are idempotent.
    """
    from storage.chroma_store import ChromaStore
    store = ChromaStore()
    # The ingest_document task uses md5 of first 120 chars of text as chunk IDs.
    # We track ingestion at the file level by querying metadatas for source_file.
    results = store.collection.get(
        where={"source_file": filename},
        limit=1,
        include=["metadatas"],
    )
    return len(results.get("ids", [])) > 0


@celery_app.task(name="tasks.local_ingestion_tasks.scan_and_ingest_local_docs")
def scan_and_ingest_local_docs(docs_dir: str = None):
    """
    Worker C Entry Task: Scans the local docs folder for PDF/DOCX files.
    Skips files already present in ChromaDB and dispatches new ones to the ingestion queue.
    """
    from tasks.document_ingestion_tasks import ingest_document

    target_dir = Path(docs_dir) if docs_dir else DOCS_DIR

    if not target_dir.exists():
        logger.warning("LOCAL INGESTION: docs folder not found at %s — creating it.", target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        return "No files found — docs folder was empty."

    # Collect all supported files recursively
    files = [
        f for f in target_dir.rglob("*")
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if not files:
        logger.info("LOCAL INGESTION: No PDF/DOCX files found in %s", target_dir)
        return "No PDF/DOCX files found."

    logger.info("LOCAL INGESTION: Found %d file(s) in %s", len(files), target_dir)

    skipped = 0
    dispatched = 0

    for file_path in files:
        filename = file_path.name

        if _already_ingested(filename):
            logger.info("LOCAL INGESTION: Skipping '%s' — already in ChromaDB.", filename)
            skipped += 1
            continue

        logger.info("LOCAL INGESTION: Dispatching '%s' to ingestion queue.", filename)
        ingest_document.apply_async(args=[str(file_path.resolve())], queue="ingestion")
        dispatched += 1

    summary = (
        f"LOCAL INGESTION complete: {dispatched} dispatched, {skipped} skipped (already ingested)."
    )
    logger.info(summary)
    return summary
