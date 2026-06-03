"""
scripts/ingest_missing.py
Finds all PDF files in data/collected_data/documents that are NOT yet in ChromaDB
and ingests them into the 'financial_data' collection.

Usage:
    python scripts/ingest_missing.py
"""

import sys
import time
from pathlib import Path

# Ensure project root is in path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

import chromadb
from chromadb.config import Settings
from storage.chroma_store import CHROMA_PERSIST_DIR, CHROMA_COLLECTION_FINANCIAL
from ingestion import IngestionPipeline


def get_ingested_filenames() -> set:
    """Get set of filenames already in ChromaDB."""
    client = chromadb.PersistentClient(
        path=CHROMA_PERSIST_DIR,
        settings=Settings(anonymized_telemetry=False),
    )
    coll = client.get_or_create_collection(CHROMA_COLLECTION_FINANCIAL)
    results = coll.get(include=["metadatas"])
    sources = set()
    for m in results["metadatas"]:
        src = m.get("source_file", m.get("source", ""))
        sources.add(Path(src).name)
    return sources


def get_all_pdfs() -> list:
    """Get all PDF files from collected_data."""
    collected_dir = ROOT / "data" / "collected_data" / "documents"
    return sorted(collected_dir.rglob("*.pdf"))


def progress_bar(current, total, width=40, elapsed=0, filename=""):
    """Print a progress bar with ETA."""
    pct = current / total if total > 0 else 0
    filled = int(width * pct)
    bar = "█" * filled + "░" * (width - filled)

    # ETA calculation
    if current > 0 and elapsed > 0:
        avg_per_file = elapsed / current
        remaining = (total - current) * avg_per_file
        if remaining > 3600:
            eta_str = f"{remaining/3600:.1f}h"
        elif remaining > 60:
            eta_str = f"{remaining/60:.0f}min"
        else:
            eta_str = f"{remaining:.0f}s"
    else:
        eta_str = "..."

    # Truncate filename
    short_name = filename[:35] + "..." if len(filename) > 38 else filename

    print(
        f"\r  [{bar}] {current}/{total} ({pct*100:.1f}%) | ETA: {eta_str} | {short_name}     ",
        end="", flush=True
    )


def main():
    print()
    print("═" * 70)
    print("  📄 BATCH INGESTION — Missing Documents → financial_data")
    print("═" * 70)

    ingested = get_ingested_filenames()
    all_pdfs = get_all_pdfs()

    missing = [f for f in all_pdfs if f.name not in ingested]

    print(f"\n  Total PDFs on disk:  {len(all_pdfs)}")
    print(f"  Already ingested:    {len(ingested)}")
    print(f"  To ingest:           {len(missing)}")
    print()

    if not missing:
        print("  All documents are already ingested!")
        return

    pipeline = IngestionPipeline(collection_override=CHROMA_COLLECTION_FINANCIAL)

    success = 0
    failed = 0
    failed_files = []
    start = time.time()

    print("─" * 70)
    for i, pdf_path in enumerate(missing, 1):
        elapsed = time.time() - start
        progress_bar(i - 1, len(missing), elapsed=elapsed, filename=pdf_path.name)

        try:
            result = pipeline.ingest_file(str(pdf_path))
            chunks = result.get("chunks", 0)
            stored = result.get("stored", 0)
            success += 1
            # Clear line and show success
            print(f"\r  ✓ [{i}/{len(missing)}] {pdf_path.name[:50]}  ({chunks} chunks, {stored} stored)     ")
        except Exception as e:
            failed += 1
            failed_files.append((pdf_path.name, str(e)[:60]))
            print(f"\r  ✗ [{i}/{len(missing)}] {pdf_path.name[:50]}  FAILED: {str(e)[:40]}     ")

    # Final progress bar at 100%
    elapsed = time.time() - start
    progress_bar(len(missing), len(missing), elapsed=elapsed, filename="Done!")
    print()

    # Summary
    print()
    print("═" * 70)
    print(f"   COMPLETED in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"     Success: {success} | Failed: {failed} | Total: {len(missing)}")
    print("═" * 70)

    if failed_files:
        print(f"\n    Failed files ({failed}):")
        for fname, err in failed_files:
            print(f"     • {fname[:55]}")
            print(f"       → {err}")
        print()


if __name__ == "__main__":
    main()
