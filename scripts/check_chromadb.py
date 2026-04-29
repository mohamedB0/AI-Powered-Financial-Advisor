"""
scripts/check_chromadb.py
Quick diagnostic to verify ChromaDB ingestion status.

Usage:
    python scripts/check_chromadb.py
"""

import sys
from pathlib import Path

# Always resolve paths relative to the project root, not the CWD
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from storage.chroma_store import ChromaStore
from collections import Counter

def main():
    print("=" * 60)
    print("  ChromaDB Ingestion Status Report")
    print("=" * 60)

    store = ChromaStore()
    total = store.count()

    if total == 0:
        print("\n[WARNING] Collection is EMPTY — no documents have been ingested yet.")
        print("  Make sure the Celery ingestion worker has processed at least one document.")
        return

    print(f"\n[OK] Total chunks stored : {total}")

    # Peek at real documents stored
    results = store.collection.get(limit=min(total, 1000), include=["metadatas", "documents"])
    metadatas = results.get("metadatas", [])
    documents = results.get("documents", [])

    # Source file breakdown
    source_counter = Counter(m.get("source_file", "unknown") for m in metadatas)
    print(f"\n[OK] Unique source files : {len(source_counter)}")
    print("\n  Top ingested documents:")
    for fname, count in source_counter.most_common(10):
        print(f"    {count:>5} chunks  ←  {fname}")

    # Source pipeline breakdown
    pipeline_counter = Counter(m.get("source", "unknown") for m in metadatas)
    print("\n  Chunks by pipeline:")
    for pipeline, count in pipeline_counter.most_common():
        print(f"    {count:>5} chunks  ←  {pipeline}")

    # Sample chunk preview
    print("\n  Sample chunk (first stored document):")
    print("  " + "-" * 56)
    if documents:
        preview = documents[0][:300].replace("\n", " ")
        print(f"  {preview}...")
    print("  " + "-" * 56)

    print(f"\n[OK] ChromaDB collection '{store.collection_name}' is healthy and populated.")
    print("=" * 60)


if __name__ == "__main__":
    main()
