"""
scripts/count_documents.py
Count scraped documents by source and file type.

Usage:
    python scripts/count_documents.py
"""

import os
from pathlib import Path
from collections import defaultdict

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
DOCS_DIR = PROJECT_ROOT / "data" / "collected_data" / "documents"


def count_documents():
    print("=" * 65)
    print("  DOCUMENT COLLECTION SUMMARY")
    print("=" * 65)
    print()

    if not DOCS_DIR.exists():
        print(f"  [ERROR] Documents directory not found: {DOCS_DIR}")
        return

    grand_total = 0
    grand_by_ext = defaultdict(int)
    source_results = []

    # Count per source folder
    for source_dir in sorted(DOCS_DIR.iterdir()):
        if not source_dir.is_dir():
            continue

        source_name = source_dir.name
        file_count = 0
        by_ext = defaultdict(int)
        total_size = 0

        for f in source_dir.rglob("*"):
            if f.is_file():
                ext = f.suffix.lower()
                by_ext[ext] += 1
                file_count += 1
                total_size += f.stat().st_size
                grand_by_ext[ext] += 1

        grand_total += file_count
        size_mb = total_size / (1024 * 1024)

        source_results.append({
            "name": source_name,
            "count": file_count,
            "size_mb": size_mb,
            "by_ext": dict(by_ext),
        })

    # Display results
    print(f"  {'Source':<25} {'Files':<8} {'Size':<10} {'Formats'}")
    print(f"  {'-'*25} {'-'*8} {'-'*10} {'-'*20}")

    for src in source_results:
        formats = ", ".join(f"{ext}({n})" for ext, n in sorted(src["by_ext"].items()))
        print(f"  {src['name']:<25} {src['count']:<8} {src['size_mb']:.1f} MB    {formats}")

    print(f"  {'-'*25} {'-'*8} {'-'*10} {'-'*20}")
    print(f"  {'TOTAL':<25} {grand_total:<8}")
    print()

    # Summary by file type
    print("  File types across all sources:")
    for ext, count in sorted(grand_by_ext.items(), key=lambda x: -x[1]):
        print(f"    {ext:<10} {count} files")

    print()

    # Also check user uploads
    uploads_dir = PROJECT_ROOT / "data" / "uploads"
    if uploads_dir.exists():
        upload_files = list(uploads_dir.rglob("*"))
        upload_files = [f for f in upload_files if f.is_file()]
        print(f"  User uploads (data/uploads/): {len(upload_files)} files")
        if upload_files:
            upload_exts = defaultdict(int)
            for f in upload_files:
                upload_exts[f.suffix.lower()] += 1
            for ext, n in sorted(upload_exts.items(), key=lambda x: -x[1]):
                print(f"    {ext:<10} {n} files")
    print()

    # ChromaDB vector counts
    print("  Vector store (ChromaDB):")
    try:
        import chromadb
        from chromadb.config import Settings
        client = chromadb.PersistentClient(
            path=str(PROJECT_ROOT / "chroma_db"),
            settings=Settings(anonymized_telemetry=False),
        )
        for coll_name in ["financial_data", "user_data"]:
            try:
                coll = client.get_or_create_collection(coll_name, metadata={"hnsw:space": "cosine"})
                print(f"    {coll_name:<20} {coll.count()} vectors")
            except Exception as e:
                print(f"    {coll_name:<20} ERROR: {e}")
    except ImportError:
        print("    (chromadb not installed)")
    except Exception as e:
        print(f"    ERROR: {e}")

    print()
    print("=" * 65)


if __name__ == "__main__":
    count_documents()
