"""
Lightweight ChromaDB check — no custom imports, no embedding model loading.
"""
import chromadb
from chromadb.config import Settings
from collections import Counter
from pathlib import Path

CHROMA_DIR = str(Path(__file__).parent.parent / "chroma_db")

print(f"ChromaDB path: {CHROMA_DIR}")
client = chromadb.PersistentClient(path=CHROMA_DIR, settings=Settings(anonymized_telemetry=False))

collections = client.list_collections()
print(f"\nCollections found: {len(collections)}")
for col in collections:
    print(f"  - {col.name}  ({col.count()} chunks)")

# Inspect each collection
for col in collections:
    print(f"\n{'='*60}")
    print(f"  Collection: {col.name}")
    print(f"  Total chunks: {col.count()}")
    print(f"{'='*60}")

    if col.count() == 0:
        print("  [EMPTY]")
        continue

    # Get a sample (up to 1000)
    sample_size = min(col.count(), 1000)
    results = col.get(limit=sample_size, include=["metadatas", "documents"])
    metadatas = results.get("metadatas", [])
    documents = results.get("documents", [])

    # Source file breakdown
    source_counter = Counter(m.get("source_file", "unknown") for m in metadatas)
    print(f"\n  Unique source files: {len(source_counter)}")
    print("  Top documents:")
    for fname, count in source_counter.most_common(15):
        print(f"    {count:>5} chunks  <-  {fname}")

    # Pipeline breakdown
    pipeline_counter = Counter(m.get("source", "unknown") for m in metadatas)
    print("\n  Chunks by pipeline/source:")
    for pipeline, count in pipeline_counter.most_common():
        print(f"    {count:>5} chunks  <-  {pipeline}")

    # Language breakdown (if present)
    lang_counter = Counter(m.get("language", "unknown") for m in metadatas)
    if any(l != "unknown" for l in lang_counter):
        print("\n  Chunks by language:")
        for lang, count in lang_counter.most_common():
            print(f"    {count:>5} chunks  <-  {lang}")

    # Year breakdown (if present)
    year_counter = Counter(m.get("year", "unknown") for m in metadatas)
    if any(y != "unknown" for y in year_counter):
        print("\n  Chunks by year:")
        for year, count in year_counter.most_common():
            print(f"    {count:>5} chunks  <-  {year}")

    # Sample chunk preview
    if documents:
        print(f"\n  Sample chunk preview (first doc):")
        print("  " + "-"*56)
        preview = documents[0][:400].replace("\n", " ")
        print(f"  {preview}...")
        print("  " + "-"*56)

    # Show all metadata keys present
    all_keys = set()
    for m in metadatas:
        all_keys.update(m.keys())
    print(f"\n  Metadata keys: {sorted(all_keys)}")

print("\n[DONE]")
