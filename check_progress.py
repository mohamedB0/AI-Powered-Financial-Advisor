import os
import glob
from pathlib import Path
import chromadb
import warnings
warnings.filterwarnings("ignore")

# 1. Get all downloaded PDFs
pdf_files = glob.glob("data/Scraping/**/*.pdf", recursive=True)
downloaded_filenames = [os.path.basename(f) for f in pdf_files]

# 2. Get ingested files from ChromaDB
client = chromadb.PersistentClient(path="chroma_db")
col = client.get_or_create_collection(name="financial_data")
metadatas = col.get(include=["metadatas"])["metadatas"]
if metadatas:
    ingested_filenames = set(m.get("source_file", "") for m in metadatas if m and "source_file" in m)
else:
    ingested_filenames = set()

# 3. Find pending files
pending = set(downloaded_filenames) - ingested_filenames

with open("ingestion_status.txt", "w", encoding="utf-8") as f:
    f.write(f"Total downloaded PDFs: {len(downloaded_filenames)}\n")
    f.write(f"Total ingested PDFs: {len(ingested_filenames)}\n")
    f.write(f"Pending/Stopped on ({len(pending)} remaining):\n")
    # Write a few pending
    for p in list(pending)[:20]:
        f.write(f" - {p}\n")
