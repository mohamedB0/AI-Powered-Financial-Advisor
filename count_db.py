import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from storage.chroma_store import ChromaStore

if __name__ == "__main__":
    store = ChromaStore()
    docs = store.collection.get(include=["metadatas"])
    metas = docs.get("metadatas", [])
    
    unique_sources = set()
    for m in metas:
        if m and "source_file" in m:
            unique_sources.add(m["source_file"])
        # Some loaders might use 'source'
        elif m and "source" in m:
            unique_sources.add(m["source"])
            
    print(f"TOTAL CHUNKS: {len(metas)}")
    print(f"UNIQUE FILES: {len(unique_sources)}")
