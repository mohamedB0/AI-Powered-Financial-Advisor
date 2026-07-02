"""
storage/chroma_store.py
ChromaDB client wrapper for financial document ingestion.

Collections:
  - financial_docs_cdc_hcp  : Scraped documents from CDC, HCP, and config-driven sites (288 docs)
  - financial_data          : Future ingestion target — new financial data added going forward

Set CHROMA_COLLECTION env var to switch the active collection.
"""
import os
import uuid
import logging
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).parent.parent
_DEFAULT_CHROMA_DIR = str(_PROJECT_ROOT / "chroma_db")

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", _DEFAULT_CHROMA_DIR)
# If the value from .env is a relative path, make it absolute from project root
if not Path(CHROMA_PERSIST_DIR).is_absolute():
    CHROMA_PERSIST_DIR = str(_PROJECT_ROOT / CHROMA_PERSIST_DIR)

# ── Known Collections ────────────────────────────────────────────────────────
# Legacy collection: scraped docs from CDC, HCP, and config-driven sites
CHROMA_COLLECTION_LEGACY      = "financial_docs_cdc_hcp"

# Pipeline collection: target for all automated ingestion tasks (scrapers, workers)
CHROMA_COLLECTION_FINANCIAL   = "financial_data"

# User collection: stores user-uploaded documents from the chat UI
CHROMA_COLLECTION_USER_DATA   = "user_data"

# Active collection — controlled via CHROMA_COLLECTION env var
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", CHROMA_COLLECTION_FINANCIAL)


class ChromaStore:
    """
    Manages ChromaDB persistence for scraped financial documents
    from all pipeline sources (CDC, HCP, config-driven, local docs).
    """

    def __init__(
        self,
        collection_name: str = CHROMA_COLLECTION,
        persist_dir: str = CHROMA_PERSIST_DIR,
    ):
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "ChromaStore: collection '%s' ready (%d existing docs)",
            collection_name,
            self.collection.count(),
        )

    def upsert(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
        ids: Optional[List[str]] = None,
        batch_size: int = 128,
    ) -> int:
        if not texts:
            return 0

        if ids is None:
            # Hash the entire text for a deterministic ID
            ids = [hashlib.md5(t.encode()).hexdigest() for t in texts]

        # Deduplicate the batch locally so ChromaDB doesn't fail on identical chunks
        unique_ids = set()
        dedup_texts, dedup_embeds, dedup_metas, dedup_ids = [], [], [], []
        
        for doc_id, txt, emb, meta in zip(ids, texts, embeddings, metadatas):
            if doc_id not in unique_ids:
                unique_ids.add(doc_id)
                dedup_texts.append(txt)
                dedup_embeds.append(emb)
                dedup_metas.append(meta)
                dedup_ids.append(doc_id)
                
        texts, embeddings, metadatas, ids = dedup_texts, dedup_embeds, dedup_metas, dedup_ids

        # ChromaDB requires metadata values to be str/int/float/bool
        clean_meta = []
        for m in metadatas:
            clean_meta.append({
                k: (str(v) if not isinstance(v, (str, int, float, bool)) else v)
                for k, v in m.items()
            })

        # Upsert in batches to avoid ChromaDB limits
        total_stored = 0
        for i in range(0, len(texts), batch_size):
            batch_ids = ids[i:i+batch_size]
            batch_emb = embeddings[i:i+batch_size]
            batch_docs = texts[i:i+batch_size]
            batch_meta = clean_meta[i:i+batch_size]

            self.collection.upsert(
                ids=batch_ids,
                embeddings=batch_emb,
                documents=batch_docs,
                metadatas=batch_meta,
            )
            total_stored += len(batch_ids)

        logger.info(
            "ChromaStore: upserted %d chunks into '%s'", total_stored, self.collection_name
        )
        return total_stored

    def count(self) -> int:
        return self.collection.count()
