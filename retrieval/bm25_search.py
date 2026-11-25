"""
retrieval/bm25_search.py
Sparse (keyword-based) retrieval using BM25 over ingested documents.

The BM25 index is built lazily from all documents stored in ChromaDB.
It's rebuilt automatically when the document count changes (new ingestion detected).

This complements the dense BGE-M3 vector search — together they form true hybrid retrieval.
"""

import re
import logging
import threading
from typing import List, Dict, Any, Optional

from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

# ── Tokenizer ─────────────────────────────────────────────────────────────────

# Multilingual stopwords (FR/EN/AR) — very common words that hurt BM25 precision
_STOPWORDS = frozenset([
    # English
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "have", "has",
    "had", "do", "does", "did", "will", "would", "could", "should", "may",
    "might", "shall", "must", "and", "or", "but", "if", "in", "on", "at",
    "to", "for", "of", "with", "by", "from", "that", "this", "it", "not",
    # French
    "le", "la", "les", "un", "une", "des", "du", "de", "et", "ou", "en",
    "dans", "sur", "pour", "par", "avec", "que", "qui", "est", "sont",
    "être", "avoir", "ce", "cette", "ces", "pas", "plus", "ne", "se",
    # Arabic particles
    "في", "من", "على", "إلى", "عن", "هل", "ما", "مع", "لا", "هذا", "هذه",
    "أن", "كان", "التي", "الذي", "هو", "هي",
])


def tokenize(text: str) -> List[str]:
    """
    Simple multilingual tokenizer for BM25.
    - Lowercases
    - Splits on non-alphanumeric (keeps Arabic/accented chars)
    - Removes stopwords and tokens shorter than 2 chars
    """
    text = text.lower()
    # Split on whitespace and punctuation, keep unicode letters/digits
    tokens = re.findall(r"[\w\u0600-\u06FF]+", text, re.UNICODE)
    return [t for t in tokens if len(t) >= 2 and t not in _STOPWORDS]


# ── BM25 Index ────────────────────────────────────────────────────────────────

class BM25Search:
    """
    Sparse retrieval engine using BM25Okapi.

    Builds its index from ChromaDB documents (same source as the dense retriever).
    The index is rebuilt lazily when the corpus size changes.

    Thread-safe: uses a lock during rebuilds to prevent concurrent corruption.
    """

    def __init__(self):
        self._index: Optional[BM25Okapi] = None
        self._corpus: List[Dict[str, Any]] = []  # [{text, metadata}, ...]
        self._corpus_size: int = 0
        self._lock = threading.Lock()
        self._tokenized_corpus: List[List[str]] = []

    def _load_corpus_from_chroma(self) -> List[Dict[str, Any]]:
        """
        Pull all documents from the active ChromaDB collections.
        Returns a list of {text, metadata} dicts.
        """
        import chromadb
        from chromadb.config import Settings
        from storage.chroma_store import (
            CHROMA_PERSIST_DIR,
            CHROMA_COLLECTION_FINANCIAL,
            CHROMA_COLLECTION_USER_DATA,
        )

        client = chromadb.PersistentClient(
            path=CHROMA_PERSIST_DIR,
            settings=Settings(anonymized_telemetry=False),
        )

        corpus = []
        for coll_name in [CHROMA_COLLECTION_FINANCIAL, CHROMA_COLLECTION_USER_DATA]:
            try:
                coll = client.get_or_create_collection(
                    name=coll_name,
                    metadata={"hnsw:space": "cosine"},
                )
                count = coll.count()
                if count == 0:
                    continue

                # ChromaDB get() returns all documents (paginated internally)
                # For large collections, batch in chunks of 5000
                batch_size = 5000
                offset = 0
                while offset < count:
                    results = coll.get(
                        limit=batch_size,
                        offset=offset,
                        include=["documents", "metadatas"],
                    )
                    if not results["ids"]:
                        break

                    for doc_id, text, meta in zip(
                        results["ids"],
                        results["documents"],
                        results["metadatas"],
                    ):
                        if text and text.strip():
                            corpus.append({
                                "text": text,
                                "metadata": meta or {},
                                "collection": coll_name,
                                "id": doc_id,
                            })
                    offset += batch_size

            except Exception as e:
                logger.warning("BM25: Failed to load collection '%s': %s", coll_name, e)

        logger.info("BM25: Loaded %d documents from ChromaDB", len(corpus))
        return corpus

    def _build_index(self):
        """Build or rebuild the BM25 index from ChromaDB documents."""
        with self._lock:
            corpus = self._load_corpus_from_chroma()
            if not corpus:
                self._index = None
                self._corpus = []
                self._tokenized_corpus = []
                self._corpus_size = 0
                logger.info("BM25: No documents available — index empty")
                return

            self._corpus = corpus
            self._tokenized_corpus = [tokenize(doc["text"]) for doc in corpus]
            self._index = BM25Okapi(self._tokenized_corpus)
            self._corpus_size = len(corpus)
            logger.info("BM25: Index built with %d documents", self._corpus_size)

    def _ensure_index(self):
        """Rebuild index if not yet built or if corpus size changed."""
        if self._index is None:
            self._build_index()
            return

        # Check if new documents were ingested (lazy refresh)
        try:
            import chromadb
            from chromadb.config import Settings
            from storage.chroma_store import (
                CHROMA_PERSIST_DIR,
                CHROMA_COLLECTION_FINANCIAL,
                CHROMA_COLLECTION_USER_DATA,
            )
            client = chromadb.PersistentClient(
                path=CHROMA_PERSIST_DIR,
                settings=Settings(anonymized_telemetry=False),
            )
            total = 0
            for name in [CHROMA_COLLECTION_FINANCIAL, CHROMA_COLLECTION_USER_DATA]:
                try:
                    total += client.get_or_create_collection(name=name).count()
                except Exception:
                    pass

            if total != self._corpus_size:
                logger.info("BM25: Corpus size changed (%d -> %d). Rebuilding...", self._corpus_size, total)
                self._build_index()
        except Exception as e:
            logger.warning("BM25: Could not check corpus size: %s", e)

    def search(
        self,
        query: str,
        top_k: int = 20,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search the BM25 index with a natural language query.
        When VECTOR_BACKEND=pgvector, delegates to PostgreSQL native full-text search
        to avoid loading 29k+ documents into memory on low-RAM cloud instances.
        """
        from storage.chroma_store import (
            is_pgvector_enabled,
            get_vector_client,
            CHROMA_COLLECTION_FINANCIAL,
            CHROMA_COLLECTION_USER_DATA,
        )
        if is_pgvector_enabled():
            client = get_vector_client()
            fts_results: List[Dict[str, Any]] = []
            for coll_name in [CHROMA_COLLECTION_FINANCIAL, CHROMA_COLLECTION_USER_DATA]:
                coll = client.get_or_create_collection(coll_name)
                fts_results.extend(coll.search_fts(query, top_k=top_k))
            if filter_metadata:
                fts_results = [
                    r for r in fts_results
                    if all(r.get("metadata", {}).get(k) == v for k, v in filter_metadata.items())
                ]
            fts_results.sort(key=lambda x: x["score"], reverse=True)
            return fts_results[:top_k]

        self._ensure_index()

        if self._index is None or not self._corpus:
            return []

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        # Get BM25 scores for all documents
        scores = self._index.get_scores(query_tokens)

        # Create (score, index) pairs and sort descending
        scored_docs = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in scored_docs:
            if score <= 0:
                break  # BM25 scores of 0 mean no term overlap
            if len(results) >= top_k:
                break

            doc = self._corpus[idx]
            meta = doc.get("metadata", {})

            # Apply metadata filter if provided
            if filter_metadata:
                match = all(
                    meta.get(k) == v for k, v in filter_metadata.items()
                )
                if not match:
                    continue

            results.append({
                "text": doc["text"],
                "score": round(float(score), 4),
                "source": meta.get("source_file", meta.get("source", "unknown")),
                "page": meta.get("page", ""),
                "collection": doc.get("collection", ""),
                "metadata": meta,
                "retrieval_method": "bm25",
            })

        logger.info("BM25: Query '%s' -> %d results (top score: %.4f)",
                    query[:50], len(results), results[0]["score"] if results else 0)
        return results

    def rebuild(self):
        """Force a full rebuild of the index. Call after bulk ingestion."""
        self._build_index()
