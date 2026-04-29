"""
retrieval/vector_search.py
Semantic search over financial documents stored in ChromaDB.

Queries ALL known collections and merges results by score, so documents
ingested into any collection (legacy or new) are always surfaced.
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import List, Dict, Any, Optional
from ingestion.embeddings import Embedder
from storage.chroma_store import (
    ChromaStore,
    CHROMA_COLLECTION_LEGACY,
    CHROMA_COLLECTION_FINANCIAL,
    CHROMA_PERSIST_DIR,
)
import chromadb
from chromadb.config import Settings


class VectorSearch:
    """
    Wraps the BGE Embedder + ChromaDB to perform semantic retrieval
    over all ingested financial documents (both collections).

    Searches:
      - financial_docs_cdc_hcp  (legacy: CDC, HCP scraped docs)
      - financial_data          (new: future ingestion target)

    Results from both collections are merged and ranked by score.
    """

    def __init__(self, embedder: Embedder = None):
        self.embedder = embedder or Embedder()
        # Connect to the ChromaDB client once
        self._client = chromadb.PersistentClient(
            path=CHROMA_PERSIST_DIR,
            settings=Settings(anonymized_telemetry=False),
        )
        # Get handles to both collections (create if missing)
        self._collections = {
            name: self._client.get_or_create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"},
            )
            for name in [CHROMA_COLLECTION_LEGACY, CHROMA_COLLECTION_FINANCIAL]
        }

    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
        collection: Optional[str] = None,  # None = search all collections
    ) -> List[Dict[str, Any]]:
        """
        Embed the query and perform ANN search across all collections.

        Args:
            query:           Natural language question.
            top_k:           Number of results to return (across all collections).
            filter_metadata: Optional ChromaDB metadata filter.
            collection:      If set, restrict search to this collection name only.

        Returns:
            List of dicts: {'text', 'score', 'source', 'page', 'collection', ...}
        """
        query_vector = self.embedder.embed_one(query, is_query=True)

        where_filter = None
        if filter_metadata:
            conditions = [
                {k: {"$eq": v}} for k, v in filter_metadata.items()
            ]
            where_filter = (
                {"$and": conditions}
                if len(conditions) > 1
                else {list(filter_metadata.keys())[0]: {"$eq": list(filter_metadata.values())[0]}}
            )

        # Decide which collections to query
        target_collections = (
            {collection: self._collections[collection]}
            if collection and collection in self._collections
            else self._collections
        )

        all_results: List[Dict[str, Any]] = []

        for coll_name, coll in target_collections.items():
            if coll.count() == 0:
                continue  # skip empty collections
            try:
                results = coll.query(
                    query_embeddings=[query_vector],
                    n_results=min(top_k, coll.count()),
                    where=where_filter,
                    include=["documents", "metadatas", "distances"],
                )
            except Exception as e:
                print(f"[VectorSearch] Error querying '{coll_name}': {e}")
                continue

            if not results or not results.get("ids") or not results["ids"][0]:
                continue

            for doc_id, text, meta, dist in zip(
                results["ids"][0],
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                score = max(0.0, 1.0 - dist)
                all_results.append({
                    **meta,
                    "text"      : text,
                    "score"     : round(score, 4),
                    "source"    : meta.get("source_file", meta.get("source", "unknown")),
                    "page"      : meta.get("page", ""),
                    "collection": coll_name,
                })

        # Merge and rank by score descending, return top_k overall
        all_results.sort(key=lambda x: x["score"], reverse=True)
        return all_results[:top_k]

    def collections_info(self) -> Dict[str, int]:
        """Return a summary of all collections and their document counts."""
        return {name: coll.count() for name, coll in self._collections.items()}

    def format_context(self, results: List[Dict[str, Any]]) -> str:
        """Format search results into a readable context string for the LLM."""
        if not results:
            return "No relevant documents found."
        parts = []
        for i, r in enumerate(results, 1):
            source = r.get("source_file", r.get("source", "unknown"))
            page   = r.get("page", "")
            score  = r.get("score", 0)
            coll   = r.get("collection", "")
            page_str = f" (page {page})" if page else ""
            coll_str = f" [{coll}]" if coll else ""
            parts.append(
                f"[Doc {i} | {source}{page_str}{coll_str} | score={score:.3f}]\n{r['text']}"
            )
        return "\n\n---\n\n".join(parts)


