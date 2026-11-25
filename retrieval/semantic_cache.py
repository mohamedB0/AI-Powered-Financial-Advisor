"""
retrieval/semantic_cache.py
Semantic caching using ChromaDB to store and instantly retrieve identical or highly similar past queries.
"""

import hashlib
import logging
from typing import Optional, Dict, Any

from ingestion.embeddings import Embedder
from storage.chroma_store import CHROMA_PERSIST_DIR, get_vector_client

logger = logging.getLogger(__name__)

class SemanticCache:
    """
    Semantic cache using ChromaDB or Supabase pgvector.
    Matches incoming user questions against past questions using cosine similarity.
    If similarity > 0.95 (distance < 0.05), we bypass the entire RAG pipeline and LLM.
    """
    def __init__(self, embedder: Optional[Embedder] = None, collection_name: str = "semantic_cache"):
        self.embedder = embedder or Embedder()
        self.client = get_vector_client(CHROMA_PERSIST_DIR)
        # Note: hnsw:space=cosine configures ChromaDB to return cosine distance (1 - similarity)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def get(self, query: str, threshold: float = 0.95) -> Optional[Dict[str, Any]]:
        """
        Check if the query matches a cached entry within the distance threshold.
        """
        if self.collection.count() == 0:
            return None

        # Symmetric match for query-to-query comparison
        query_vec = self.embedder.embed_one(query, is_query=False)
        results = self.collection.query(
            query_embeddings=[query_vec],
            n_results=1,
            include=["documents", "metadatas", "distances"]
        )
        
        if not results["ids"] or not results["ids"][0]:
            return None

        distance = results["distances"][0][0]
        similarity = 1.0 - distance
        
        if similarity >= threshold:
            meta = results["metadatas"][0][0]
            logger.info(f"Semantic Cache HIT: score={similarity:.4f} for query '{query}'")
            return {
                "answer": results["documents"][0][0],
                "context": meta.get("context", "Cached context"),
                "route": meta.get("route", "cache"),
                "cached_query": meta.get("original_query", "Unknown"),
                "similarity": similarity
            }
        
        logger.info(f"Semantic Cache MISS. Highest similarity was {similarity:.4f}")
        return None

    def set(self, query: str, answer: str, context: str, route: str):
        """
        Cache a newly generated answer.
        """
        query_vec = self.embedder.embed_one(query, is_query=False)
        doc_id = hashlib.md5(query.encode()).hexdigest()
        
        self.collection.upsert(
            ids=[doc_id],
            embeddings=[query_vec],
            documents=[answer],
            metadatas=[{
                 "original_query": query,
                 "context": context[:8000],  # Max length protection for ChromaDB metadata
                 "route": route
            }]
        )
        logger.info(f"Semantic Cache updated for query: '{query}'")
