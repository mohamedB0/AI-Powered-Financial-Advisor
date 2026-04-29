"""
retrieval/reranker.py
Cross-encoder reranking to improve retrieval precision.
"""

import logging
from typing import List, Dict, Any

try:
    from sentence_transformers import CrossEncoder
    HAS_CROSS_ENCODER = True
except ImportError:
    HAS_CROSS_ENCODER = False

logger = logging.getLogger(__name__)


class Reranker:
    """
    Uses a cross-encoder model to re-score (query, document) pairs.
    Cross-encoders are slower but much more accurate than bi-encoders (vector search).
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self.model = None
        if HAS_CROSS_ENCODER:
            try:
                self.model = CrossEncoder(model_name)
                logger.info("Reranker: loaded %s", model_name)
            except Exception as e:
                logger.warning("Reranker: failed to load model %s: %s", model_name, e)
        else:
            logger.warning("Reranker: sentence-transformers not installed. Reranking disabled.")

    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_n: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Re-scores documents based on the query.
        Each document dict must have a 'text' field.
        """
        if not documents or not self.model:
            return documents[:top_n]

        # Prepare pairs for cross-encoder
        pairs = [[query, doc.get("text", "")] for doc in documents]
        
        # Predict relevance scores
        scores = self.model.predict(pairs)
        
        # Attach scores to documents
        for i, doc in enumerate(documents):
            doc["rerank_score"] = float(scores[i])
            
        # Sort by rerank_score descending
        results = sorted(documents, key=lambda x: x["rerank_score"], reverse=True)
        
        return results[:top_n]
