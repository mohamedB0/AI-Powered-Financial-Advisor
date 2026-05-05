"""
retrieval/reranker.py
Cross-encoder reranking via Jina AI API to improve retrieval precision.
"""

import os
import requests
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class Reranker:
    """
    Uses Jina AI's Multilingual Reranker API to re-score (query, document) pairs.
    This avoids heavy local model downloads while providing SOTA multilingual accuracy.
    """

    def __init__(self, model_name: str = "jina-reranker-v2-base-multilingual"):
        self.model_name = model_name
        self.api_key = os.getenv("JINA_API_KEY")
        if not self.api_key:
            logger.warning("Reranker: JINA_API_KEY not found in environment. Reranking will be disabled.")

    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_n: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Re-scores documents based on the query via Jina API.
        Each document dict must have a 'text' field.
        """
        if not documents or not self.api_key:
            return documents[:top_n]

        # Extract only texts, as the API expects a list of strings
        texts = [doc.get("text", "") for doc in documents]

        url = "https://api.jina.ai/v1/rerank"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        data = {
            "model": self.model_name,
            "query": query,
            "top_n": top_n,
            "documents": texts
        }

        try:
            response = requests.post(url, headers=headers, json=data, timeout=10)
            response.raise_for_status()
            results = response.json().get("results", [])
            
            # The API returns results sorted by relevance!
            # Format: [{"index": <original_index>, "relevance_score": <score>, ...}, ...]
            reranked_docs = []
            for item in results:
                original_index = item["index"]
                doc = documents[original_index].copy()
                doc["rerank_score"] = float(item["relevance_score"])
                reranked_docs.append(doc)
                
            return reranked_docs

        except Exception as e:
            logger.error(f"Jina Reranker API failed: {e}")
            return documents[:top_n]

