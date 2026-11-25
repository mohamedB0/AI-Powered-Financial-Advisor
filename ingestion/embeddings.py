"""
ingestion/embeddings.py
Generates dense vector embeddings using either:
  1. Local sentence-transformers (EMBEDDING_BACKEND=local, default when installed)
  2. OpenRouter API `baai/bge-m3` (EMBEDDING_BACKEND=openrouter) — 0 MB local RAM,
     identical 1024-dim vector space (0.999999 cosine similarity with local BGE-M3).
"""

import logging
import os
from typing import List

logger = logging.getLogger(__name__)

# BGE-family models need a special query prefix for retrieval
_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder:
    """
    Generates dense embeddings via local SentenceTransformer or OpenRouter API.
    """

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or os.getenv(
            "EMBEDDING_MODEL", "BAAI/bge-m3"
        )
        self._is_bge = "bge" in self.model_name.lower()
        backend = os.getenv("EMBEDDING_BACKEND", "auto").lower()

        if backend == "auto":
            try:
                import sentence_transformers  # noqa: F401
                backend = "local"
            except ImportError:
                backend = "openrouter"

        self.backend = backend

        if self.backend == "openrouter":
            self.api_key = os.getenv("OPENROUTER_API_KEY", "")
            self.or_model = os.getenv("OPENROUTER_EMBEDDING_MODEL", "baai/bge-m3")
            self.dimension = 1024
            self.model = None
            logger.info(
                "Embedder: using OpenRouter API '%s' (dim=%d, 0 MB local RAM)",
                self.or_model,
                self.dimension,
            )
        else:
            from sentence_transformers import SentenceTransformer

            logger.info("Embedder: loading local model '%s'", self.model_name)
            self.model = SentenceTransformer(self.model_name)
            self.dimension = self.model.get_sentence_embedding_dimension()
            logger.info(
                "Embedder: local model loaded (dim=%d, bge=%s)",
                self.dimension,
                self._is_bge,
            )

    def _embed_openrouter(self, texts: List[str]) -> List[List[float]]:
        """Call OpenRouter /embeddings endpoint in batches."""
        import requests

        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY is required when EMBEDDING_BACKEND=openrouter")

        batch_size = 32
        all_embeddings: List[List[float]] = []
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            resp = requests.post(
                "https://openrouter.ai/api/v1/embeddings",
                headers=headers,
                json={"model": self.or_model, "input": batch},
                timeout=30,
            )
            resp.raise_for_status()
            data = sorted(resp.json()["data"], key=lambda x: x["index"])
            all_embeddings.extend([item["embedding"] for item in data])
        return all_embeddings

    def embed(self, texts: List[str], *, is_query: bool = False) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts.
        """
        if not texts:
            return []

        if self._is_bge and is_query:
            texts = [_BGE_QUERY_PREFIX + t for t in texts]

        if self.backend == "openrouter":
            return self._embed_openrouter(texts)

        batch_size = 4
        all_embeddings = []
        total_batches = (len(texts) + batch_size - 1) // batch_size

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch_emb = self.model.encode(batch, show_progress_bar=False, convert_to_numpy=True)
            all_embeddings.extend(batch_emb.tolist())

            batch_num = (i // batch_size) + 1
            progress_percent = batch_num / total_batches
            bar_length = 25
            filled_length = int(bar_length * progress_percent)
            bar = "█" * filled_length + "░" * (bar_length - filled_length)
            logger.info(
                "[EMBEDDER] %s ─ [%s] %3d%% ─ (Batch %d/%d)",
                self.model_name,
                bar,
                int(progress_percent * 100),
                batch_num,
                total_batches,
            )

        return all_embeddings

    def embed_one(self, text: str, *, is_query: bool = False) -> List[float]:
        """Embed a single text string."""
        return self.embed([text], is_query=is_query)[0]
