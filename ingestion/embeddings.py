"""
ingestion/embeddings.py
Generates dense vector embeddings using sentence-transformers.
Loads the model once and exposes embed(texts).
Handles BGE query-prefix injection automatically.
"""

import logging
import os
from typing import List

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# BGE-family models need a special query prefix for retrieval
_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder:
    """
    Wraps sentence-transformers to generate dense embeddings.

    Args:
        model_name: HuggingFace model id.
                     Defaults to env EMBEDDING_MODEL or 'BAAI/bge-large-en-v1.5'.
    """

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or os.getenv(
            "EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5"
        )
        logger.info("Embedder: loading model '%s'", self.model_name)
        self.model = SentenceTransformer(self.model_name)
        self.dimension: int = self.model.get_sentence_embedding_dimension()
        self._is_bge = "bge" in self.model_name.lower()
        logger.info(
            "Embedder: model loaded (dim=%d, bge=%s)", self.dimension, self._is_bge
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def embed(self, texts: List[str], *, is_query: bool = False) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts.

        Args:
            texts:    Strings to embed.
            is_query: If True and the model is BGE-family, prepend the
                      retrieval prefix automatically.
        Returns:
            List of embedding vectors.
        """
        if self._is_bge and is_query:
            texts = [_BGE_QUERY_PREFIX + t for t in texts]

        embeddings = self.model.encode(
            texts, show_progress_bar=True, convert_to_numpy=True
        )
        return embeddings.tolist()

    def embed_one(self, text: str, *, is_query: bool = False) -> List[float]:
        """Embed a single text string."""
        return self.embed([text], is_query=is_query)[0]
