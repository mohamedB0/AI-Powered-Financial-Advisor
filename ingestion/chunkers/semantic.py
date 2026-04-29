"""
ingestion/chunkers/semantic.py
SemanticChunker — splits text at topic boundaries rather than
fixed character counts.  Best for long transcripts and analyst reports.
"""

import logging
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)

try:
    from langchain_experimental.text_splitter import SemanticChunker as _LCSemanticChunker  # type: ignore
    from langchain.embeddings import HuggingFaceEmbeddings  # type: ignore
    _HAS_SEMANTIC = True
except ImportError:
    _HAS_SEMANTIC = False


class SemanticChunker:
    """
    Splits text at semantic / topic boundaries using embeddings.

    Falls back to paragraph-based splitting when langchain_experimental
    is not installed.
    """

    def __init__(
        self,
        embed_model: str = "BAAI/bge-large-en-v1.5",
        breakpoint_threshold_type: str = "percentile",
    ):
        import os
        self.embed_model = os.getenv("EMBEDDING_MODEL", embed_model)
        self.breakpoint_threshold_type = breakpoint_threshold_type

    def chunk(
        self,
        text: str,
        metadata: Dict[str, Any],
    ) -> List[Tuple[str, Dict[str, Any]]]:
        if _HAS_SEMANTIC:
            return self._semantic_split(text, metadata)
        return self._paragraph_fallback(text, metadata)

    # ── LangChain semantic splitting ──────────────────────────────────────────

    def _semantic_split(
        self, text: str, metadata: Dict[str, Any]
    ) -> List[Tuple[str, Dict[str, Any]]]:
        embeddings = HuggingFaceEmbeddings(model_name=self.embed_model)
        chunker = _LCSemanticChunker(
            embeddings=embeddings,
            breakpoint_threshold_type=self.breakpoint_threshold_type,
        )
        docs = chunker.create_documents([text])
        results = []
        for idx, doc in enumerate(docs):
            chunk_meta = {**metadata, "chunk_index": idx}
            results.append((doc.page_content.strip(), chunk_meta))
        logger.debug("SemanticChunker: %d chunks (semantic)", len(results))
        return results

    # ── Paragraph-based fallback ──────────────────────────────────────────────

    @staticmethod
    def _paragraph_fallback(
        text: str, metadata: Dict[str, Any]
    ) -> List[Tuple[str, Dict[str, Any]]]:
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        results = []
        for idx, para in enumerate(paragraphs):
            chunk_meta = {**metadata, "chunk_index": idx}
            results.append((para, chunk_meta))
        logger.debug("SemanticChunker: %d chunks (paragraph fallback)", len(results))
        return results
