"""
ingestion/chunkers/recursive.py
RecursiveCharacterTextSplitter — default chunker for PDF prose.
Uses LangChain's splitter under the hood.
"""

import logging
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter  # type: ignore
    _HAS_LANGCHAIN = True
except ImportError:
    _HAS_LANGCHAIN = False


class RecursiveChunker:
    """
    Splits text using recursive character splitting.
    Default: chunk_size=512, overlap=64.
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(
        self,
        text: str,
        metadata: Dict[str, Any],
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Split *text* into overlapping chunks, each inheriting *metadata*.

        Returns:
            List of (chunk_text, chunk_metadata) tuples.
        """
        if _HAS_LANGCHAIN:
            return self._langchain_split(text, metadata)
        return self._fallback_split(text, metadata)

    # ── LangChain-based splitting ─────────────────────────────────────────────

    def _langchain_split(
        self, text: str, metadata: Dict[str, Any]
    ) -> List[Tuple[str, Dict[str, Any]]]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        parts = splitter.split_text(text)
        results = []
        for idx, part in enumerate(parts):
            chunk_meta = {**metadata, "chunk_index": idx}
            results.append((part.strip(), chunk_meta))
        logger.debug("RecursiveChunker: %d chunks (langchain)", len(results))
        return results

    # ── Pure-Python fallback ──────────────────────────────────────────────────

    def _fallback_split(
        self, text: str, metadata: Dict[str, Any]
    ) -> List[Tuple[str, Dict[str, Any]]]:
        results = []
        start = 0
        idx = 0
        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunk_meta = {**metadata, "chunk_index": idx}
                results.append((chunk_text, chunk_meta))
                idx += 1
            start += self.chunk_size - self.chunk_overlap
        logger.debug("RecursiveChunker: %d chunks (fallback)", len(results))
        return results
