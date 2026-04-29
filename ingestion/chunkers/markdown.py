"""
ingestion/chunkers/markdown.py
MarkdownTextSplitter — used when PDFs are first converted to markdown
(e.g. via pymupdf4llm).  Preserves heading structure.
"""

import logging
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)

try:
    from langchain.text_splitter import MarkdownTextSplitter as _LCMarkdownSplitter  # type: ignore
    _HAS_LANGCHAIN = True
except ImportError:
    _HAS_LANGCHAIN = False


class MarkdownChunker:
    """
    Splits markdown-formatted text while preserving heading structure.
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(
        self,
        text: str,
        metadata: Dict[str, Any],
    ) -> List[Tuple[str, Dict[str, Any]]]:
        if _HAS_LANGCHAIN:
            return self._langchain_split(text, metadata)
        return self._heading_fallback(text, metadata)

    # ── LangChain markdown splitting ──────────────────────────────────────────

    def _langchain_split(
        self, text: str, metadata: Dict[str, Any]
    ) -> List[Tuple[str, Dict[str, Any]]]:
        splitter = _LCMarkdownSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        parts = splitter.split_text(text)
        results = []
        for idx, part in enumerate(parts):
            chunk_meta = {**metadata, "chunk_index": idx}
            results.append((part.strip(), chunk_meta))
        logger.debug("MarkdownChunker: %d chunks (langchain)", len(results))
        return results

    # ── Heading-based fallback ────────────────────────────────────────────────

    @staticmethod
    def _heading_fallback(
        text: str, metadata: Dict[str, Any]
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """Split on markdown headings (# lines)."""
        import re

        sections = re.split(r"(?=^#{1,3}\s)", text, flags=re.MULTILINE)
        results = []
        for idx, section in enumerate(sections):
            section = section.strip()
            if not section:
                continue
            chunk_meta = {**metadata, "chunk_index": idx}
            results.append((section, chunk_meta))
        logger.debug("MarkdownChunker: %d chunks (heading fallback)", len(results))
        return results
