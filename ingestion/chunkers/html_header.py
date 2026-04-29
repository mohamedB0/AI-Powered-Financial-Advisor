"""
ingestion/chunkers/html_header.py
HTMLHeaderTextSplitter — splits HTML by heading hierarchy (h1/h2/h3)
and preserves the heading path as metadata on each chunk.
"""

import logging
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)

try:
    from langchain.text_splitter import HTMLHeaderTextSplitter as _LCHTMLSplitter  # type: ignore
    _HAS_LANGCHAIN = True
except ImportError:
    _HAS_LANGCHAIN = False


# Heading tags to split on
_HEADERS_TO_SPLIT = [
    ("h1", "header_1"),
    ("h2", "header_2"),
    ("h3", "header_3"),
]


class HTMLHeaderChunker:
    """
    Splits HTML documents by heading tags, preserving the heading
    hierarchy (h1 → h2 → h3) as metadata on each chunk.
    """

    def chunk(
        self,
        html: str,
        metadata: Dict[str, Any],
    ) -> List[Tuple[str, Dict[str, Any]]]:
        if _HAS_LANGCHAIN:
            return self._langchain_split(html, metadata)
        return self._bs4_fallback(html, metadata)

    # ── LangChain HTML header splitting ───────────────────────────────────────

    def _langchain_split(
        self, html: str, metadata: Dict[str, Any]
    ) -> List[Tuple[str, Dict[str, Any]]]:
        splitter = _LCHTMLSplitter(headers_to_split_on=_HEADERS_TO_SPLIT)
        docs = splitter.split_text(html)
        results = []
        for idx, doc in enumerate(docs):
            chunk_meta = {**metadata, **doc.metadata, "chunk_index": idx}
            results.append((doc.page_content.strip(), chunk_meta))
        logger.debug("HTMLHeaderChunker: %d chunks (langchain)", len(results))
        return results

    # ── BeautifulSoup fallback ────────────────────────────────────────────────

    @staticmethod
    def _bs4_fallback(
        html: str, metadata: Dict[str, Any]
    ) -> List[Tuple[str, Dict[str, Any]]]:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "lxml")
        for tag in soup(["script", "style"]):
            tag.decompose()

        results = []
        current_headers: Dict[str, str] = {}
        current_text_parts: list[str] = []
        idx = 0

        for element in soup.find_all(["h1", "h2", "h3", "p", "li", "td"]):
            tag_name = element.name
            text = element.get_text(strip=True)
            if not text:
                continue

            if tag_name in ("h1", "h2", "h3"):
                # Flush previous section
                if current_text_parts:
                    combined = "\n".join(current_text_parts)
                    chunk_meta = {**metadata, **current_headers, "chunk_index": idx}
                    results.append((combined, chunk_meta))
                    idx += 1
                    current_text_parts = []

                level = f"header_{tag_name[1]}"
                current_headers[level] = text
            else:
                current_text_parts.append(text)

        # Flush remaining
        if current_text_parts:
            combined = "\n".join(current_text_parts)
            chunk_meta = {**metadata, **current_headers, "chunk_index": idx}
            results.append((combined, chunk_meta))

        logger.debug("HTMLHeaderChunker: %d chunks (bs4 fallback)", len(results))
        return results
