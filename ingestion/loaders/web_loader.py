"""
ingestion/loaders/web_loader.py
Extracts article text from HTML files and URLs.

Uses BeautifulSoup for static HTML.  Falls back to trafilatura when available
for better boilerplate removal.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

from bs4 import BeautifulSoup

from ingestion.metadata import build_metadata, validate_metadata

logger = logging.getLogger(__name__)

# Optional: trafilatura for higher-quality text extraction
try:
    import trafilatura  # type: ignore
    _HAS_TRAFILATURA = True
except ImportError:
    _HAS_TRAFILATURA = False


class WebLoader:
    """
    Loads HTML files or fetches URLs and returns (text, metadata) tuples.
    """

    # ── Load from local file ──────────────────────────────────────────────────

    def load(self, file_path: str) -> List[Tuple[str, Dict[str, Any]]]:
        """Load a local HTML file."""
        path = Path(file_path)
        raw_html = path.read_text(encoding="utf-8", errors="ignore")
        return self._extract(raw_html, source=path.name, file_path=str(path))

    # ── Load from URL ─────────────────────────────────────────────────────────

    def load_url(self, url: str) -> List[Tuple[str, Dict[str, Any]]]:
        """Fetch and extract text from a URL."""
        import requests

        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        return self._extract(resp.text, source=url)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _extract(
        self,
        html: str,
        source: str,
        file_path: str | None = None,
    ) -> List[Tuple[str, Dict[str, Any]]]:
        # Try trafilatura first (better boilerplate removal)
        if _HAS_TRAFILATURA:
            text = trafilatura.extract(html) or ""
        else:
            text = self._bs4_extract(html)

        if not text.strip():
            return []

        meta = build_metadata(source=source, doc_type="html")
        if file_path:
            meta["file_path"] = file_path
        validate_metadata(meta)

        logger.info("WebLoader: extracted %d chars from '%s'", len(text), source)
        return [(text.strip(), meta)]

    @staticmethod
    def _bs4_extract(html: str) -> str:
        """Fallback extraction with BeautifulSoup."""
        soup = BeautifulSoup(html, "lxml")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        return soup.get_text(separator="\n", strip=True)
