"""
ingestion/loaders/pdf_loader.py
Extracts text from PDF files using PyMuPDF (fitz) for prose and pdfplumber for tables.
Each page becomes one (text, metadata) record with page_num in metadata.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pdfplumber

from ingestion.metadata import build_metadata, validate_metadata

logger = logging.getLogger(__name__)


class PDFLoader:
    """
    Loads a PDF and returns a list of (text, metadata) tuples — one per page.

    Uses pdfplumber for both prose and table extraction.  If a page contains
    tables, table text is appended after the main prose block.
    """

    def load(self, file_path: str) -> List[Tuple[str, Dict[str, Any]]]:
        path = Path(file_path)
        results: List[Tuple[str, Dict[str, Any]]] = []

        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                # ── Prose extraction ──────────────────────────────────────
                prose = page.extract_text() or ""

                # ── Table extraction ──────────────────────────────────────
                table_texts: list[str] = []
                for table in page.extract_tables():
                    rows = [
                        " | ".join(str(cell) if cell else "" for cell in row)
                        for row in table
                    ]
                    table_texts.append("\n".join(rows))

                combined = prose.strip()
                if table_texts:
                    combined += "\n\n[TABLE]\n" + "\n\n".join(table_texts)

                if not combined.strip():
                    continue

                metadata = build_metadata(
                    source=path.name,
                    doc_type="pdf",
                    page=page_num,
                    file_path=str(path),
                )
                validate_metadata(metadata)
                results.append((combined.strip(), metadata))

        logger.info("PDFLoader: loaded %d pages from '%s'", len(results), path.name)
        return results
