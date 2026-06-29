"""
ingestion/chunkers/table_aware.py
Table-aware recursive chunker for financial documents.

Detects table structures in text and treats them as atomic units (never splits
within a table). Remaining prose is split with standard recursive chunking.

This is critical for financial PDFs where key metrics are in tables that lose
all meaning when split into individual rows.
"""

import re
import logging
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)

# Patterns that indicate a line is part of a table
_TABLE_LINE_PATTERNS = [
    re.compile(r".*\|.*\|"),              # Pipe-separated (markdown tables, extracted tables)
    re.compile(r"^\s*[\w\s]+\$[\d,.]+"),  # "Label $123.45" financial line items
    re.compile(r"^\s*\d[\d,.]+\s+\d"),    # Consecutive numbers (table rows)
]


def _is_table_line(line: str) -> bool:
    """Heuristic: does this line look like part of a table?"""
    stripped = line.strip()
    if not stripped:
        return False
    # Pipe-separated content
    if "|" in stripped and stripped.count("|") >= 1:
        return True
    # Financial line items with dollar amounts
    if re.search(r"\$\s*[\d,.]+", stripped):
        return True
    # Lines that are mostly numbers (3+ number groups)
    if len(re.findall(r"[\d,.]+", stripped)) >= 3 and len(stripped) < 200:
        return True
    return False


def _extract_tables_and_prose(text: str) -> List[Tuple[str, str]]:
    """
    Split text into alternating (type, content) segments where type is
    'table' or 'prose'. Consecutive table lines are merged into one block.
    """
    lines = text.split("\n")
    segments = []
    current_type = "prose"
    current_lines = []

    for line in lines:
        line_is_table = _is_table_line(line)
        new_type = "table" if line_is_table else "prose"

        if new_type != current_type:
            # Flush current segment
            content = "\n".join(current_lines).strip()
            if content:
                segments.append((current_type, content))
            current_lines = [line]
            current_type = new_type
        else:
            current_lines.append(line)

    # Flush last segment
    content = "\n".join(current_lines).strip()
    if content:
        segments.append((current_type, content))

    return segments


class TableAwareChunker:
    """
    Splits text while preserving table integrity.

    Strategy:
    - Detect contiguous table blocks and keep each as an atomic chunk
      (with up to 2 preceding prose lines as context/header)
    - Split remaining prose with recursive character splitting

    Args:
        chunk_size: Max chars for prose chunks (tables can exceed this)
        chunk_overlap: Overlap for prose chunks
        max_table_size: If a table exceeds this, split it into groups of rows
    """

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 100,
        max_table_size: int = 2000,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_table_size = max_table_size

    def chunk(
        self,
        text: str,
        metadata: Dict[str, Any],
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Split text into chunks while preserving table structure.
        """
        segments = _extract_tables_and_prose(text)
        results = []
        chunk_idx = 0
        prev_prose_tail = ""  # Last 2 lines of preceding prose (table header context)

        for seg_type, content in segments:
            if seg_type == "table":
                # Prepend the preceding prose lines as header context
                table_with_context = (prev_prose_tail + "\n" + content).strip() if prev_prose_tail else content

                # If table is too large, split into groups
                if len(table_with_context) > self.max_table_size:
                    table_lines = table_with_context.split("\n")
                    # Keep header (first 2 lines) with each group
                    header = "\n".join(table_lines[:2])
                    data_lines = table_lines[2:]
                    group = [header]
                    group_len = len(header)

                    for line in data_lines:
                        if group_len + len(line) > self.max_table_size:
                            chunk_meta = {**metadata, "chunk_index": chunk_idx, "chunk_type": "table"}
                            results.append(("\n".join(group).strip(), chunk_meta))
                            chunk_idx += 1
                            group = [header]
                            group_len = len(header)
                        group.append(line)
                        group_len += len(line) + 1

                    if group:
                        chunk_meta = {**metadata, "chunk_index": chunk_idx, "chunk_type": "table"}
                        results.append(("\n".join(group).strip(), chunk_meta))
                        chunk_idx += 1
                else:
                    chunk_meta = {**metadata, "chunk_index": chunk_idx, "chunk_type": "table"}
                    results.append((table_with_context, chunk_meta))
                    chunk_idx += 1

            else:
                # Prose: use recursive splitting
                prose_chunks = self._split_prose(content)
                for pc in prose_chunks:
                    chunk_meta = {**metadata, "chunk_index": chunk_idx, "chunk_type": "prose"}
                    results.append((pc, chunk_meta))
                    chunk_idx += 1

                # Remember the tail of this prose for next table's context
                lines = content.strip().split("\n")
                prev_prose_tail = "\n".join(lines[-2:]) if len(lines) >= 2 else content.strip()

        logger.debug("TableAwareChunker: %d chunks (%d tables, %d prose)",
                     len(results),
                     sum(1 for _, m in results if m.get("chunk_type") == "table"),
                     sum(1 for _, m in results if m.get("chunk_type") == "prose"))
        return results

    def _split_prose(self, text: str) -> List[str]:
        """Split prose text using recursive character splitting."""
        try:
            from langchain.text_splitter import RecursiveCharacterTextSplitter
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=["\n\n", "\n", ". ", " ", ""],
            )
            return [p.strip() for p in splitter.split_text(text) if p.strip()]
        except ImportError:
            # Fallback: simple splitting
            parts = []
            start = 0
            while start < len(text):
                end = start + self.chunk_size
                parts.append(text[start:end].strip())
                start += self.chunk_size - self.chunk_overlap
            return [p for p in parts if p]
