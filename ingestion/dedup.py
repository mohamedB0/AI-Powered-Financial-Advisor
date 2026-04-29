"""
ingestion/dedup.py
Deterministic chunk IDs to prevent re-ingesting duplicate content.
"""

import hashlib
import logging
from typing import Any, Dict, List, Set, Tuple

logger = logging.getLogger(__name__)


def generate_chunk_id(
    source: str,
    text: str,
    pub_date: str = "",
    prefix_len: int = 120,
) -> str:
    """
    Create a deterministic string ID for a chunk.

    Uses ``source + pub_date + first *prefix_len* chars`` of the text
    so the same document ingested twice produces the same IDs.
    The ID is a SHA-256 hash used as the ChromaDB document ID for deterministic chunking.
    """
    key = f"{source}|{pub_date}|{text[:prefix_len]}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def deduplicate(
    records: List[Tuple[str, Dict[str, Any]]],
    existing_ids: Set[str] | None = None,
) -> Tuple[List[Tuple[str, Dict[str, Any]]], List[str]]:
    """
    Remove duplicate (text, metadata) records.

    Args:
        records:      List of (text, metadata) tuples.
        existing_ids: Optional set of IDs already in the vector store;
                      any record whose ID matches will be skipped.

    Returns:
        (unique_records, chunk_ids) — parallel lists ready for upsert.
    """
    existing_ids = existing_ids or set()
    seen: Set[str] = set()
    unique_records: List[Tuple[str, Dict[str, Any]]] = []
    chunk_ids: List[str] = []

    for text, meta in records:
        cid = generate_chunk_id(
            source=meta.get("source", ""),
            text=text,
            pub_date=meta.get("pub_date", ""),
        )

        if cid in seen or cid in existing_ids:
            continue

        seen.add(cid)
        unique_records.append((text, meta))
        chunk_ids.append(cid)

    skipped = len(records) - len(unique_records)
    if skipped:
        logger.info("Dedup: skipped %d duplicate chunks", skipped)
    return unique_records, chunk_ids
