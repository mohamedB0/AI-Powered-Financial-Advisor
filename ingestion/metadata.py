"""
ingestion/metadata.py
Defines and validates the required metadata schema for ingested documents.
"""

from typing import Any, Dict, Optional


REQUIRED_FIELDS = {"source", "doc_type"}
OPTIONAL_FIELDS = {"company", "report_type", "pub_date", "sector", "page", "chunk_index"}


class MetadataError(Exception):
    """Raised when a loader omits a required metadata field."""


def validate_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure *metadata* contains every required field.

    Raises:
        MetadataError: if any required key is missing or empty.
    """
    missing = REQUIRED_FIELDS - set(metadata.keys())
    if missing:
        raise MetadataError(
            f"Metadata is missing required fields: {missing}. "
            f"Got keys: {set(metadata.keys())}"
        )

    for key in REQUIRED_FIELDS:
        if metadata[key] is None or (isinstance(metadata[key], str) and not metadata[key].strip()):
            raise MetadataError(f"Required metadata field '{key}' is empty.")

    return metadata


def build_metadata(
    source: str,
    doc_type: str,
    *,
    company: Optional[str] = None,
    report_type: Optional[str] = None,
    pub_date: Optional[str] = None,
    sector: Optional[str] = None,
    page: Optional[int] = None,
    **extra: Any,
) -> Dict[str, Any]:
    """
    Helper to build a metadata dict with required + optional fields.
    """
    meta: Dict[str, Any] = {"source": source, "doc_type": doc_type}

    if company:
        meta["company"] = company
    if report_type:
        meta["report_type"] = report_type
    if pub_date:
        meta["pub_date"] = pub_date
    if sector:
        meta["sector"] = sector
    if page is not None:
        meta["page"] = page

    meta.update(extra)
    return meta
