"""
tests/test_ingestion.py
Smoke tests for the ingestion pipeline components.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from unittest.mock import patch, MagicMock


# ── Fixtures ─────────────────────────────────────────────────────────────────

SAMPLE_TEXT = "Apple reported revenue of $383 billion in fiscal 2023 driven by strong iPhone sales."
SAMPLE_METADATA = {"source": "test.pdf", "page": 1, "doc_type": "pdf"}


# ── Metadata Tests ────────────────────────────────────────────────────────────

def test_metadata_validation_passes():
    from ingestion.metadata import validate_metadata
    meta = {"source": "report.pdf", "doc_type": "pdf", "page": 1}
    result = validate_metadata(meta)
    assert result["source"] == "report.pdf"


def test_metadata_validation_fails():
    from ingestion.metadata import validate_metadata, MetadataError
    with pytest.raises(MetadataError):
        validate_metadata({"page": 1})  # missing source & doc_type


# ── Chunker Tests ─────────────────────────────────────────────────────────────

def test_recursive_chunker_splits_text():
    from ingestion.chunkers.recursive import RecursiveChunker
    chunker = RecursiveChunker(chunk_size=50, chunk_overlap=10)
    chunks = chunker.chunk(SAMPLE_TEXT, SAMPLE_METADATA)
    assert len(chunks) >= 1
    for text, meta in chunks:
        assert text.strip()
        assert "source" in meta
        assert "chunk_index" in meta


def test_recursive_chunker_metadata_preserved():
    from ingestion.chunkers.recursive import RecursiveChunker
    chunks = RecursiveChunker().chunk(SAMPLE_TEXT, SAMPLE_METADATA)
    assert chunks[0][1]["source"] == "test.pdf"
    assert "chunk_index" in chunks[0][1]


# ── Embedder Tests ────────────────────────────────────────────────────────────

def test_embedder_returns_vectors():
    from ingestion.embeddings import Embedder
    embedder = Embedder()
    vectors = embedder.embed(["Apple revenue 2023", "stock market analysis"])
    assert len(vectors) == 2
    assert len(vectors[0]) == embedder.dimension


def test_embedder_single_text():
    from ingestion.embeddings import Embedder
    embedder = Embedder()
    vec = embedder.embed_one("financial analysis")
    assert isinstance(vec, list)
    assert len(vec) == embedder.dimension


# ── Dedup Tests ───────────────────────────────────────────────────────────────

def test_dedup_removes_duplicates():
    from ingestion.dedup import deduplicate
    records = [
        (SAMPLE_TEXT, SAMPLE_METADATA),
        (SAMPLE_TEXT, SAMPLE_METADATA),  # exact duplicate
        ("Different text entirely", {"source": "other.pdf", "doc_type": "pdf"}),
    ]
    unique, ids = deduplicate(records)
    assert len(unique) == 2
    assert len(ids) == 2


# ── Config Tests ──────────────────────────────────────────────────────────────

def test_config_defaults():
    from ingestion.config import IngestionConfig
    config = IngestionConfig()
    assert config.chunk_size == 512
    assert config.chunk_overlap == 64
    assert config.embed_batch_size == 64


# ── Full Pipeline Smoke Test ────────────────────────────────────────────────

def test_full_pipeline_smoke():
    from ingestion.chunkers.recursive import RecursiveChunker
    from ingestion.embeddings import Embedder

    chunker = RecursiveChunker(chunk_size=100, chunk_overlap=20)
    chunks = chunker.chunk(SAMPLE_TEXT, SAMPLE_METADATA)
    assert len(chunks) >= 1

    embedder = Embedder()
    texts = [c[0] for c in chunks]
    embeddings = embedder.embed(texts)
    assert len(embeddings) == len(chunks)
    assert all(len(e) == embedder.dimension for e in embeddings)
