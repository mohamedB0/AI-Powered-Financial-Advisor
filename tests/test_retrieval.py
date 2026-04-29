"""
tests/test_retrieval.py
Basic tests for the hybrid retrieval system (SQL + Vector).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from retrieval.hybrid_merger import HybridMerger


# ── HybridMerger Tests ────────────────────────────────────────────────────────

@pytest.fixture
def sample_vector_results():
    return [
        {"text": "Amazon revenue grew 12% in 2023.", "score": 0.91, "source": "amazon_10k.pdf", "page": 5},
        {"text": "Strong AWS segment drove EBITDA improvement.", "score": 0.85, "source": "amazon_10k.pdf", "page": 12},
    ]


@pytest.fixture
def sample_sql_results():
    return [
        {"ticker": "AMZN", "date": "2024-01-10", "close": 185.4, "volume": 45000000},
        {"ticker": "AMZN", "date": "2024-01-09", "close": 183.2, "volume": 42000000},
    ]


def test_merger_produces_string(sample_vector_results, sample_sql_results):
    merger = HybridMerger()
    context = merger.merge(sample_vector_results, sample_sql_results)
    assert isinstance(context, str)
    assert len(context) > 0


def test_merger_includes_doc_excerpts(sample_vector_results, sample_sql_results):
    merger = HybridMerger()
    context = merger.merge(sample_vector_results, sample_sql_results)
    assert "amazon_10k.pdf" in context or "Document Excerpts" in context


def test_merger_includes_sql_table(sample_vector_results, sample_sql_results):
    merger = HybridMerger()
    context = merger.merge(sample_vector_results, sample_sql_results)
    assert "AMZN" in context or "Structured" in context


def test_merger_empty_inputs():
    merger = HybridMerger()
    context = merger.merge([], [])
    assert "No context" in context


def test_merger_only_vector(sample_vector_results):
    merger = HybridMerger()
    context = merger.merge(sample_vector_results, [])
    assert "Amazon revenue" in context


def test_merger_only_sql(sample_sql_results):
    merger = HybridMerger()
    context = merger.merge([], sample_sql_results)
    assert "AMZN" in context
