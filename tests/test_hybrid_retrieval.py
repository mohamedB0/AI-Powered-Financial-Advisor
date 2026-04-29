"""
tests/test_hybrid_retrieval.py
Verification script for the new retrieval layer.

Note: These tests require ChromaDB and PostgreSQL services to be running.
Run with: docker-compose up -d
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from retrieval.reranker import Reranker


def test_reranker_instantiation():
    """Test that Reranker can be instantiated without external services."""
    reranker = Reranker()
    assert reranker is not None
    print(" Reranker instantiated successfully.")


@pytest.mark.skipif(
    True,  # Skip by default - requires Docker services
    reason="Requires ChromaDB and PostgreSQL services"
)
def test_retrieval_flow():
    """Integration test that requires Docker services running."""
    from retrieval.sql_search import SQLSearch
    from retrieval.vector_search import VectorSearch
    from retrieval.hybrid_retriever import HybridRetriever
    
    print("--- 1. Testing Component Imports ---")
    sql = SQLSearch()
    vector = VectorSearch()
    reranker = Reranker()
    hybrid = HybridRetriever(vector, sql, reranker)
    print(" All components instantiated.")

    print("\n--- 2. Testing SQLSearch Table Access ---")
    try:
        res = sql.get_latest_price("AAPL")
        print(f" SQL query executed. Result: {res}")
    except Exception as e:
        print(f" SQL query failed: {e}")

    print("\n--- 3. Testing Hybrid Retrieval ---")
    try:
        import asyncio
        async def run_test():
            return await hybrid.retrieve("Apple revenue in 2023", ticker="AAPL", top_k=2)
        
        res = asyncio.run(run_test())
        print(" Hybrid retrieval flow completed.")
        print(f"   Semantic Context Length: {len(res['semantic_context'])}")
        print(f"   Structured Results: {list(res['structured_results'].keys())}")
    except Exception as e:
        print(f" Hybrid retrieval failed: {e}")


if __name__ == "__main__":
    test_retrieval_flow()
