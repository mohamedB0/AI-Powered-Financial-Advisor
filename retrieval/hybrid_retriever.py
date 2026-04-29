"""
retrieval/hybrid_retriever.py
Orchestrates Vector Search and SQL Search with RRF merge and Reranking.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional

from retrieval.vector_search import VectorSearch
from retrieval.sql_search import SQLSearch
from retrieval.reranker import Reranker

logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    Combines semantic (vector) and structured (SQL) search.
    """

    def __init__(
        self,
        vector_search: Optional[VectorSearch] = None,
        sql_search: Optional[SQLSearch] = None,
        reranker: Optional[Reranker] = None,
    ):
        self.vector_search = vector_search or VectorSearch()
        self.sql_search = sql_search or SQLSearch()
        self.reranker = reranker or Reranker()

    async def retrieve(
        self,
        query: str,
        ticker: Optional[str] = None,
        top_k: int = 10,
        rerank: bool = True,
    ) -> Dict[str, Any]:
        """
        Runs vector search and SQL search in parallel and merges results.
        Returns a dict with 'semantic_context' and 'structured_data'.
        """
        # 1. Run searches in parallel (using threads for sync methods)
        loop = asyncio.get_event_loop()
        
        # Vector search
        vector_task = loop.run_in_executor(
            None, self.vector_search.search, query, top_k * 2
        )
        
        # SQL search (if ticker provided, get latest price and metrics)
        sql_results = {}
        if ticker:
            ticker = ticker.upper()
            price_task = loop.run_in_executor(None, self.sql_search.get_latest_price, ticker)
            metrics_task = loop.run_in_executor(None, self.sql_search.get_financial_metrics, ticker)
            
            # Wait for SQL tasks
            prices, metrics = await asyncio.gather(price_task, metrics_task)
            sql_results = {
                "latest_price": prices,
                "metrics": metrics
            }

        # 2. Wait for Vector results
        vector_docs = await vector_task

        # 3. Rerank semantic results
        final_docs = vector_docs
        if rerank and self.reranker:
            final_docs = self.reranker.rerank(query, vector_docs, top_n=top_k)

        # 4. Format contexts
        semantic_context = self.vector_search.format_context(final_docs)
        
        # Combine SQL results into a structured context
        structured_parts = []
        if sql_results.get("latest_price"):
            structured_parts.append("Latest Price Data:\n" + self.sql_search.format_results(sql_results["latest_price"]))
        if sql_results.get("metrics"):
            structured_parts.append("Key Financial Metrics:\n" + self.sql_search.format_results(sql_results["metrics"]))
            
        structured_context = "\n\n".join(structured_parts) if structured_parts else "No structured data available."

        return {
            "semantic_docs": final_docs,
            "semantic_context": semantic_context,
            "structured_results": sql_results,
            "structured_context": structured_context,
        }
