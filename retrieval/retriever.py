"""
retrieval/retriever.py
Unified retrieval interface — orchestrates vector search + SQL + hybrid merge.
"""

import re
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import Dict, Any, Optional
from retrieval.vector_search import VectorSearch
from retrieval.sql_search import SQLSearch
from retrieval.hybrid_merger import HybridMerger


# Keywords that signal the query needs structured SQL data
_SQL_SIGNALS = [
    r"\bprice\b", r"\bstock\b", r"\breturn\b", r"\bclose\b",
    r"\brevenue\b", r"\bebitda\b", r"\bnet.?income\b", r"\beps\b",
    r"\bgdp\b", r"\binflation\b", r"\bunemployment\b",
    r"\bvolume\b", r"\bhigh\b", r"\blow\b", r"\bopen\b",
    r"\b(aapl|msft|amzn|googl|meta|tsla|nvda)\b",
]


def _needs_sql(query: str) -> bool:
    q = query.lower()
    return any(re.search(pat, q) for pat in _SQL_SIGNALS)


def _extract_ticker(query: str) -> Optional[str]:
    tickers = ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "NVDA"]
    q = query.upper()
    for t in tickers:
        if t in q:
            return t
    return None


class Retriever:
    """
    High-level retrieval interface.
    Automatically decides whether to run vector search, SQL, or both.
    """

    def __init__(self):
        self.vector   = VectorSearch()
        self.sql      = SQLSearch()
        self.merger   = HybridMerger()

    def query(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Execute hybrid retrieval for a natural language question.

        Returns:
            dict with:
              - 'context': merged string for LLM
              - 'vector_results': raw vector hits
              - 'sql_results': raw SQL rows
        """
        vector_results = []
        sql_results    = []

        # Always do vector search for document context
        try:
            vector_results = self.vector.search(question, top_k=top_k)
        except Exception as e:
            print(f"[Retriever] Vector search error: {e}")

        # Conditionally run SQL search
        if _needs_sql(question):
            ticker = _extract_ticker(question)
            try:
                if ticker:
                    sql_results = self.sql.get_price_history(ticker, days=10)
                elif any(kw in question.lower() for kw in ["gdp", "inflation", "unemployment"]):
                    sql_results = self.sql.get_macro_indicator("United States", "gdp")
                elif "revenue" in question.lower() or "earnings" in question.lower():
                    sql_results = self.sql.get_top_by_revenue()
                else:
                    sql_results = self.sql.get_top_movers()
            except Exception as e:
                print(f"[Retriever] SQL search error: {e}")

        context = self.merger.merge(vector_results, sql_results)

        return {
            "context"        : context,
            "vector_results" : vector_results,
            "sql_results"    : sql_results,
        }
