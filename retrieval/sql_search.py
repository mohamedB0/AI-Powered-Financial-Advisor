"""
retrieval/sql_search.py
SQL-based retrieval of structured financial data (stocks, metrics, macros).
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import List, Dict, Any, Optional
from storage.relational_db import RelationalDB


class SQLSearch:
    """
    Natural-language-to-SQL retrieval for structured financial data.
    Provides pre-built query templates for common financial questions.
    """

    def __init__(self, db: Optional[RelationalDB] = None):
        self.db = db or RelationalDB()

    # ── Stock Queries ─────────────────────────────────────────────────────────

    def get_latest_price(self, ticker: str) -> List[Dict[str, Any]]:
        sql = """
            SELECT ticker, date, open, high, low, close, volume
            FROM stock_prices
            WHERE ticker = :ticker
            ORDER BY date DESC
            LIMIT 1
        """
        return self.db.execute_sql(sql, {"ticker": ticker.upper()})

    def get_price_history(self, ticker: str, days: int = 30) -> List[Dict[str, Any]]:
        sql = """
            SELECT ticker, date, open, high, low, close, volume
            FROM stock_prices
            WHERE ticker = :ticker
            ORDER BY date DESC
            LIMIT :days
        """
        return self.db.execute_sql(sql, {"ticker": ticker.upper(), "days": days})

    def get_top_movers(self, limit: int = 5) -> List[Dict[str, Any]]:
        sql = """
            SELECT ticker,
                   MAX(close) AS max_close,
                   MIN(close) AS min_close,
                   ROUND(CAST((MAX(close) - MIN(close)) / NULLIF(MIN(close), 0) * 100 AS numeric), 2) AS pct_change
            FROM stock_prices
            WHERE date >= (SELECT MAX(date) FROM stock_prices) - INTERVAL '30 days'
            GROUP BY ticker
            ORDER BY pct_change DESC
            LIMIT :limit
        """
        return self.db.execute_sql(sql, {"limit": limit})

    def get_highest_volume(self, limit: int = 5) -> List[Dict[str, Any]]:
        sql = """
            SELECT ticker,
                   SUM(volume) AS total_volume,
                   ROUND(CAST(AVG(close) AS numeric), 2) AS avg_price
            FROM stock_prices
            WHERE date >= (SELECT MAX(date) FROM stock_prices) - INTERVAL '30 days'
            GROUP BY ticker
            ORDER BY total_volume DESC
            LIMIT :limit
        """
        return self.db.execute_sql(sql, {"limit": limit})

    def get_bvc_latest_prices(self) -> List[Dict[str, Any]]:
        sql = """
            SELECT ticker, date, open, high, low, close, volume
            FROM stock_prices
            WHERE ticker LIKE 'BVC.%'
            ORDER BY date DESC
        """
        return self.db.execute_sql(sql)

    # ── Financial Metrics Queries ─────────────────────────────────────────────

    def get_financial_metrics(self, company: str) -> List[Dict[str, Any]]:
        sql = """
            SELECT ticker, company, metric_name, value, period, report_date
            FROM financial_metrics
            WHERE LOWER(company) LIKE :company OR LOWER(ticker) LIKE :company
            ORDER BY report_date DESC, metric_name
            LIMIT 20
        """
        return self.db.execute_sql(sql, {"company": f"%{company.lower()}%"})

    def get_top_by_revenue(self, year: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get companies ranked by revenue.
        """
        if year:
            sql = """
                SELECT ticker, company, year, revenue
                FROM financial_metrics
                WHERE year = :year AND revenue IS NOT NULL
                ORDER BY revenue DESC
                LIMIT 10
            """
            params = {"year": year}
        else:
            sql = """
                SELECT ticker, company, MAX(year) as year, MAX(revenue) as revenue
                FROM financial_metrics
                WHERE revenue IS NOT NULL
                GROUP BY ticker, company
                ORDER BY revenue DESC
                LIMIT 10
            """
            params = {}
        return self.db.execute_sql(sql, params)

    # ── Macro Queries ─────────────────────────────────────────────────────────

    def format_results(self, rows: List[Dict[str, Any]]) -> str:
        """Format SQL result rows as a readable table string."""
        if not rows:
            return "No structured data found."
        headers = list(rows[0].keys())
        headers = [h for h in headers if h not in ("id", "created_at")]
        lines = [" | ".join(headers)]
        lines.append("-" * len(lines[0]))
        for row in rows:
            lines.append(" | ".join(str(row.get(h, "")) for h in headers))
        return "\n".join(lines)

    # ── Document Queries ────────────────────────────────────────────────────────

    def get_document_list(self, file_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all uploaded documents."""
        if file_type:
            sql = """
                SELECT id, filename, file_type, file_size, status, chunks_count, created_at
                FROM documents
                WHERE file_type = :file_type
                ORDER BY created_at DESC
            """
            return self.db.execute_sql(sql, {"file_type": file_type})
        sql = """
            SELECT id, filename, file_type, file_size, status, chunks_count, created_at
            FROM documents
            ORDER BY created_at DESC
        """
        return self.db.execute_sql(sql)

    def format_documents_for_context(self) -> str:
        """Format all documents as readable context for LLM."""
        docs = self.get_document_list()
        if not docs:
            return "No documents uploaded."
        
        parts = ["## Uploaded Documents\n"]
        for doc in docs:
            parts.append(f"- **{doc['filename']}** ({doc['file_type']}) - {doc['status']}")
        
        parts.append("\n## Document Data Samples\n")
        for doc in docs:
            data = self.get_excel_data(doc['id'])
            if data:
                parts.append(f"\n### {doc['filename']}:")
                for row in data[:3]:
                    parts.append(f"  - {row['data_json']}")
        
        return "\n".join(parts)

    def get_excel_data(self, document_id: int) -> List[Dict[str, Any]]:
        """Get Excel/CSV data for a specific document."""
        sql = """
            SELECT source_file, sheet_name, row_index, data_json
            FROM excel_data
            WHERE document_id = :doc_id
            ORDER BY row_index
        """
        return self.db.execute_sql(sql, {"doc_id": document_id})

    def get_json_data(self, document_id: Optional[int] = None, source_file: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get JSON data, optionally filtered by document or file."""
        if document_id:
            sql = """
                SELECT source_file, key_path, data_json
                FROM json_data
                WHERE document_id = :doc_id
                ORDER BY key_path
            """
            return self.db.execute_sql(sql, {"doc_id": document_id})
        elif source_file:
            sql = """
                SELECT source_file, key_path, data_json
                FROM json_data
                WHERE source_file = :source
                ORDER BY key_path
            """
            return self.db.execute_sql(sql, {"source": source_file})
        sql = """
            SELECT source_file, key_path, data_json
            FROM json_data
            ORDER BY source_file, key_path
        """
        return self.db.execute_sql(sql)

    def get_word_content(self, document_id: Optional[int] = None, source_file: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get Word document content, optionally filtered."""
        if document_id:
            sql = """
                SELECT source_file, heading, content
                FROM word_documents
                WHERE document_id = :doc_id
                ORDER BY id
            """
            return self.db.execute_sql(sql, {"doc_id": document_id})
        elif source_file:
            sql = """
                SELECT source_file, heading, content
                FROM word_documents
                WHERE source_file = :source
                ORDER BY id
            """
            return self.db.execute_sql(sql, {"source": source_file})
        sql = """
            SELECT source_file, heading, content
            FROM word_documents
            ORDER BY source_file, id
        """
        return self.db.execute_sql(sql)
