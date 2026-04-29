"""
ingestion/loaders/csv_loader.py
Loads CSV and Excel files, validates with pandas, and generates
row-to-text summaries for the vector store.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple
import math

from ingestion.metadata import build_metadata, validate_metadata

logger = logging.getLogger(__name__)

try:
    import pandas as pd  # type: ignore
    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False


def _clean_value(val: Any) -> Any:
    """Convert NaN/inf to None for JSON compatibility."""
    if val is None:
        return None
    if isinstance(val, float):
        if math.isnan(val) or math.isinf(val):
            return None
    return val


def _clean_dict(d: Dict) -> Dict:
    """Clean a dictionary by replacing NaN/inf with None."""
    return {k: _clean_value(v) for k, v in d.items()}


class CSVLoader:
    """
    Loads CSV / Excel files and returns (text, metadata) tuples.

    Each row (or group of rows) is converted to a natural-language summary
    suitable for embedding.  The raw DataFrame is also available for
    insertion into PostgreSQL via the storage layer.
    """

    def __init__(self, rows_per_chunk: int = 20):
        self.rows_per_chunk = rows_per_chunk

    def load(self, file_path: str) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Read a CSV or Excel file and return text summaries.
        
        Also stores structured data in PostgreSQL.
        """
        if not _HAS_PANDAS:
            raise ImportError("pandas is required: pip install pandas")

        path = Path(file_path)
        ext = path.suffix.lower()

        if ext in (".csv", ".tsv"):
            df = pd.read_csv(file_path)
        elif ext in (".xlsx", ".xls"):
            df = pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

        results: List[Tuple[str, Dict[str, Any]]] = []
        
        # Normalize column names for processing
        df.columns = df.columns.str.lower()
        
        # Normalize date column if present
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date

        # Chunk rows to keep text size manageable
        for start in range(0, len(df), self.rows_per_chunk):
            chunk_df = df.iloc[start : start + self.rows_per_chunk]
            text = self._dataframe_to_text(chunk_df, path.name)

            meta = build_metadata(
                source=path.name,
                doc_type="csv",
                file_path=str(path),
                row_start=start,
                row_end=start + len(chunk_df),
                total_rows=len(df),
            )
            validate_metadata(meta)
            results.append((text, meta))

        # ── Insert into PostgreSQL ────────────────────────────────────────
        try:
            from storage.relational_db import RelationalDB
            db = RelationalDB()
            db.setup_database()
            
            # Create document record
            clean_cols = [_clean_value(c) for c in df.columns]
            doc_id = db.insert_document(
                filename=path.name,
                file_type="csv",
                file_size=path.stat().st_size,
                metadata={"columns": clean_cols, "rows": len(df)},
            )
            
            cols = set(df.columns)
            
            if {"ticker", "date", "close"}.issubset(cols):
                # StockPrice data
                rows = df.to_dict(orient="records")
                db.insert_prices(rows)
                logger.info("CSVLoader: Inserted %d stock prices", len(rows))
            elif {"ticker", "metric_name", "value"}.issubset(cols) or {"company", "metric_name", "value"}.issubset(cols):
                # FinancialMetric data
                rows = df.to_dict(orient="records")
                db.insert_metrics(rows)
                logger.info("CSVLoader: Inserted %d financial metrics", len(rows))
            else:
                # Generic structured data
                rows = [_clean_dict(r) for r in df.to_dict(orient="records")]
                db.insert_excel_data(doc_id, path.name, rows)
                logger.info("CSVLoader: Inserted %d rows into excel_data", len(rows))
            
            # Update document status
            db.update_document_status(doc_id, "completed", chunks_count=len(results))
            
        except Exception as e:
            logger.warning("CSVLoader: Failed to insert into SQL: %s", e)

        logger.info(
            "CSVLoader: loaded %d chunks from '%s' (%d rows)",
            len(results),
            path.name,
            len(df),
        )
        return results

    def load_dataframe(self, file_path: str) -> Any:
        """Return the raw pandas DataFrame for SQL insertion."""
        if not _HAS_PANDAS:
            raise ImportError("pandas is required: pip install pandas")

        path = Path(file_path)
        ext = path.suffix.lower()

        if ext in (".csv", ".tsv"):
            return pd.read_csv(file_path)
        elif ext in (".xlsx", ".xls"):
            return pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

    # ── Internal ──────────────────────────────────────────────────────────────

    @staticmethod
    def _dataframe_to_text(df: Any, source_name: str) -> str:
        """Convert a DataFrame chunk to a natural-language summary."""
        cols = [str(c) if c is not None and not (isinstance(c, float) and str(c) == 'nan') else f"col_{i}" 
                for i, c in enumerate(df.columns)]
        lines = [f"Data from {source_name}:", f"Columns: {', '.join(cols)}"]
        for _, row in df.iterrows():
            parts = [f"{cols[i]}: {val}" for i, val in enumerate(row.values)]
            lines.append(" | ".join(parts))
        return "\n".join(lines)
