"""
ingestion/loaders/financial_parser_loader.py
Loads XLSX/CSV files using ParserAI library.
Provides RAG-ready chunks with token counting and source URI tracking.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

from ingestion.metadata import build_metadata, validate_metadata

logger = logging.getLogger(__name__)

# Try to import ParserAI, fallback to legacy CSVLoader if not available
try:
    from financial_parser import FinancialParser
    _HAS_PARSERAI = True
except ImportError:
    _HAS_PARSERAI = False
    logger.warning("ParserAI not installed. Install with: pip install git+https://github.com/mohamedB0/ParserAI.git")


class FinancialParserLoader:
    """
    Loads XLSX/CSV files using ParserAI for RAG-ready output.
    
    Features:
    - Parses merged cells, multi-sheets, formulas
    - Auto-detects financial statement types (balance sheet, income statement, etc.)
    - Token-aware chunking via tiktoken
    - Source URI tracking for citations
    
    Returns (text, metadata) tuples compatible with IngestionPipeline.
    """

    def __init__(self, max_tokens_per_chunk: int = 800):
        self.max_tokens_per_chunk = max_tokens_per_chunk
        self._parser = FinancialParser() if _HAS_PARSERAI else None

    def load(self, file_path: str) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Parse XLSX/CSV and return RAG-ready chunks.
        
        Returns list of (text, metadata) tuples for ChromaDB ingestion.
        """
        if not _HAS_PARSERAI:
            raise ImportError(
                "ParserAI not installed. Install with: "
                "pip install git+https://github.com/mohamedB0/ParserAI.git"
            )

        path = Path(file_path)
        ext = path.suffix.lower()

        if ext not in (".csv", ".tsv", ".xlsx", ".xls"):
            raise ValueError(f"Unsupported file type: {ext}")

        logger.info(f"FinancialParserLoader: Parsing {path.name} with ParserAI")

        # Parse the document
        try:
            doc = self._parser.parse(file_path)
        except Exception as e:
            logger.error(f"FinancialParserLoader: Failed to parse {path.name}: {e}")
            raise

        results: List[Tuple[str, Dict[str, Any]]] = []

        # Process each RAG chunk from ParserAI
        for chunk in doc.chunks:
            # Build metadata
            meta = build_metadata(
                source=path.name,
                doc_type=chunk.chunk_type,
                file_path=str(path),
                collection="user_data",
            )
            
            # Add ParserAI-specific metadata
            meta.update({
                "source_uri": chunk.source_uri,
                "token_count": chunk.token_count,
                "statement_type": chunk.metadata.get("statement_type", "unknown"),
                "sheet_name": chunk.metadata.get("sheet_name"),
                "row_start": chunk.metadata.get("row_start"),
                "row_end": chunk.metadata.get("row_end"),
            })
            
            validate_metadata(meta)
            results.append((chunk.content_text, meta))

        # Also insert structured data into PostgreSQL
        self._insert_to_postgresql(doc, path)

        logger.info(
            f"FinancialParserLoader: Loaded {len(results)} chunks from '{path.name}'"
        )
        return results

    def _insert_to_postgresql(self, doc, path: Path) -> None:
        """Insert structured data into PostgreSQL."""
        try:
            from storage.relational_db import RelationalDB
            db = RelationalDB()
            db.setup_database()

            # Create document record
            doc_id = db.insert_document(
                filename=path.name,
                file_type=path.suffix.lower().lstrip("."),
                file_size=path.stat().st_size,
                metadata={"sheets": len(doc.sheets), "chunks": len(doc.chunks)},
            )

            # Extract and insert structured data from sheets
            for sheet in doc.sheets:
                for table in sheet.tables:
                    rows = table.rows
                    if not rows:
                        continue

                    # Determine data type based on statement type
                    statement = table.statement_type.value if table.statement_type else "unknown"

                    # Convert to dict records
                    records = []
                    headers = table.headers or []
                    for row in rows:
                        record = {}
                        for i, val in enumerate(row):
                            key = headers[i] if i < len(headers) else f"col_{i}"
                            record[key] = val
                        records.append(record)

                    # Insert based on detected type
                    if statement == "balance_sheet" or statement == "income_statement":
                        # Try as financial metrics
                        db.insert_excel_data(doc_id, f"{path.name}!{sheet.name}", records)
                    elif statement == "cash_flow":
                        db.insert_excel_data(doc_id, f"{path.name}!{sheet.name}", records)
                    else:
                        # Generic structured data
                        db.insert_excel_data(doc_id, f"{path.name}!{sheet.name}", records)

            db.update_document_status(doc_id, "completed", chunks_count=len(doc.chunks))
            logger.info(f"FinancialParserLoader: Inserted {path.name} into PostgreSQL")

        except Exception as e:
            logger.warning(f"FinancialParserLoader: Failed to insert into SQL: {e}")

    def get_supported_extensions(self) -> List[str]:
        """Return list of supported file extensions."""
        return [".csv", ".tsv", ".xlsx", ".xls"]