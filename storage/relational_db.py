"""
storage/relational_db.py
SQLAlchemy engine + connection pool for PostgreSQL.
Exposes typed insert methods for structured financial data.
"""

import logging
import os
from datetime import date
from typing import Any, Dict, List, Optional

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.pool import NullPool

from storage.models import Base, StockPrice, FinancialMetric, Document, ExcelData, JSONData, WordDocument

logger = logging.getLogger(__name__)


class RelationalDB:
    """
    PostgreSQL wrapper using SQLAlchemy ORM.
    """

    def __init__(self, database_url: str | None = None):
        if database_url:
            self.database_url = database_url
        elif os.getenv("DATABASE_URL"):
            self.database_url = os.getenv("DATABASE_URL", "")
        else:
            pg_host = os.getenv("POSTGRES_HOST", "localhost")
            pg_port = os.getenv("POSTGRES_PORT", "5432")
            pg_db = os.getenv("POSTGRES_DB", "financial_advisor")
            pg_user = os.getenv("POSTGRES_USER", "postgres")
            pg_pass = os.getenv("POSTGRES_PASSWORD", "root")
            sslmode = os.getenv("POSTGRES_SSLMODE", "")
            query = f"?sslmode={sslmode}" if sslmode else ""
            self.database_url = f"postgresql+psycopg2://{pg_user}:{pg_pass}@{pg_host}:{pg_port}/{pg_db}{query}"

        # Disable client-side pooling when using external poolers (e.g. Supabase)
        use_null_pool = (
            "pooler.supabase.com" in self.database_url
            or os.getenv("POSTGRES_POOL_NULL", "").lower() == "true"
        )
        engine_kwargs: Dict[str, Any] = {"pool_pre_ping": True}
        if use_null_pool:
            engine_kwargs["poolclass"] = NullPool

        self.engine = create_engine(self.database_url, **engine_kwargs)
        self.SessionLocal = sessionmaker(bind=self.engine)
        logger.info("RelationalDB: connected to %s", self.database_url.split("@")[-1])

    def setup_database(self) -> None:
        """Create all ORM tables if they don't exist."""
        Base.metadata.create_all(self.engine)
        logger.info("RelationalDB: tables created / verified")

    @property
    def session(self) -> Session:
        """Provide a new session."""
        return self.SessionLocal()

    # ── Insert/Upsert methods ────────────────────────────────────────────────

    def insert_prices(self, records: List[Dict[str, Any]]) -> int:
        """
        Upsert stock price rows (skip on ticker+date conflict).
        """
        if not records:
            return 0
        with self.SessionLocal() as session:
            stmt = insert(StockPrice).values(records)
            stmt = stmt.on_conflict_do_nothing(index_elements=["ticker", "date"])
            result = session.execute(stmt)
            session.commit()
            count = result.rowcount
            logger.info("Upserted %d price records", count)
            return count

    def insert_metrics(self, records: List[Dict[str, Any]]) -> int:
        """
        Upsert financial metric rows (skip on ticker+metric+period conflict).
        """
        if not records:
            return 0
        with self.SessionLocal() as session:
            stmt = insert(FinancialMetric).values(records)
            stmt = stmt.on_conflict_do_nothing(
                index_elements=["ticker", "metric_name", "period"]
            )
            result = session.execute(stmt)
            session.commit()
            count = result.rowcount
            logger.info("Upserted %d metric records", count)
            return count

    def insert_document(self, filename: str, file_type: str, file_size: int = 0, metadata: Optional[Dict] = None) -> int:
        """
        Insert a document record.
        
        Returns the document ID.
        """
        with self.SessionLocal() as session:
            doc = Document(
                filename=filename,
                file_type=file_type,
                file_size=file_size,
                metadata_json=metadata,
                status="processing",
            )
            session.add(doc)
            session.commit()
            doc_id = doc.id
            logger.info("Inserted document: %s (id=%d)", filename, doc_id)
            return doc_id

    def update_document_status(self, doc_id: int, status: str, chunks_count: int = 0) -> None:
        """Update document processing status."""
        with self.SessionLocal() as session:
            doc = session.query(Document).filter(Document.id == doc_id).first()
            if doc:
                doc.status = status
                doc.chunks_count = chunks_count
                session.commit()
                logger.info("Updated document %d status to %s", doc_id, status)

    def insert_excel_data(self, document_id: int, source_file: str, rows: List[Dict], sheet_name: str = "Sheet1") -> int:
        """
        Insert Excel/CSV data rows.
        
        Returns number of rows inserted.
        """
        if not rows:
            return 0
        with self.SessionLocal() as session:
            excel_records = [
                ExcelData(
                    document_id=document_id,
                    source_file=source_file,
                    sheet_name=sheet_name,
                    row_index=i,
                    data_json=row,
                )
                for i, row in enumerate(rows)
            ]
            session.add_all(excel_records)
            session.commit()
            count = len(excel_records)
            logger.info("Inserted %d Excel rows for document %d", count, document_id)
            return count

    def insert_json_data(self, document_id: int, source_file: str, data: Any, key_path: str = "") -> int:
        """
        Insert JSON data.
        
        Returns the JSON data record ID.
        """
        with self.SessionLocal() as session:
            json_record = JSONData(
                document_id=document_id,
                source_file=source_file,
                key_path=key_path,
                data_json=data,
            )
            session.add(json_record)
            session.commit()
            record_id = json_record.id
            logger.info("Inserted JSON data for document %d (id=%d)", document_id, record_id)
            return record_id

    def insert_word_content(self, document_id: int, source_file: str, heading: str, content: str) -> int:
        """
        Insert Word document content.
        
        Returns the Word document record ID.
        """
        with self.SessionLocal() as session:
            word_record = WordDocument(
                document_id=document_id,
                source_file=source_file,
                heading=heading,
                content=content,
            )
            session.add(word_record)
            session.commit()
            record_id = word_record.id
            logger.info("Inserted Word content for document %d (id=%d)", document_id, record_id)
            return record_id

    # ── Query helpers ─────────────────────────────────────────────────────────

    def execute_sql(self, sql: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Run a raw SQL query and return rows as dicts. Supports SELECT and DML."""
        with self.engine.connect() as conn:
            result = conn.execute(text(sql), params or {})
            # DML statements (INSERT, UPDATE, DELETE) won't have rows to fetch
            if result.returns_rows:
                columns = list(result.keys())
                return [dict(zip(columns, row)) for row in result.fetchall()]
            conn.commit()
            return []
