"""
storage/models.py
SQLAlchemy ORM models for structured financial data.
"""

from datetime import date, datetime
from sqlalchemy import (
    Column,
    Date,
    DateTime,
    Float,
    Integer,
    BigInteger,
    String,
    Text,
    UniqueConstraint,
    ForeignKey,
    JSON,
)
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class User(Base):
    """Registered users for API authentication."""

    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(150))
    role = Column(String(20), default="user")  # user, admin
    is_active = Column(Integer, default=1)  # 1=active, 0=disabled
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime, nullable=True)

    def __repr__(self) -> str:
        return f"<User {self.email} role={self.role}>"


class StockPrice(Base):
    """Daily OHLCV price records."""

    __tablename__ = "stock_prices"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(10), nullable=False, index=True)
    date = Column(Date, nullable=False, index=True)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(BigInteger)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("ticker", "date", name="uq_price_ticker_date"),
    )

    def __repr__(self) -> str:
        return f"<StockPrice {self.ticker} {self.date} close={self.close}>"


class FinancialMetric(Base):
    """Key financial metrics (PE, EPS, revenue, etc.)."""

    __tablename__ = "financial_metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(10), index=True)
    company = Column(String(100), index=True)
    metric_name = Column(String(50), nullable=False)
    value = Column(Float)
    period = Column(String(10))
    report_date = Column(Date)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint(
            "ticker", "metric_name", "period",
            name="uq_metric_ticker_name_period",
        ),
    )

    def __repr__(self) -> str:
        name = self.ticker or self.company or "Unknown"
        return f"<FinancialMetric {name} {self.metric_name}={self.value}>"


class MacroIndicator(Base):
    """Macroeconomic indicators: GDP, inflation, BVC indices, BKAM rates, HCP data."""

    __tablename__ = "macro_indicators"

    id = Column(Integer, primary_key=True, autoincrement=True)
    country = Column(String(50), index=True)
    indicator_name = Column(String(150), nullable=False, index=True)
    value = Column(Float)
    unit = Column(String(30))
    date = Column(Date, index=True)
    source = Column(String(100))
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self) -> str:
        return f"<MacroIndicator {self.country} {self.indicator_name}={self.value} ({self.date})>"


class Document(Base):
    """Uploaded documents metadata."""

    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, autoincrement=True)
    filename = Column(String(255), nullable=False)
    file_type = Column(String(50), nullable=False)  # pdf, docx, xlsx, csv, json
    file_path = Column(String(500))
    file_size = Column(BigInteger)
    chunks_count = Column(Integer, default=0)
    status = Column(String(50), default="pending")  # pending, processing, completed, failed
    metadata_json = Column(JSON)  # Additional metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self) -> str:
        return f"<Document {self.filename} ({self.file_type})>"


class ExcelData(Base):
    """Generic structured data from Excel/CSV files."""

    __tablename__ = "excel_data"

    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(Integer, ForeignKey("documents.id"), index=True)
    source_file = Column(String(255), index=True)
    sheet_name = Column(String(255))
    row_index = Column(Integer)
    data_json = Column(JSON, nullable=False)  # Row data as JSON
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self) -> str:
        return f"<ExcelData {self.source_file} row={self.row_index}>"


class JSONData(Base):
    """Structured data from JSON files."""

    __tablename__ = "json_data"

    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(Integer, ForeignKey("documents.id"), index=True)
    source_file = Column(String(255), index=True)
    key_path = Column(String(500))  # Path to the data in JSON
    data_json = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self) -> str:
        return f"<JSONData {self.source_file} path={self.key_path}>"


class WordDocument(Base):
    """Word document content and metadata."""

    __tablename__ = "word_documents"

    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(Integer, ForeignKey("documents.id"), index=True)
    source_file = Column(String(255), nullable=False)
    heading = Column(String(500))
    content = Column(Text)
    content_json = Column(JSON)  # For structured extraction
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self) -> str:
        return f"<WordDocument {self.source_file} heading={self.heading}>"


class Conversation(Base):
    """A chat conversation belonging to a user."""

    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    title = Column(String(255), default="New Chat")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("id", "user_id", name="uq_conv_id_user"),
    )

    def __repr__(self) -> str:
        return f"<Conversation {self.id} user={self.user_id} title={self.title}>"


class Message(Base):
    """A single message within a conversation."""

    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False, index=True)
    role = Column(String(20), nullable=False)  # 'user' or 'assistant'
    content = Column(Text, nullable=False)
    route = Column(String(100))  # retrieval route used (Vector, SQL, Web, etc.)
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self) -> str:
        return f"<Message {self.id} conv={self.conversation_id} role={self.role}>"
