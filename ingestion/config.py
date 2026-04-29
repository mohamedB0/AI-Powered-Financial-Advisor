"""
ingestion/config.py
Centralised configuration for the ingestion pipeline.
"""

import os
from dataclasses import dataclass, field


@dataclass
class IngestionConfig:
    """
    All tunables for the ingestion pipeline in one place.
    Loaders, chunkers, and the pipeline read from this config.
    """

    # ── Chunking ──────────────────────────────────────────────────────────────
    chunk_size: int = 512
    chunk_overlap: int = 64

    # ── Embedding ─────────────────────────────────────────────────────────────
    embed_model: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")
    )
    embed_dimension: int = field(
        default_factory=lambda: int(os.getenv("EMBEDDING_DIMENSION", "1024"))
    )
    embed_batch_size: int = 64

    # ── ChromaDB ───────────────────────────────────────────────────────────────
    chroma_collection: str = field(
        default_factory=lambda: os.getenv("CHROMA_COLLECTION", "financial_data")
    )

    # ── PostgreSQL ────────────────────────────────────────────────────────────
    database_url: str = field(
        default_factory=lambda: os.getenv(
            "DATABASE_URL",
            "postgresql://postgres:root@localhost:5432/findb",
        )
    )

    # ── Pipeline ─────────────────────────────────────────────────────────────
    batch_size: int = 128          # chunks per ChromaDB upsert batch
    dedup_enabled: bool = True     # skip already-indexed chunks
