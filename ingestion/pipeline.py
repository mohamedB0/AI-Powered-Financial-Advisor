"""
ingestion/pipeline.py
Core orchestrator for the ingestion pipeline.

Accepts raw text + metadata → selects chunker → embeds → dedup → upserts
to the vector store.  Used by all loaders and the API ingest route.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from ingestion.config import IngestionConfig
from ingestion.chunkers.recursive import RecursiveChunker
from ingestion.embeddings import Embedder
from ingestion.dedup import deduplicate
from ingestion.metadata import validate_metadata

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """
    End-to-end ingestion: chunk → embed → dedup → store.

    Usage::

        from ingestion import IngestionPipeline

        pipeline = IngestionPipeline()
        pipeline.ingest_file("data/report.pdf")
    """

    def __init__(self, config: IngestionConfig | None = None, collection_override: str | None = None):
        self.config = config or IngestionConfig()
        self._collection_override = collection_override
        self.embedder = Embedder(model_name=self.config.embed_model)
        self._chunker = RecursiveChunker(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )
        self._store: Any = None  # lazy-loaded

    # ── Lazy store access ─────────────────────────────────────────────────────

    @property
    def store(self):
        if self._store is None:
            from storage.chroma_store import ChromaStore
            collection_name = self._collection_override or self.config.chroma_collection
            self._store = ChromaStore(collection_name=collection_name)
        return self._store

    # ── High-level ingest methods ─────────────────────────────────────────────

    def ingest_file(self, file_path: str) -> Dict[str, Any]:
        """
        Load, chunk, embed, and store a single file.

        Selects the appropriate loader based on file extension.

        Supported formats: PDF, HTML, CSV, Excel, Word, Markdown, JSON, Images

        Returns:
            Summary dict with counts.
        """
        from pathlib import Path

        path = Path(file_path)
        ext = path.suffix.lower()
        records = []
        
        is_structured = ext in (".csv", ".tsv", ".xlsx", ".xls")

        # Select loader based on file type
        if ext == ".pdf":
            from ingestion.loaders.pdf_loader import PDFLoader
            records = PDFLoader().load(file_path)
        elif ext in (".html", ".htm"):
            from ingestion.loaders.web_loader import WebLoader
            records = WebLoader().load(file_path)
        elif ext in (".csv", ".tsv", ".xlsx", ".xls"):
            from ingestion.loaders.csv_loader import CSVLoader
            records = CSVLoader().load(file_path)
        elif ext == ".docx":
            from ingestion.loaders.docx_loader import DocxLoader
            records = DocxLoader().load(file_path)
        elif ext in (".png", ".jpg", ".jpeg", ".tiff"):
            from ingestion.loaders.image_loader import ImageLoader
            records = ImageLoader().load(file_path)
        elif ext in (".md", ".markdown"):
            from ingestion.loaders.markdown_loader import MarkdownLoader
            records = MarkdownLoader().load(file_path)
        elif ext == ".json":
            from ingestion.loaders.json_loader import JSONLoader
            records = JSONLoader().load(file_path)
        else:
            raise ValueError(
                f"Unsupported file type: {ext}. "
                f"Supported: pdf, html, csv, xlsx, docx, md, json, png, jpg"
            )

        if not records:
            logger.warning("No content extracted from '%s'", file_path)
            return {"documents": 0, "chunks": 0, "stored": 0}

        # Structured data (Excel, CSV) is stored in PostgreSQL only
        if is_structured:
            return {"documents": len(records), "chunks": 0, "stored": 0, "stored_in": "postgresql"}

        return self.ingest_records(records)

    def ingest_directory(self, directory: str) -> Dict[str, Any]:
        """
        Walk a directory and ingest all supported files.
        """
        import os
        from pathlib import Path

        SUPPORTED = {
            ".pdf", ".html", ".htm",
            ".csv", ".tsv", ".xlsx", ".xls",
            ".docx", ".md", ".markdown", ".json",
            ".png", ".jpg", ".jpeg", ".tiff",
        }
        total = {"documents": 0, "chunks": 0, "stored": 0}

        for root, _, files in os.walk(directory):
            for fname in files:
                if Path(fname).suffix.lower() in SUPPORTED:
                    try:
                        result = self.ingest_file(os.path.join(root, fname))
                        total["documents"] += result["documents"]
                        total["chunks"] += result["chunks"]
                        total["stored"] += result["stored"]
                    except Exception as e:
                        logger.error("Error ingesting '%s': %s", fname, e)

        logger.info(
            "Directory ingest complete: %d docs, %d chunks, %d stored",
            total["documents"],
            total["chunks"],
            total["stored"],
        )
        return total

    def ingest_records(
        self,
        records: List[Tuple[str, Dict[str, Any]]],
        chunker: Any | None = None,
    ) -> Dict[str, Any]:
        """
        Core pipeline: chunk → embed → dedup → store.

        Args:
            records: List of (text, metadata) tuples from any loader.
            chunker: Optional custom chunker; defaults to RecursiveChunker.

        Returns:
            Summary dict.
        """
        chunker = chunker or self._chunker

        # ── 1. Chunk ──────────────────────────────────────────────────────
        all_chunks: List[Tuple[str, Dict[str, Any]]] = []
        for text, meta in records:
            validate_metadata(meta)
            chunks = chunker.chunk(text, meta)
            all_chunks.extend(chunks)

        if not all_chunks:
            return {"documents": len(records), "chunks": 0, "stored": 0}

        # ── 2. Dedup ──────────────────────────────────────────────────────
        if self.config.dedup_enabled:
            all_chunks, chunk_ids = deduplicate(all_chunks)
        else:
            chunk_ids = None

        if not all_chunks:
            logger.info("All chunks were duplicates — nothing to store")
            return {"documents": len(records), "chunks": 0, "stored": 0}

        # ── 3. Embed ──────────────────────────────────────────────────────
        texts = [c[0] for c in all_chunks]
        metadatas = [c[1] for c in all_chunks]
        embeddings = self.embedder.embed(texts)

        # ── 4. Store ──────────────────────────────────────────────────────
        stored = self.store.upsert(
            texts=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=chunk_ids,
            batch_size=self.config.batch_size,
        )

        logger.info(
            "Pipeline complete: %d docs → %d chunks → %d stored",
            len(records),
            len(all_chunks),
            stored,
        )
        return {
            "documents": len(records),
            "chunks": len(all_chunks),
            "stored": stored,
        }

    # ── Utilities ─────────────────────────────────────────────────────────────

    def count(self) -> int:
        """Number of vectors in the collection."""
        return self.store.count()
