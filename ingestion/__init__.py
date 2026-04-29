"""
ingestion/__init__.py
Exposes IngestionPipeline as the public interface for the ingestion package.
"""

from ingestion.pipeline import IngestionPipeline
from ingestion.config import IngestionConfig

__all__ = ["IngestionPipeline", "IngestionConfig"]
