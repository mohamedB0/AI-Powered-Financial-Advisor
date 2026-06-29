"""
ingestion/chunkers/__init__.py
Re-exports all chunking strategies.
"""

from ingestion.chunkers.recursive import RecursiveChunker

__all__ = ["RecursiveChunker"]
