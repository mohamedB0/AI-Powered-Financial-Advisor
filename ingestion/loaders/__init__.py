"""
ingestion/loaders/__init__.py
Re-exports all loaders so callers can do:
    from ingestion.loaders import PDFLoader, WebLoader, ...
"""

from ingestion.loaders.pdf_loader import PDFLoader
from ingestion.loaders.web_loader import WebLoader
from ingestion.loaders.api_loader import APILoader
from ingestion.loaders.csv_loader import CSVLoader
from ingestion.loaders.docx_loader import DocxLoader
from ingestion.loaders.image_loader import ImageLoader
from ingestion.loaders.markdown_loader import MarkdownLoader
from ingestion.loaders.json_loader import JSONLoader

__all__ = [
    "PDFLoader",
    "WebLoader", 
    "APILoader",
    "CSVLoader",
    "DocxLoader",
    "ImageLoader",
    "MarkdownLoader",
    "JSONLoader",
]
