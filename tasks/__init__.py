"""
tasks/__init__.py
Celery tasks for the Financial Advisor project.

This module contains all async tasks for:
- PDF/Document ingestion
- Stock data fetching
- Scheduled data updates
"""

from tasks.celery_app import celery_app
from tasks.data_fetch_tasks import *
from tasks.document_ingestion_tasks import *
from tasks.local_ingestion_tasks import *
from tasks.config_driven_scraping_tasks import *

__all__ = ["celery_app"]
