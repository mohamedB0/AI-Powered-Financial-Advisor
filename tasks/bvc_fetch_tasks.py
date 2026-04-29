"""
tasks/bvc_fetch_tasks.py
Async tasks for fetching BVC (Bourse de Casablanca) market data.
Stores structured OHLCV data in PostgreSQL.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from tasks.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(name="tasks.bvc_fetch.fetch_bvc_daily")
def fetch_bvc_daily():
    """
    Fetch daily BVC market data (stock prices and indices).
    Scheduled to run daily during market hours.
    """
    task_id = fetch_bvc_daily.request.id
    logger.info("[%s] Starting BVC daily fetch", task_id)

    try:
        from ingestion.loaders.bvc_loader import BVCLoader
        from storage.relational_db import RelationalDB

        db = RelationalDB()
        db.setup_database()

        loader = BVCLoader()
        results = loader.fetch_all()

        logger.info("[%s] BVC fetch completed: %s", task_id, results)
        return {
            "task_id": task_id,
            "timestamp": datetime.now().isoformat(),
            "status": "success",
            "results": results,
        }

    except Exception as e:
        logger.error("[%s] BVC fetch failed: %s", task_id, e)
        return {
            "task_id": task_id,
            "timestamp": datetime.now().isoformat(),
            "status": "failed",
            "error": str(e),
        }


@celery_app.task(name="tasks.bvc_fetch.fetch_bvc_cotation")
def fetch_bvc_cotation():
    """
    Fetch only BVC stock cotation (prices).
    Can be triggered on-demand.
    """
    task_id = fetch_bvc_cotation.request.id
    logger.info("[%s] Starting BVC cotation fetch", task_id)

    try:
        from ingestion.loaders.bvc_loader import BVCLoader
        from storage.relational_db import RelationalDB

        db = RelationalDB()
        db.setup_database()

        loader = BVCLoader()
        text_records, price_df = loader.fetch_cotation()

        count = len(price_df) if not price_df.empty else 0
        tickers = price_df["ticker"].unique().tolist() if not price_df.empty else []

        logger.info("[%s] BVC cotation completed: %d records, %d tickers", task_id, count, len(tickers))
        return {
            "task_id": task_id,
            "timestamp": datetime.now().isoformat(),
            "status": "success",
            "records": count,
            "tickers": tickers,
        }

    except Exception as e:
        logger.error("[%s] BVC cotation fetch failed: %s", task_id, e)
        return {
            "task_id": task_id,
            "timestamp": datetime.now().isoformat(),
            "status": "failed",
            "error": str(e),
        }


@celery_app.task(name="tasks.bvc_fetch.fetch_bvc_indices")
def fetch_bvc_indices():
    """
    Fetch only BVC index data.
    Can be triggered on-demand.
    """
    task_id = fetch_bvc_indices.request.id
    logger.info("[%s] Starting BVC indices fetch", task_id)

    try:
        from ingestion.loaders.bvc_loader import BVCLoader
        from storage.relational_db import RelationalDB

        db = RelationalDB()
        db.setup_database()

        loader = BVCLoader()
        text_records, index_df = loader.fetch_indices()

        count = len(index_df) if not index_df.empty else 0

        logger.info("[%s] BVC indices completed: %d records", task_id, count)
        return {
            "task_id": task_id,
            "timestamp": datetime.now().isoformat(),
            "status": "success",
            "records": count,
        }

    except Exception as e:
        logger.error("[%s] BVC indices fetch failed: %s", task_id, e)
        return {
            "task_id": task_id,
            "timestamp": datetime.now().isoformat(),
            "status": "failed",
            "error": str(e),
        }
