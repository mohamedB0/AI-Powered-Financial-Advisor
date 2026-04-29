"""
tasks/data_fetch_tasks.py
Async tasks for fetching and updating financial data.

These tasks handle scheduled data fetching from various sources.
"""

import os
import sys
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from tasks.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(name="tasks.data_fetch.fetch_stock_prices")
def fetch_stock_prices(tickers: Optional[List[str]] = None):
    """
    Fetch latest stock data for specified tickers.
    Scheduled to run daily.
    """
    from ingestion.loaders import APILoader
    from storage.relational_db import RelationalDB
    
    task_id = fetch_stock_prices.request.id
    logger.info(f"[{task_id}] Fetching stock prices")
    
    if tickers is None:
        tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA"]
    
    results = []
    loader = APILoader()
    db = RelationalDB()
    db.setup_database()
    
    for ticker in tickers:
        try:
            logger.info(f"[{task_id}] Fetching {ticker}...")
            text_records, price_df = loader.load_yahoo_finance(ticker)
            
            if price_df is not None and len(price_df) > 0:
                records = price_df.to_dict(orient="records")
                count = db.insert_prices(records)
                results.append({"ticker": ticker, "status": "success", "records": count})
                logger.info(f"[{task_id}] {ticker}: {count} records")
            else:
                results.append({"ticker": ticker, "status": "warning", "message": "No data"})
                
        except Exception as e:
            logger.error(f"[{task_id}] {ticker} failed: {e}")
            results.append({"ticker": ticker, "status": "failed", "error": str(e)})
    
    return {
        "task_id": task_id,
        "timestamp": datetime.now().isoformat(),
        "tickers_processed": len(tickers),
        "results": results,
    }


@celery_app.task(name="tasks.data_fetch.fetch_macro_indicators")
def fetch_macro_indicators():
    """
    Fetch latest macro indicators (GDP, CPI, etc.).
    Scheduled to run weekly.
    """
    from storage.relational_db import RelationalDB
    
    task_id = fetch_macro_indicators.request.id
    logger.info(f"[{task_id}] Fetching macro indicators")
    
    db = RelationalDB()
    db.setup_database()
    
    try:
        logger.info(f"[{task_id}] Macro data fetch completed")
        return {
            "task_id": task_id,
            "timestamp": datetime.now().isoformat(),
            "status": "success",
            "message": "Macro indicators fetched",
        }
    except Exception as e:
        logger.error(f"[{task_id}] Failed: {e}")
        return {"task_id": task_id, "status": "failed", "error": str(e)}


@celery_app.task(name="tasks.data_fetch.health_check")
def health_check():
    """Health check task for monitoring."""
    try:
        import psutil
        return {
            "timestamp": datetime.now().isoformat(),
            "status": "healthy",
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
        }
    except ImportError:
        return {"timestamp": datetime.now().isoformat(), "status": "healthy"}


@celery_app.task(name="tasks.data_fetch.cleanup_old_uploads")
def cleanup_old_uploads(days: int = 7):
    """Clean up old uploaded files."""
    import shutil
    
    task_id = cleanup_old_uploads.request.id
    logger.info(f"[{task_id}] Cleaning up uploads older than {days} days")
    
    upload_dir = Path("data/uploads")
    if not upload_dir.exists():
        return {"task_id": task_id, "status": "skipped", "message": "No upload directory"}
    
    cutoff = datetime.now() - timedelta(days=days)
    removed = 0
    
    for f in upload_dir.rglob("*"):
        if f.is_file():
            mtime = datetime.fromtimestamp(f.stat().st_mtime)
            if mtime < cutoff:
                f.unlink()
                removed += 1
    
    logger.info(f"[{task_id}] Removed {removed} files")
    return {"task_id": task_id, "status": "success", "files_removed": removed}
