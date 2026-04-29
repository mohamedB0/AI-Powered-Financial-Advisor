"""
ingestion/scheduler.py
APScheduler jobs for incremental / scheduled ingestion.

Runs api_loader hourly, optional RSS scraper every 30 min.
Tracks last_fetched per source to avoid re-fetching recently ingested data.
"""

import logging
from datetime import datetime
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

try:
    from apscheduler.schedulers.background import BackgroundScheduler  # type: ignore
    from apscheduler.triggers.interval import IntervalTrigger  # type: ignore
    _HAS_APSCHEDULER = True
except ImportError:
    _HAS_APSCHEDULER = False


class IngestionScheduler:
    """
    Manages recurring ingestion jobs via APScheduler.

    Usage::

        scheduler = IngestionScheduler()
        scheduler.add_api_job(pipeline, tickers=["AAPL", "MSFT"])
        scheduler.start()
    """

    def __init__(self):
        if not _HAS_APSCHEDULER:
            raise ImportError(
                "APScheduler is required: pip install apscheduler"
            )
        self.scheduler = BackgroundScheduler()
        self._last_fetched: Dict[str, datetime] = {}

    # ── Job management ────────────────────────────────────────────────────────

    def add_api_job(
        self,
        pipeline: Any,
        tickers: list[str],
        interval_hours: int = 1,
    ) -> None:
        """Schedule Yahoo Finance ingestion for *tickers* every N hours."""
        from ingestion.loaders.api_loader import APILoader

        def _job():
            loader = APILoader()
            for ticker in tickers:
                try:
                    text_records, price_df = loader.load_yahoo_finance(ticker)
                    pipeline.ingest_records(text_records)
                    self._last_fetched[f"yahoo:{ticker}"] = datetime.now()
                    logger.info("Scheduled ingest: %s OK", ticker)
                except Exception as e:
                    logger.error("Scheduled ingest: %s failed — %s", ticker, e)

        self.scheduler.add_job(
            _job,
            trigger=IntervalTrigger(hours=interval_hours),
            id="api_loader",
            replace_existing=True,
        )
        logger.info(
            "Scheduled API ingestion every %dh for %s",
            interval_hours,
            tickers,
        )

    def add_custom_job(
        self,
        job_id: str,
        func: Callable,
        interval_minutes: int = 30,
    ) -> None:
        """Add a generic recurring job (e.g. RSS scraper)."""
        self.scheduler.add_job(
            func,
            trigger=IntervalTrigger(minutes=interval_minutes),
            id=job_id,
            replace_existing=True,
        )
        logger.info("Scheduled custom job '%s' every %d min", job_id, interval_minutes)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        self.scheduler.start()
        logger.info("IngestionScheduler started")

    def stop(self) -> None:
        self.scheduler.shutdown()
        logger.info("IngestionScheduler stopped")

    @property
    def last_fetched(self) -> Dict[str, datetime]:
        return dict(self._last_fetched)
