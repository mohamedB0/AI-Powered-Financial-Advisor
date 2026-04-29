"""
ingestion/loaders/api_loader.py
Fetches financial data from external APIs (Yahoo Finance, etc.).

Numeric data is routed to PostgreSQL via the storage layer.
Text summaries are returned as (text, metadata) tuples for the vector store.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from ingestion.metadata import build_metadata, validate_metadata

logger = logging.getLogger(__name__)

# Optional: yfinance for Yahoo Finance data
try:
    import yfinance as yf  # type: ignore
    _HAS_YFINANCE = True
except ImportError:
    _HAS_YFINANCE = False

try:
    import pandas as pd  # type: ignore
    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False


class APILoader:
    """
    Fetches data from financial APIs and returns (text, metadata) tuples.

    Also provides structured DataFrames for relational storage.
    """

    def __init__(self, max_retries: int = 3, timeout: int = 30):
        self.max_retries = max_retries
        self.timeout = timeout

    # ── Yahoo Finance ─────────────────────────────────────────────────────────

    def load_yahoo_finance(
        self,
        ticker: str,
        period: str = "1y",
    ) -> Tuple[List[Tuple[str, Dict[str, Any]]], Any]:
        """
        Fetch stock data from Yahoo Finance.

        Returns:
            (text_records, price_df) — text for vector store, DataFrame for SQL.
        """
        if not _HAS_YFINANCE:
            raise ImportError("yfinance is required: pip install yfinance")

        stock = yf.Ticker(ticker)
        info = stock.info or {}
        hist = stock.history(period=period)

        # ── Build text summary for vector store ──────────────────────────
        summary_parts = [
            f"Company: {info.get('longName', ticker)}",
            f"Sector: {info.get('sector', 'N/A')}",
            f"Industry: {info.get('industry', 'N/A')}",
            f"Market Cap: {info.get('marketCap', 'N/A')}",
            f"PE Ratio: {info.get('trailingPE', 'N/A')}",
            f"52-Week High: {info.get('fiftyTwoWeekHigh', 'N/A')}",
            f"52-Week Low: {info.get('fiftyTwoWeekLow', 'N/A')}",
        ]
        summary_text = "\n".join(summary_parts)

        meta = build_metadata(
            source=f"yahoo_finance:{ticker}",
            doc_type="api",
            company=info.get("longName", ticker),
            sector=info.get("sector"),
            pub_date=datetime.now().strftime("%Y-%m-%d"),
        )
        validate_metadata(meta)

        text_records = [(summary_text, meta)]

        # ── Insert into PostgreSQL ───────────────────────────────────────
        if not hist.empty:
            from storage.relational_db import RelationalDB
            db = RelationalDB()
            
            # Prepare rows for StockPrice model
            hist_reset = hist.reset_index()
            # Handle multi-index columns if present
            hist_reset.columns = [c[0] if isinstance(c, tuple) else c for c in hist_reset.columns]
            
            df_prepared = hist_reset.rename(columns={
                "Date": "date", "Open": "open", "High": "high",
                "Low": "low", "Close": "close", "Volume": "volume"
            })
            df_prepared["ticker"] = ticker
            # Select only columns matching the model
            records = df_prepared[["ticker", "date", "open", "high", "low", "close", "volume"]].to_dict(orient="records")
            db.insert_prices(records)

        logger.info(
            "APILoader: fetched %d price rows + summary for '%s'",
            len(hist),
            ticker,
        )
        return text_records, hist

    # ── Generic JSON API ──────────────────────────────────────────────────────

    def load_json_api(
        self,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Fetch JSON from a URL and return the raw text representation.
        """
        import requests
        import json

        resp = requests.get(
            url, headers=headers, params=params, timeout=self.timeout
        )
        resp.raise_for_status()
        data = resp.json()

        text = json.dumps(data, indent=2, default=str)
        meta = build_metadata(source=url, doc_type="api")
        validate_metadata(meta)

        logger.info("APILoader: fetched %d chars from '%s'", len(text), url)
        return [(text, meta)]
