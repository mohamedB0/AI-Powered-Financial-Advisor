"""
api/routes/stocks.py
GET /stocks/{ticker}         — latest closing price + 5-day snapshot.
GET /stocks/{ticker}/history — configurable day-range price history.

Both endpoints use the existing SQLSearch helpers (same source used by
agents/tools.py) and run in a thread-pool (plain `def`) to avoid
blocking the event loop.
"""

from typing import Annotated

from fastapi import APIRouter, HTTPException, Path, Query
from pydantic import BaseModel

router = APIRouter(prefix="/stocks", tags=["Stocks"])


# ── Response models ────────────────────────────────────────────────────────────

class PriceSnapshot(BaseModel):
    ticker: str
    latest: list[dict]
    history_5d: list[dict]


class PriceHistory(BaseModel):
    ticker: str
    days: int
    history: list[dict]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_sql():
    from retrieval.sql_search import SQLSearch
    return SQLSearch()


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/{ticker}")
def get_stock_snapshot(
    ticker: Annotated[str, Path(description="Stock ticker symbol, e.g. AAPL")],
) -> PriceSnapshot:
    """
    Return the latest closing price and a 5-day price snapshot for *ticker*.
    Executes in a thread-pool (plain `def`) — SQLSearch is blocking.
    """
    sql = _get_sql()
    latest = sql.get_latest_price(ticker.upper())
    if not latest:
        raise HTTPException(
            status_code=404, detail=f"No price data found for ticker '{ticker}'."
        )
    history = sql.get_price_history(ticker.upper(), days=5)
    return PriceSnapshot(
        ticker=ticker.upper(),
        latest=latest if isinstance(latest, list) else [latest],
        history_5d=history or [],
    )


@router.get("/{ticker}/history")
def get_stock_history(
    ticker: Annotated[str, Path(description="Stock ticker symbol, e.g. AAPL")],
    days: Annotated[int, Query(ge=1, le=365, description="Number of trading days")] = 30,
) -> PriceHistory:
    """
    Return *days* of price history for *ticker*.
    Executes in a thread-pool (plain `def`) — SQLSearch is blocking.
    """
    sql = _get_sql()
    history = sql.get_price_history(ticker.upper(), days=days)
    if not history:
        raise HTTPException(
            status_code=404,
            detail=f"No history data found for ticker '{ticker}' over {days} days.",
        )
    return PriceHistory(
        ticker=ticker.upper(),
        days=days,
        history=history,
    )
