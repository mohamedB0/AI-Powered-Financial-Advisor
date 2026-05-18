"""
api/routes/bvc.py
POST /bvc/refresh     — Trigger BVC data fetch via Celery.
GET  /bvc/status/{id} — Check BVC fetch task status.
GET  /bvc/prices      — Latest BVC prices from PostgreSQL.
GET  /bvc/live        — Live stock prices proxied from Casablanca Bourse.
GET  /bvc/indices     — Live index values (MASI, MADEX, CFG25…).
"""

from typing import Optional

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from api.auth import get_current_user

router = APIRouter(prefix="/bvc", tags=["BVC"])


class BVCRefreshResponse(BaseModel):
    task_id: str
    status: str
    message: str
    status_url: str


class BVCStatusResponse(BaseModel):
    task_id: str
    status: str
    ready: bool
    result: Optional[dict] = None


class BVCDataResponse(BaseModel):
    tickers: list[str]
    latest_prices: list[dict]
    indices: list[dict]


@router.post("/refresh", response_model=BVCRefreshResponse)
async def refresh_bvc_data(user: dict = Depends(get_current_user)):
    """Trigger a BVC data refresh (stock prices + indices) via Celery."""
    try:
        from tasks.bvc_fetch_tasks import fetch_bvc_daily
        task = fetch_bvc_daily.apply_async()
        return BVCRefreshResponse(
            task_id=task.id,
            status="PENDING",
            message="BVC data refresh task submitted.",
            status_url=f"/bvc/status/{task.id}",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status/{task_id}", response_model=BVCStatusResponse)
async def get_bvc_task_status(task_id: str):
    """Get the status of a BVC refresh task."""
    from celery.result import AsyncResult
    result = AsyncResult(task_id)
    return BVCStatusResponse(
        task_id=task_id,
        status=result.status,
        ready=result.ready(),
        result=result.result if result.ready() else None,
    )


@router.get("/prices", response_model=BVCDataResponse)
async def get_bvc_prices():
    """Get latest BVC stock prices from PostgreSQL."""
    try:
        from retrieval.sql_search import SQLSearch
        sql = SQLSearch()
        prices = sql.get_bvc_latest_prices()
        if not prices:
            return BVCDataResponse(tickers=[], latest_prices=[], indices=[])
        tickers = list(set(p["ticker"] for p in prices))
        return BVCDataResponse(tickers=tickers, latest_prices=prices, indices=[])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/live")
async def get_bvc_live():
    """
    Proxy live stock data directly from Casablanca Bourse API.
    Returns all listed instruments with last price, variation %, volume, high, low.
    Intended for polling every 30s from the UI market panel.
    Runs in a thread pool to avoid blocking uvicorn's async event loop.
    """
    import asyncio
    try:
        from retrieval.bvc_scraper import get_market_live, get_market_summary
        stocks  = await asyncio.to_thread(get_market_live)
        summary = await asyncio.to_thread(get_market_summary)
        return {"stocks": stocks, "summary": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/indices")
async def get_bvc_indices():
    """Proxy live index data (MASI, MADEX, CFG25, etc.) from Casablanca Bourse."""
    import asyncio
    try:
        from retrieval.bvc_scraper import get_indices
        indices = await asyncio.to_thread(get_indices)
        return {"indices": indices}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
