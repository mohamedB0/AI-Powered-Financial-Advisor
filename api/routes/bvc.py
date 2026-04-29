"""
api/routes/bvc.py
POST /bvc/refresh — Trigger BVC data fetch via Celery.
GET /bvc/status/{task_id} — Check BVC fetch task status.
"""

from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

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
async def refresh_bvc_data():
    """
    Trigger a BVC data refresh (stock prices + indices).
    Runs asynchronously via Celery worker.
    """
    try:
        from tasks.bvc_fetch_tasks import fetch_bvc_daily

        task = fetch_bvc_daily.apply_async()
        return BVCRefreshResponse(
            task_id=task.id,
            status="PENDING",
            message="BVC data refresh task submitted. Check status_url for progress.",
            status_url=f"/bvc/status/{task.id}",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status/{task_id}", response_model=BVCStatusResponse)
async def get_bvc_task_status(task_id: str) -> BVCStatusResponse:
    """
    Get the status of a BVC refresh task.
    """
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
    """
    Get latest BVC stock prices from PostgreSQL.
    """
    try:
        from storage.relational_db import RelationalDB
        from retrieval.sql_search import SQLSearch

        sql = SQLSearch()
        prices = sql.get_bvc_latest_prices()

        if not prices:
            return BVCDataResponse(tickers=[], latest_prices=[], indices=[])

        tickers = list(set(p["ticker"] for p in prices))
        return BVCDataResponse(
            tickers=tickers,
            latest_prices=prices,
            indices=[],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
