"""
tests/test_api.py
In-process API-level tests using FastAPI's TestClient (Starlette/requests).
No separate server needed.

Run with:
    pytest tests/test_api.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


# ── Health endpoints ──────────────────────────────────────────────────────────

def test_root_returns_ok():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "AI Financial Advisor API"


def test_health_returns_healthy():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


# ── Query endpoint ────────────────────────────────────────────────────────────

def test_query_empty_question_rejected():
    """Empty question should return 400 (our guard) or 422 (Pydantic)."""
    response = client.post("/query/", json={"question": "   "})
    assert response.status_code in (400, 422)


def test_query_returns_answer():
    """
    Happy-path: mock the entire agents.financial_agent module so that
    langchain_ollama (not installed in the test environment) is never imported.
    """
    import sys
    from types import ModuleType

    mock_result = {"answer": "Apple revenue was $383B.", "context": "Source: test.pdf"}

    # Build a lightweight fake module with just the `ask` function
    fake_fa = ModuleType("agents.financial_agent")
    fake_fa.ask = MagicMock(return_value=mock_result)  # type: ignore[attr-defined]

    with patch.dict(sys.modules, {"agents.financial_agent": fake_fa}):
        response = client.post("/query/", json={"question": "What is Apple revenue?"})

    assert response.status_code == 200
    data = response.json()
    assert data["question"] == "What is Apple revenue?"
    assert "Apple" in data["answer"]
    assert "context" in data


# ── Stocks endpoint ───────────────────────────────────────────────────────────

def test_stocks_snapshot_returns_json():
    """Happy-path: mock SQLSearch so no PostgreSQL is needed."""
    fake_latest = [{"ticker": "AAPL", "date": "2024-01-10", "close": 185.4}]
    fake_history = [
        {"ticker": "AAPL", "date": "2024-01-09", "close": 182.0},
        {"ticker": "AAPL", "date": "2024-01-08", "close": 180.5},
    ]
    mock_sql = MagicMock()
    mock_sql.get_latest_price.return_value = fake_latest
    mock_sql.get_price_history.return_value = fake_history

    with patch("api.routes.stocks._get_sql", return_value=mock_sql):
        response = client.get("/stocks/AAPL")
    assert response.status_code == 200
    data = response.json()
    assert data["ticker"] == "AAPL"
    assert len(data["latest"]) >= 1
    assert len(data["history_5d"]) == 2


def test_stocks_not_found():
    """Unknown ticker with no data should return 404."""
    mock_sql = MagicMock()
    mock_sql.get_latest_price.return_value = None

    with patch("api.routes.stocks._get_sql", return_value=mock_sql):
        response = client.get("/stocks/XXXXXX")
    assert response.status_code == 404


def test_stocks_history_defaults_to_30_days():
    """History endpoint defaults to 30 days."""
    fake_history = [{"ticker": "MSFT", "date": "2024-01-01", "close": 370.0}] * 20
    mock_sql = MagicMock()
    mock_sql.get_price_history.return_value = fake_history

    with patch("api.routes.stocks._get_sql", return_value=mock_sql):
        response = client.get("/stocks/MSFT/history")
    assert response.status_code == 200
    data = response.json()
    assert data["ticker"] == "MSFT"
    assert data["days"] == 30
