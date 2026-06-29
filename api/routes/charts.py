"""
api/routes/charts.py
Chart data generation for inline visualization in chat responses.

Detects if a user query warrants a chart, fetches the relevant SQL data,
and returns a Plotly-compatible JSON spec that the frontend renders.
"""

import re
from typing import Optional
from retrieval.sql_search import SQLSearch


# Keywords that suggest the user wants a visualization
_CHART_KEYWORDS = [
    # English
    "chart", "graph", "plot", "show me", "visualize", "trend", "evolution",
    "history", "over time", "last days", "last month",
    # French
    "graphique", "courbe", "montrer", "evolution", "historique", "tendance",
    "afficher", "visualiser", "derniers jours",
    # Arabic
    "رسم", "بياني", "اظهر", "تطور", "اتجاه",
]

# Tickers we can chart (from the DB)
_KNOWN_TICKERS = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "NVDA", "GS",
    "BVC.ATW", "BVC.BCP", "BVC.IAM", "BVC.MNG", "BVC.CSR", "BVC.LHM",
    "ATW", "BCP", "IAM", "MNG", "CSR",
]

_MACRO_INDICATORS = ["VIX", "SPY", "GLD", "USO", "DX-Y.NYB"]


def detect_chart_intent(question: str) -> Optional[dict]:
    """
    Detect if the question warrants a chart and determine what to plot.

    Returns:
        None if no chart needed, or a dict with:
        {
            "type": "line" | "bar",
            "chart_for": "stock_price" | "top_movers" | "macro",
            "ticker": "AAPL" (if applicable),
            "title": "Chart title"
        }
    """
    q_lower = question.lower()

    # Check if the question has chart intent
    has_chart_intent = any(kw in q_lower for kw in _CHART_KEYWORDS)
    if not has_chart_intent:
        return None

    # Detect what to chart
    # 1. Specific ticker price history
    for ticker in _KNOWN_TICKERS:
        if ticker.lower() in q_lower or ticker.replace("BVC.", "").lower() in q_lower:
            return {
                "type": "line",
                "chart_for": "stock_price",
                "ticker": ticker,
                "title": f"Price History — {ticker}",
            }

    # 2. Top movers / gainers / losers
    if any(kw in q_lower for kw in ["mover", "gainer", "loser", "perform", "hausse", "baisse"]):
        return {
            "type": "bar",
            "chart_for": "top_movers",
            "ticker": None,
            "title": "Top Market Movers (30 days)",
        }

    # 3. Macro indicators
    for indicator in _MACRO_INDICATORS:
        if indicator.lower() in q_lower:
            return {
                "type": "line",
                "chart_for": "macro",
                "ticker": indicator,
                "title": f"Macro Indicator — {indicator}",
            }

    # 4. Generic "show chart" without specific target
    if any(kw in q_lower for kw in ["stock", "action", "cours", "prix", "سهم"]):
        return {
            "type": "bar",
            "chart_for": "top_movers",
            "ticker": None,
            "title": "Top Market Movers (30 days)",
        }

    return None


def generate_chart_data(chart_intent: dict) -> Optional[dict]:
    """
    Fetch SQL data and produce a Plotly-compatible chart spec.

    Returns a dict with:
    {
        "plotly": { ...Plotly JSON spec... },
        "title": "Chart Title"
    }
    """
    sql = SQLSearch()
    chart_type = chart_intent["chart_for"]

    try:
        if chart_type == "stock_price":
            ticker = chart_intent["ticker"]
            rows = sql.get_price_history(ticker, days=30)
            if not rows:
                return None

            dates = [str(r.get("date", "")) for r in reversed(rows)]
            closes = [float(r.get("close", 0)) for r in reversed(rows)]

            return {
                "title": chart_intent["title"],
                "plotly": {
                    "data": [{
                        "x": dates,
                        "y": closes,
                        "type": "scatter",
                        "mode": "lines+markers",
                        "name": ticker,
                        "line": {"color": "#10b981", "width": 2},
                        "marker": {"size": 4},
                    }],
                    "layout": {
                        "title": chart_intent["title"],
                        "xaxis": {"title": "Date"},
                        "yaxis": {"title": "Close Price"},
                        "template": "plotly_white",
                        "height": 300,
                        "margin": {"t": 40, "b": 40, "l": 50, "r": 20},
                    },
                },
            }

        elif chart_type == "top_movers":
            rows = sql.get_top_movers(limit=8)
            if not rows:
                return None

            tickers = [r.get("ticker", "") for r in rows]
            changes = [float(r.get("pct_change", 0)) for r in rows]
            colors = ["#10b981" if c >= 0 else "#ef4444" for c in changes]

            return {
                "title": chart_intent["title"],
                "plotly": {
                    "data": [{
                        "x": tickers,
                        "y": changes,
                        "type": "bar",
                        "marker": {"color": colors},
                    }],
                    "layout": {
                        "title": chart_intent["title"],
                        "xaxis": {"title": "Ticker"},
                        "yaxis": {"title": "% Change (30d)"},
                        "template": "plotly_white",
                        "height": 300,
                        "margin": {"t": 40, "b": 40, "l": 50, "r": 20},
                    },
                },
            }

        elif chart_type == "macro":
            # Query macro_indicators for the specific indicator
            ticker = chart_intent["ticker"]
            rows = sql.get_price_history(ticker, days=60)
            if not rows:
                return None

            dates = [str(r.get("date", "")) for r in reversed(rows)]
            values = [float(r.get("close", 0)) for r in reversed(rows)]

            return {
                "title": chart_intent["title"],
                "plotly": {
                    "data": [{
                        "x": dates,
                        "y": values,
                        "type": "scatter",
                        "mode": "lines",
                        "name": ticker,
                        "line": {"color": "#6366f1", "width": 2},
                    }],
                    "layout": {
                        "title": chart_intent["title"],
                        "xaxis": {"title": "Date"},
                        "yaxis": {"title": "Value"},
                        "template": "plotly_white",
                        "height": 300,
                        "margin": {"t": 40, "b": 40, "l": 50, "r": 20},
                    },
                },
            }

    except Exception as e:
        print(f"[Chart] Error generating chart data: {e}")
        return None

    return None
