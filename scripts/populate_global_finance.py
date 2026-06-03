"""
scripts/populate_global_finance.py
Populates PostgreSQL with global finance data from yfinance:
  - stock_prices: 1 year of daily OHLCV for ~30 major global tickers
  - financial_metrics: Key fundamentals (P/E, EPS, revenue, etc.)
  - macro_indicators: Global macro data (VIX, Gold, Oil, Rates, FX)

Usage:
    python scripts/populate_global_finance.py

Run daily via cron/scheduler to keep data fresh:
    python scripts/populate_global_finance.py --update
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

import yfinance as yf
import pandas as pd
from storage.relational_db import RelationalDB

# ── Global Tickers ────────────────────────────────────────────────────────────

STOCK_TICKERS = [
    # US Tech
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
    # US Finance
    "JPM", "GS", "V", "MA", "BAC",
    # US Healthcare/Consumer
    "JNJ", "UNH", "PG", "KO",
    # Europe
    "ASML", "SAP", "SHEL", "NVS",
    # China/Asia
    "TSM", "BABA",
    # Energy/Commodities
    "XOM", "CVX",
    # ETFs (broad market exposure)
    "SPY", "QQQ", "EEM", "VTI",
]

# Macro indicators tracked via ETFs/indices
MACRO_TICKERS = {
    "^VIX": ("VIX (Volatility Index)", "index"),
    "^GSPC": ("S&P 500", "index"),
    "^IXIC": ("NASDAQ Composite", "index"),
    "^DJI": ("Dow Jones Industrial", "index"),
    "GC=F": ("Gold (USD/oz)", "commodity"),
    "CL=F": ("Crude Oil WTI (USD/bbl)", "commodity"),
    "BZ=F": ("Brent Oil (USD/bbl)", "commodity"),
    "EURUSD=X": ("EUR/USD", "forex"),
    "GBPUSD=X": ("GBP/USD", "forex"),
    "JPY=X": ("USD/JPY", "forex"),
    "^TNX": ("US 10Y Treasury Yield", "rate"),
    "^FVX": ("US 5Y Treasury Yield", "rate"),
}


def progress(current, total, label=""):
    pct = current / total if total > 0 else 0
    filled = int(30 * pct)
    bar = "█" * filled + "░" * (30 - filled)
    print(f"\r  [{bar}] {current}/{total} {label}    ", end="", flush=True)


def populate_stock_prices(db, period="1y"):
    """Download OHLCV data for all tickers and insert into stock_prices."""
    print("\n── Stock Prices ────────────────────────────────────────────────")
    
    # Clear old data
    db.execute_sql("DELETE FROM stock_prices")
    
    total = len(STOCK_TICKERS)
    inserted = 0

    for i, ticker in enumerate(STOCK_TICKERS, 1):
        progress(i, total, ticker)
        try:
            data = yf.download(ticker, period=period, progress=False, auto_adjust=True)
            if data.empty:
                continue

            # Flatten multi-level columns (yfinance returns ('Close', 'AAPL'))
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            for date, row in data.iterrows():
                db.execute_sql(
                    """INSERT INTO stock_prices (ticker, date, open, high, low, close, volume, created_at)
                       VALUES (:ticker, :date, :open, :high, :low, :close, :volume, :created_at)
                       ON CONFLICT ON CONSTRAINT uq_price_ticker_date DO UPDATE
                       SET open = :open, high = :high, low = :low, close = :close, volume = :volume""",
                    {
                        "ticker": ticker,
                        "date": date.strftime("%Y-%m-%d"),
                        "open": float(row["Open"]) if pd.notna(row["Open"]) else None,
                        "high": float(row["High"]) if pd.notna(row["High"]) else None,
                        "low": float(row["Low"]) if pd.notna(row["Low"]) else None,
                        "close": float(row["Close"]) if pd.notna(row["Close"]) else None,
                        "volume": int(row["Volume"]) if pd.notna(row["Volume"]) else None,
                        "created_at": datetime.now(timezone.utc),
                    }
                )
            inserted += len(data)
        except Exception as e:
            print(f"\n  ⚠ {ticker}: {str(e)[:50]}")

    print(f"\n  ✓ Inserted {inserted:,} price rows for {total} tickers")


def populate_financial_metrics(db):
    """Download key fundamentals for all tickers."""
    print("\n── Financial Metrics ───────────────────────────────────────────")

    # Clear old data
    db.execute_sql("DELETE FROM financial_metrics")

    metrics_to_fetch = [
        "marketCap", "trailingPE", "forwardPE", "trailingEps", "forwardEps",
        "revenue", "netIncome", "ebitda", "operatingIncome", "grossProfits",
        "freeCashflow", "operatingMargins", "profitMargins",
        "debtToEquity", "returnOnEquity", "returnOnAssets",
        "beta", "dividendYield", "fiftyTwoWeekHigh", "fiftyTwoWeekLow",
        "averageVolume", "bookValue",
    ]

    total = len(STOCK_TICKERS)
    inserted = 0

    for i, ticker in enumerate(STOCK_TICKERS, 1):
        progress(i, total, ticker)
        try:
            info = yf.Ticker(ticker).info
            for metric in metrics_to_fetch:
                value = info.get(metric)
                if value is not None:
                    # Convert metric name to snake_case
                    metric_name = ''.join(
                        ['_' + c.lower() if c.isupper() else c for c in metric]
                    ).lstrip('_')

                    db.execute_sql(
                        """INSERT INTO financial_metrics (ticker, metric_name, value, period, created_at)
                           VALUES (:ticker, :metric_name, :value, :period, :created_at)
                           ON CONFLICT ON CONSTRAINT uq_metric_ticker_name_period
                           DO UPDATE SET value = :value, created_at = :created_at""",
                        {
                            "ticker": ticker,
                            "metric_name": metric_name,
                            "value": float(value) if isinstance(value, (int, float)) else None,
                            "period": "latest",
                            "created_at": datetime.now(timezone.utc),
                        }
                    )
                    inserted += 1
        except Exception as e:
            print(f"\n  ⚠ {ticker}: {str(e)[:50]}")

    print(f"\n  ✓ Inserted {inserted:,} metrics for {total} tickers")


def populate_macro_indicators(db):
    """Download macro indicators (indices, commodities, FX, rates)."""
    print("\n── Macro Indicators ───────────────────────────────────────────")

    # Clear old data
    db.execute_sql("DELETE FROM macro_indicators")

    total = len(MACRO_TICKERS)
    inserted = 0

    for i, (ticker, (name, category)) in enumerate(MACRO_TICKERS.items(), 1):
        progress(i, total, name[:20])
        try:
            data = yf.download(ticker, period="1y", progress=False, auto_adjust=True)
            if data.empty:
                continue

            # Flatten multi-level columns
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            for date, row in data.iterrows():
                close_val = row["Close"]
                if pd.notna(close_val):
                    db.execute_sql(
                        """INSERT INTO macro_indicators (country, indicator_name, value, unit, date, source, created_at)
                           VALUES (:country, :indicator_name, :value, :unit, :date, :source, :created_at)""",
                        {
                            "country": "Global",
                            "indicator_name": name,
                            "value": float(close_val),
                            "unit": category,
                            "date": date.strftime("%Y-%m-%d"),
                            "source": "yfinance",
                            "created_at": datetime.now(timezone.utc),
                        }
                    )
                    inserted += 1
        except Exception as e:
            print(f"\n  ⚠ {name}: {str(e)[:50]}")

    print(f"\n  ✓ Inserted {inserted:,} macro data points")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--update", action="store_true", help="Only fetch last 5 days (daily update mode)")
    args = parser.parse_args()

    period = "5d" if args.update else "1y"

    print()
    print("═" * 70)
    print("  📊 Global Finance Data Population")
    print(f"  Period: {period} | Tickers: {len(STOCK_TICKERS)} stocks + {len(MACRO_TICKERS)} macro")
    print("═" * 70)

    db = RelationalDB()

    populate_stock_prices(db, period=period)
    populate_financial_metrics(db)
    populate_macro_indicators(db)

    # Final count
    prices = db.execute_sql("SELECT COUNT(*) as cnt FROM stock_prices")[0]["cnt"]
    metrics = db.execute_sql("SELECT COUNT(*) as cnt FROM financial_metrics")[0]["cnt"]
    macro = db.execute_sql("SELECT COUNT(*) as cnt FROM macro_indicators")[0]["cnt"]

    print()
    print("═" * 70)
    print(f"  ✅ DONE")
    print(f"     stock_prices:      {prices:,} rows")
    print(f"     financial_metrics: {metrics:,} rows")
    print(f"     macro_indicators:  {macro:,} rows")
    print("═" * 70)
    print()
    print("  💡 To update daily, run: python scripts/populate_global_finance.py --update")
    print()


if __name__ == "__main__":
    main()
