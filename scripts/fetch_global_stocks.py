"""
scripts/fetch_global_stocks.py
Fetch and store global stock data (prices + financial metrics) from Yahoo Finance.

Stores:
  - Daily OHLCV prices → stock_prices table (1 year of history)
  - Key financial metrics → financial_metrics table (PE, EPS, revenue, etc.)

Usage:
    python scripts/fetch_global_stocks.py                   # All default tickers
    python scripts/fetch_global_stocks.py --tickers AAPL MSFT
    python scripts/fetch_global_stocks.py --period 2y       # 2 years of history
"""

import sys
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Default Tickers: Global Blue-Chip Companies ──────────────────────────────

DEFAULT_TICKERS = [
    # US Tech Giants
    "AAPL",     # Apple
    "MSFT",     # Microsoft
    "GOOGL",    # Alphabet (Google)
    "AMZN",     # Amazon
    "META",     # Meta (Facebook)
    "NVDA",     # NVIDIA
    "TSLA",     # Tesla
    # US Finance
    "JPM",      # JPMorgan Chase
    "V",        # Visa
    "MA",       # Mastercard
    # US Healthcare
    "JNJ",      # Johnson & Johnson
    "UNH",      # UnitedHealth
    # US Consumer
    "WMT",      # Walmart
    "KO",       # Coca-Cola
    # European
    "ASML",     # ASML (Netherlands)
    "SAP",      # SAP (Germany)
    "MC.PA",    # LVMH (France)
    "NESN.SW",  # Nestlé (Switzerland)
    # Asian
    "TSM",      # TSMC (Taiwan)
    "BABA",     # Alibaba (China)
]


def fetch_prices(ticker: str, period: str = "1y") -> dict:
    """Fetch OHLCV prices and store in PostgreSQL."""
    import yfinance as yf
    from storage.relational_db import RelationalDB

    db = RelationalDB()
    stock = yf.Ticker(ticker)
    hist = stock.history(period=period)

    if hist.empty:
        return {"ticker": ticker, "prices": 0, "status": "no_data"}

    hist_reset = hist.reset_index()
    # Handle multi-index columns
    hist_reset.columns = [c[0] if isinstance(c, tuple) else c for c in hist_reset.columns]

    records = []
    for _, row in hist_reset.iterrows():
        records.append({
            "ticker": ticker,
            "date": row["Date"].date() if hasattr(row["Date"], "date") else row["Date"],
            "open": round(float(row.get("Open", 0)), 4),
            "high": round(float(row.get("High", 0)), 4),
            "low": round(float(row.get("Low", 0)), 4),
            "close": round(float(row.get("Close", 0)), 4),
            "volume": int(row.get("Volume", 0)),
        })

    count = db.insert_prices(records)
    return {"ticker": ticker, "prices": count, "status": "success"}


def fetch_metrics(ticker: str) -> dict:
    """Fetch key financial metrics and store in PostgreSQL."""
    import yfinance as yf
    from storage.relational_db import RelationalDB

    db = RelationalDB()
    stock = yf.Ticker(ticker)
    info = stock.info or {}

    if not info:
        return {"ticker": ticker, "metrics": 0, "status": "no_info"}

    company_name = info.get("longName", info.get("shortName", ticker))
    current_year = str(datetime.now().year)

    # Key metrics to extract
    metrics_map = {
        "market_cap": info.get("marketCap"),
        "pe_ratio": info.get("trailingPE"),
        "forward_pe": info.get("forwardPE"),
        "eps": info.get("trailingEps"),
        "forward_eps": info.get("forwardEps"),
        "revenue": info.get("totalRevenue"),
        "gross_profit": info.get("grossProfits"),
        "ebitda": info.get("ebitda"),
        "net_income": info.get("netIncomeToCommon"),
        "profit_margin": info.get("profitMargins"),
        "operating_margin": info.get("operatingMargins"),
        "roe": info.get("returnOnEquity"),
        "roa": info.get("returnOnAssets"),
        "debt_to_equity": info.get("debtToEquity"),
        "current_ratio": info.get("currentRatio"),
        "dividend_yield": info.get("dividendYield"),
        "beta": info.get("beta"),
        "52w_high": info.get("fiftyTwoWeekHigh"),
        "52w_low": info.get("fiftyTwoWeekLow"),
        "avg_volume": info.get("averageVolume"),
        "book_value": info.get("bookValue"),
        "price_to_book": info.get("priceToBook"),
        "free_cash_flow": info.get("freeCashflow"),
    }

    records = []
    for metric_name, value in metrics_map.items():
        if value is not None:
            records.append({
                "ticker": ticker,
                "company": company_name,
                "metric_name": metric_name,
                "value": float(value) if value else 0,
                "period": current_year,
                "report_date": datetime.now().date(),
            })

    count = db.insert_metrics(records)

    # Also fetch annual financials if available
    try:
        financials = stock.financials
        if financials is not None and not financials.empty:
            for col in financials.columns[:4]:  # Last 4 years
                year = str(col.year) if hasattr(col, "year") else str(col)[:4]
                annual_metrics = {
                    "revenue": financials.loc["Total Revenue", col] if "Total Revenue" in financials.index else None,
                    "gross_profit": financials.loc["Gross Profit", col] if "Gross Profit" in financials.index else None,
                    "operating_income": financials.loc["Operating Income", col] if "Operating Income" in financials.index else None,
                    "net_income": financials.loc["Net Income", col] if "Net Income" in financials.index else None,
                    "ebitda": financials.loc["EBITDA", col] if "EBITDA" in financials.index else None,
                }
                for m_name, m_val in annual_metrics.items():
                    if m_val is not None:
                        records.append({
                            "ticker": ticker,
                            "company": company_name,
                            "metric_name": m_name,
                            "value": float(m_val),
                            "period": year,
                            "report_date": datetime.now().date(),
                        })

            # Insert annual metrics
            if records:
                db.insert_metrics(records)
                count += len(records)
    except Exception as e:
        logger.warning(f"  Could not fetch annual financials for {ticker}: {e}")

    return {"ticker": ticker, "metrics": count, "status": "success", "company": company_name}


def main():
    parser = argparse.ArgumentParser(description="Fetch global stock data from Yahoo Finance")
    parser.add_argument("--tickers", nargs="+", default=None, help="Tickers to fetch (default: all)")
    parser.add_argument("--period", default="1y", help="Price history period (1d,5d,1mo,3mo,6mo,1y,2y,5y)")
    parser.add_argument("--skip-prices", action="store_true", help="Skip price fetching")
    parser.add_argument("--skip-metrics", action="store_true", help="Skip metrics fetching")
    args = parser.parse_args()

    tickers = args.tickers or DEFAULT_TICKERS

    # Ensure DB tables exist
    from storage.relational_db import RelationalDB
    db = RelationalDB()
    db.setup_database()

    print(f"\n{'='*60}")
    print(f"  GLOBAL STOCK DATA FETCH (Yahoo Finance)")
    print(f"{'='*60}")
    print(f"  Tickers: {len(tickers)}")
    print(f"  Period: {args.period}")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*60}\n")

    total_prices = 0
    total_metrics = 0
    errors = []

    for i, ticker in enumerate(tickers, 1):
        print(f"  [{i:2d}/{len(tickers)}] {ticker:8s}", end=" ")

        # Fetch prices
        if not args.skip_prices:
            try:
                price_result = fetch_prices(ticker, args.period)
                total_prices += price_result.get("prices", 0)
                print(f"| {price_result['prices']:4d} prices", end=" ")
            except Exception as e:
                print(f"| prices FAIL", end=" ")
                errors.append(f"{ticker} prices: {str(e)[:50]}")

        # Fetch metrics
        if not args.skip_metrics:
            try:
                metric_result = fetch_metrics(ticker)
                total_metrics += metric_result.get("metrics", 0)
                company = metric_result.get("company", "")[:25]
                print(f"| {metric_result['metrics']:3d} metrics | {company}")
            except Exception as e:
                print(f"| metrics FAIL")
                errors.append(f"{ticker} metrics: {str(e)[:50]}")
        else:
            print()

        # Polite delay (Yahoo Finance rate limit)
        time.sleep(2)

    # Summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"  Total price records stored:  {total_prices}")
    print(f"  Total metric records stored: {total_metrics}")
    print(f"  Errors: {len(errors)}")
    if errors:
        for e in errors[:5]:
            print(f"    - {e}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
