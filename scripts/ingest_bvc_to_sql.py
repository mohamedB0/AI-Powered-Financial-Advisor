"""
scripts/ingest_bvc_to_sql.py
One-shot ingestion of live BVC market data into PostgreSQL `stock_prices`.

Uses the reliable JSON API (retrieval.bvc_scraper.get_market_live) instead of
the HTML cotation page (which is JS-rendered and often empty), then upserts
OHLCV rows so the SQL route can answer Moroccan stock questions.
"""

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from retrieval.bvc_scraper import get_market_live
from storage.relational_db import RelationalDB


def _f(val):
    """Parse BVC string numbers to float, tolerant of None / blanks."""
    if val is None or str(val).strip() in ("", "-", "nan"):
        return None
    try:
        return float(str(val).replace(",", ".").replace(" ", "").replace("\xa0", ""))
    except (ValueError, AttributeError):
        return None


def main():
    stocks = get_market_live()
    print(f"Fetched {len(stocks)} live BVC instruments from the JSON API.")
    if not stocks:
        print("No data returned — aborting.")
        return

    today = datetime.now().date()
    records = []
    for s in stocks:
        ticker = (s.get("ticker") or "").strip().upper()
        if not ticker:
            continue
        close = _f(s.get("last_price"))
        if close is None:
            continue  # skip untraded rows with no price
        # Store with the BVC. prefix so it matches the router's ticker
        # extraction (e.g. "BVC.ATW") and the get_bvc_latest_prices convention.
        records.append({
            "ticker": f"BVC.{ticker}",
            "date": today,
            "open": _f(s.get("open")),
            "high": _f(s.get("high")),
            "low": _f(s.get("low")),
            "close": close,
            "volume": int(_f(s.get("volume")) or 0),
        })

    print(f"Prepared {len(records)} priced records (with a close value).")

    db = RelationalDB()
    db.setup_database()
    inserted = db.insert_prices(records)
    print(f"Upserted {inserted} rows into stock_prices (skips existing ticker+date).")

    # Show a few well-known tickers so we can confirm the demo will work
    sample = db.execute_sql(
        "SELECT ticker, close, volume, date FROM stock_prices "
        "WHERE ticker IN ('BVC.ATW','BVC.BCP','BVC.IAM','BVC.CIH','BVC.BOA','BVC.MNG','BVC.LHM','BVC.CSR') "
        "ORDER BY ticker"
    )
    print("\nSample of key tickers now in DB:")
    for r in sample:
        print(" ", r)


if __name__ == "__main__":
    main()
