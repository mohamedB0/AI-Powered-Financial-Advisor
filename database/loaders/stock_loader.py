"""
database/loaders/stock_loader.py
Loads stock price data from CSV files into PostgreSQL.

Usage:
    python database/loaders/stock_loader.py --file data/csv/stock_prices.csv
    python database/loaders/stock_loader.py --dir data/csv/
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import logging
import argparse
from datetime import datetime

from storage.relational_db import RelationalDB

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def load_stock_csv(file_path: str) -> int:
    """
    Load stock prices from a CSV file into PostgreSQL.
    
    Expected CSV columns: ticker, date, open, high, low, close, volume
    
    Returns:
        Number of records inserted.
    """
    import pandas as pd
    
    path = Path(file_path)
    if not path.exists():
        logger.error("File not found: %s", file_path)
        return 0
    
    logger.info("Loading stock data from: %s", file_path)
    
    df = pd.read_csv(file_path)
    
    required_cols = {"ticker", "date", "close"}
    missing = required_cols - set(df.columns.str.lower())
    if missing:
        logger.error("Missing required columns: %s", missing)
        return 0
    
    df.columns = df.columns.str.lower()
    
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.date
    
    records = df.to_dict(orient="records")
    
    db = RelationalDB()
    db.setup_database()
    
    count = db.insert_prices(records)
    logger.info("Inserted %d stock price records", count)
    
    return count


def load_from_directory(directory: str) -> int:
    """
    Load all CSV files from a directory.
    """
    dir_path = Path(directory)
    if not dir_path.exists():
        logger.error("Directory not found: %s", directory)
        return 0
    
    total = 0
    for csv_file in dir_path.glob("*.csv"):
        try:
            count = load_stock_csv(str(csv_file))
            total += count
        except Exception as e:
            logger.error("Failed to load %s: %s", csv_file, e)
    
    logger.info("Total records loaded: %d", total)
    return total


def load_financial_metrics_csv(file_path: str) -> int:
    """
    Load financial metrics from a CSV file.
    
    Expected CSV columns: ticker, company, metric_name, value, period, report_date
    """
    import pandas as pd
    
    path = Path(file_path)
    if not path.exists():
        logger.error("File not found: %s", file_path)
        return 0
    
    logger.info("Loading financial metrics from: %s", file_path)
    
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.lower()
    
    if "report_date" in df.columns:
        df["report_date"] = pd.to_datetime(df["report_date"], errors="coerce").dt.date
    
    records = df.to_dict(orient="records")
    
    db = RelationalDB()
    db.setup_database()
    
    count = db.insert_metrics(records)
    logger.info("Inserted %d financial metric records", count)
    
    return count


def load_macro_csv(file_path: str) -> int:
    """
    Load macroeconomic indicators from a CSV file.
    
    Expected CSV columns: country, indicator_name, value, unit, date, source
    """
    import pandas as pd
    
    path = Path(file_path)
    if not path.exists():
        logger.error("File not found: %s", file_path)
        return 0
    
    logger.info("Loading macro indicators from: %s", file_path)
    
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.lower()
    
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.date
    
    records = df.to_dict(orient="records")
    
    db = RelationalDB()
    db.setup_database()
    
    count = db.insert_macro(records)
    logger.info("Inserted %d macro indicator records", count)
    
    return count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load financial data into PostgreSQL")
    parser.add_argument("--file", type=str, help="Path to CSV file")
    parser.add_argument("--dir", type=str, help="Path to directory containing CSV files")
    parser.add_argument("--type", type=str, choices=["stocks", "metrics", "macro"], default="stocks",
                        help="Type of data to load")
    
    args = parser.parse_args()
    
    db = RelationalDB()
    db.setup_database()
    
    if args.file:
        if args.type == "stocks":
            load_stock_csv(args.file)
        elif args.type == "metrics":
            load_financial_metrics_csv(args.file)
        else:
            load_macro_csv(args.file)
    elif args.dir:
        load_from_directory(args.dir)
    else:
        parser.print_help()
