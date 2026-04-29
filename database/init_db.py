"""
database/init_db.py
Initializes the PostgreSQL database with all ORM tables.

Usage:
    python database/init_db.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
from dotenv import load_dotenv
load_dotenv()

from storage.relational_db import RelationalDB
from storage.models import Base

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def init_database(drop_existing: bool = False):
    """
    Create all tables defined in storage/models.py.
    
    Args:
        drop_existing: If True, drops existing tables before recreating.
    """
    db = RelationalDB()
    
    if drop_existing:
        logger.warning("Dropping all existing tables...")
        Base.metadata.drop_all(db.engine)
    
    logger.info("Creating database tables...")
    db.setup_database()
    
    logger.info("Database initialization complete!")
    logger.info("Tables created: stock_prices, financial_metrics, macro_indicators")
    
    return db


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Initialize the financial advisor database")
    parser.add_argument("--reset", action="store_true", help="Drop existing tables before creating")
    args = parser.parse_args()
    
    init_database(drop_existing=args.reset)
