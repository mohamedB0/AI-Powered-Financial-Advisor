"""
scripts/trigger_scraping.py
Triggers the universal web scraping pipeline for Cour des Comptes (CDC).

Usage:
    python scripts/trigger_scraping.py

Make sure the scraping worker is running:
    celery -A tasks.celery_app worker -Q scraping --loglevel=INFO --pool=solo -n scraping_worker@%h
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tasks.web_scraping_tasks import scrape_catalog_index

if __name__ == '__main__':
    print("=" * 60)
    print("TRIGGERING WEB SCRAPING PIPELINE (CDC)")
    print("=" * 60)
    promise = scrape_catalog_index.apply_async()
    print(f"Task sent to the scraping worker! (UUID: {promise.id})")
    print("Check your Celery scraping terminal or Flower to monitor progress.")
