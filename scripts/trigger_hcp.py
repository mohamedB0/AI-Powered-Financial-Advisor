"""
scripts/trigger_hcp.py   — launches the HCP financial document crawling pipeline.
scripts/trigger_cdc.py   — launches the CDC publication crawling pipeline.

Run in separate terminals alongside the Celery workers.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tasks.hcp_scraping_tasks import scrape_hcp_catalog

if __name__ == '__main__':
    print("=" * 60)
    print("TRIGGERING HCP FINANCIAL DOCUMENTS PIPELINE")
    print("=" * 60)
    promise = scrape_hcp_catalog.apply_async(queue="hcp_scraping")
    print(f"Initial task sent to HCP Scraping Worker! (UUID: {promise.id})")
    print("Watch your Celery worker terminals for progress.")
