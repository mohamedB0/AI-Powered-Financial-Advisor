"""
scripts/trigger_all.py
Triggers workers in the financial advisor pipeline:
  - Worker A (sites): config_driven_scraping    (queue: scraping) ← reads sites.json
  - Worker B:         local docs ingestion      (queue: local_ingestion → ingestion)

Usage:
    python scripts/trigger_all.py              # Everything
    python scripts/trigger_all.py --local-only
    python scripts/trigger_all.py --sites-only # Configured sites (sites.json)
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tasks.local_ingestion_tasks import scan_and_ingest_local_docs
from tasks.config_driven_scraping_tasks import scrape_all_configured_sites


def trigger_local():
    promise = scan_and_ingest_local_docs.apply_async(queue="local_ingestion")
    print(f"  [Local Docs]        Task sent → UUID: {promise.id}")


def trigger_sites():
    promise = scrape_all_configured_sites.apply_async(queue="scraping")
    print(f"  [Config Sites]     Task sent → UUID: {promise.id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trigger financial advisor pipeline workers")
    parser.add_argument("--local-only", action="store_true", help="Only trigger local docs ingestion")
    parser.add_argument("--sites-only", action="store_true", help="Only trigger sites.json scraping")
    args = parser.parse_args()

    print("=" * 60)
    print("  FINANCIAL ADVISOR — PIPELINE TRIGGER")
    print("=" * 60)

    if args.local_only:
        trigger_local()
    elif args.sites_only:
        trigger_sites()
    else:
        trigger_local()
        trigger_sites()

    print("=" * 60)
    print("All tasks dispatched. Monitor via Celery terminal.")