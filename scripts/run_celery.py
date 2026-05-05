#!/usr/bin/env python
"""
scripts/run_celery.py
Start Celery workers for async task processing.

Usage:
    # All-in-one (1 worker handling all queues)
    python scripts/run_celery.py --worker

    # Separate workers for different task types
    python scripts/run_celery.py --worker --queue ingestion --concurrency 4
    python scripts/run_celery.py --worker --queue data_fetch --concurrency 2

    # Beat scheduler (for scheduled tasks)
    python scripts/run_celery.py --beat

    # Flower monitoring
    python scripts/run_celery.py --flower
"""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


def start_worker(concurrency: int = 4, loglevel: str = "info", queue: str = "celery"):
    """Start Celery worker for a specific queue."""
    from tasks.celery_app import celery_app
    
    queues = [queue] if queue != "celery" else ["ingestion", "data_fetch", "default", "local_ingestion", "scraping"]
    logger.info(f"Starting Celery worker (queue={queues}, concurrency={concurrency})")
    
    celery_app.worker_main([
        "worker",
        f"--loglevel={loglevel}",
        f"--concurrency={concurrency}",
        f"--queues={','.join(queues)}",
        "--pool=solo",
    ])


def start_beat():
    """Start Celery beat scheduler for periodic tasks."""
    from tasks.celery_app import celery_app
    
    logger.info("Starting Celery beat scheduler")
    
    celery_app.Beat(
        loglevel="info",
        scheduler="celery.beat:PersistentScheduler",
    ).run()


def start_flower():
    """Open Flower monitoring dashboard."""
    import webbrowser
    from threading import Timer
    
    logger.info("Starting Flower monitoring dashboard...")
    
    def open_browser():
        webbrowser.open("http://localhost:5555")
    
    Timer(3, open_browser).start()
    
    from tasks.celery_app import celery_app
    celery_app.start()


def main():
    parser = argparse.ArgumentParser(description="Celery worker management")
    parser.add_argument(
        "--worker",
        action="store_true",
        help="Start Celery worker",
    )
    parser.add_argument(
        "--beat",
        action="store_true",
        help="Start Celery beat scheduler",
    )
    parser.add_argument(
        "--flower",
        action="store_true",
        help="Open Flower monitoring dashboard",
    )
    parser.add_argument(
        "--queue",
        type=str,
        default="all",
        choices=["all", "ingestion", "data_fetch", "default"],
        help="Queue to process (default: all)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Number of concurrent workers (default: 1)",
    )
    parser.add_argument(
        "--loglevel",
        type=str,
        default="info",
        choices=["debug", "info", "warning", "error"],
        help="Logging level (default: info)",
    )
    
    args = parser.parse_args()
    
    if args.beat:
        start_beat()
    elif args.flower:
        start_flower()
    elif args.worker or not any(vars(args).values()):
        queue = args.queue if args.queue != "all" else "celery"
        start_worker(concurrency=args.concurrency, loglevel=args.loglevel, queue=queue)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
