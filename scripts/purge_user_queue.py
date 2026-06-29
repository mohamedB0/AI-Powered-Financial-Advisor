"""
scripts/purge_user_queue.py
Purge pending user-upload ingestion tasks from the broker.

Only the 'user_ingestion' queue is purged — other queues (scraping,
data_fetch, ingestion, etc.) are left untouched.

Usage:
    python scripts/purge_user_queue.py
    python scripts/purge_user_queue.py --all   # purge every queue
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tasks.celery_app import celery_app

USER_QUEUE = "user_ingestion"


def purge_queue(queue_name: str) -> int:
    """Purge a single named queue. Returns the number of messages removed."""
    with celery_app.connection_for_write() as conn:
        channel = conn.default_channel
        try:
            return channel.queue_purge(queue_name) or 0
        except Exception as e:
            print(f"  ! Could not purge '{queue_name}': {e}")
            return 0


def main() -> None:
    purge_all = "--all" in sys.argv

    print(f"Broker: {celery_app.conf.broker_url}")

    if purge_all:
        removed = celery_app.control.purge() or 0
        print(f"Purged ALL queues — broker reports {removed} message(s) removed.")
    else:
        removed = purge_queue(USER_QUEUE)
        print(f"Purged '{USER_QUEUE}' — {removed} pending task(s) removed.")


if __name__ == "__main__":
    main()
