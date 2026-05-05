"""
scripts/run_flower.py
Start Flower monitoring UI for Celery tasks.

Usage:
    python scripts/run_flower.py
    
Then open http://localhost:5555 in your browser.

NOTE: When using SQLite broker (local mode), Flower has limited real-time
worker visibility. Workers must be started with -E (events) flag:
  celery -A tasks.celery_app worker -Q ingestion --pool=solo -n ingestion_worker@%h --loglevel=INFO -E
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv
load_dotenv()

USE_UPSTASH = os.getenv("USE_UPSTASH", "false").lower() == "true"
REDIS_HOST = os.getenv("REDIS_HOST", "")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "")
REDIS_DB = int(os.getenv("REDIS_DB", "0"))

if USE_UPSTASH and REDIS_HOST and REDIS_HOST not in ("local", "localhost"):
    BROKER_URL = f"rediss://:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}?ssl_cert_reqs=CERT_NONE"
    print(f"Using Upstash Redis broker: {REDIS_HOST}")
elif REDIS_HOST in ("localhost", "127.0.0.1"):
    if REDIS_PASSWORD:
        BROKER_URL = f"redis://:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}"
    else:
        BROKER_URL = f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}"
    print(f"Using local Docker Redis broker: {REDIS_HOST}:{REDIS_PORT}")
else:
    BROKER_URL = "sqla+sqlite:///celery_broker.sqlite"
    print("Using SQLite broker (limited real-time monitoring)")

if __name__ == "__main__":
    print("=" * 50)
    print("Starting Flower monitoring UI...")
    print("Broker URL:", BROKER_URL)
    print("Open http://localhost:5555 in your browser")
    print("=" * 50)
    
    import subprocess
    cmd = [
        sys.executable, "-m", "celery",
        "-A", "tasks.celery_app",
        "flower",
        "--broker", BROKER_URL,
        "--port=5555",
        "--persistent=True",
        "--db=flower.db",
    ]
    subprocess.run(cmd)