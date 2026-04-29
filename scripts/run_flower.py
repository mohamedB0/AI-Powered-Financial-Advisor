"""
scripts/run_flower.py
Start Flower monitoring UI for Celery tasks.

Usage:
    python scripts/run_flower.py
    
Then open http://localhost:5555 in your browser.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv
load_dotenv()

USE_UPSTASH = os.getenv("USE_UPSTASH", "false").lower() == "true"
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")

if not USE_UPSTASH and REDIS_HOST in ("local", "localhost"):
    # Use SQLite broker
    BROKER_URL = "sqla+sqlite:///celery_broker.sqlite"
    print("Using SQLite broker")
else:
    # Use Redis/Upstash
    REDIS_HOST = os.getenv("REDIS_HOST")
    REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "")
    REDIS_DB = int(os.getenv("REDIS_DB", "0"))
    BROKER_URL = f"rediss://:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}?ssl_cert_reqs=CERT_NONE"
    print(f"Using Redis broker: {REDIS_HOST}")

if __name__ == "__main__":
    print("=" * 50)
    print("Starting Flower monitoring UI...")
    print("Broker URL:", BROKER_URL)
    print("Open http://localhost:5555 in your browser")
    print("=" * 50)
    
    import subprocess
    subprocess.run([
        sys.executable, "-m", "celery",
        "-A", "tasks.celery_app",
        "flower",
        "-b", BROKER_URL,
        "--port=5555"
    ])