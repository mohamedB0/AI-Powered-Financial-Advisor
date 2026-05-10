"""
celery_config.py
Celery configuration for the Financial Advisor project.
"""

import os
from dotenv import load_dotenv
load_dotenv()

REDIS_HOST = os.getenv("REDIS_HOST", "")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "")

USE_UPSTASH = os.getenv("USE_UPSTASH", "false").lower() == "true"

if USE_UPSTASH and REDIS_HOST and REDIS_HOST not in ("local", "localhost"):
    # Cloud Redis (Upstash) — uses rediss:// (TLS)
    BROKER_URL = f"rediss://:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}?ssl_cert_reqs=CERT_NONE"
    RESULT_BACKEND = BROKER_URL
elif REDIS_HOST in ("localhost", "127.0.0.1", "redis"):
    # Local or Docker Redis — uses redis:// (no TLS)
    if REDIS_PASSWORD:
        BROKER_URL = f"redis://:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}"
    else:
        BROKER_URL = f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}"
    
    # We MUST use SQLite for results so the Flask UI can run SELECT queries for the Pipeline Monitor
    RESULT_BACKEND = "db+sqlite:///celery_results.sqlite"
else:
    # Fallback: SQLite file-based broker (no Redis available)
    BROKER_URL = "sqla+sqlite:///celery_broker.sqlite"
    RESULT_BACKEND = "db+sqlite:///celery_results.sqlite"

broker_url = BROKER_URL
result_backend = RESULT_BACKEND

task_serializer = "json"
result_serializer = "json"
accept_content = ["json"]
timezone = "UTC"
enable_utc = True

task_track_started = True
task_time_limit = 3600
task_soft_time_limit = 1800
task_acks_late = True
task_reject_on_worker_lost = True

task_default_retry_delay = 60
task_max_retries = 3

worker_prefetch_multiplier = 1
worker_max_tasks_per_child = 100
worker_pool = "threads"

# Enable events for Flower monitoring
worker_send_task_events = True
task_send_sent_event = True
result_expires = 3600
result_persistent = True

task_routes = {
    "tasks.data_fetch.*": {"queue": "data_fetch"},
    "tasks.config_driven_scraping_tasks.*": {"queue": "scraping"},
    "tasks.document_ingestion_tasks.*": {"queue": "ingestion"},
    "tasks.local_ingestion_tasks.*": {"queue": "local_ingestion"},
    "tasks.bvc_fetch.*": {"queue": "data_fetch"},
}

task_default_queue = "default"

beat_schedule = {
    "fetch-stock-prices-daily": {
        "task": "tasks.data_fetch.fetch_stock_prices",
        "schedule": 86400.0,
        "options": {"queue": "data_fetch"},
    },
    "fetch-macro-indicators-weekly": {
        "task": "tasks.data_fetch.fetch_macro_indicators",
        "schedule": 604800.0,
        "options": {"queue": "data_fetch"},
    },
    "fetch-bvc-daily": {
        "task": "tasks.bvc_fetch.fetch_bvc_daily",
        "schedule": 86400.0,
        "options": {"queue": "data_fetch"},
    },
}

beat_scheduler = "celery.beat:PersistentScheduler"
beat_schedule_filename = "celerybeat-schedule.db"