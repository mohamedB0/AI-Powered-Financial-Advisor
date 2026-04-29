"""
tasks/celery_app.py
Celery application instance configuration.
"""

from celery import Celery
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import configuration
import celery_config

# Create Celery app
celery_app = Celery("financial_advisor")

# Configure from module
celery_app.config_from_object(celery_config)

# Auto-discover tasks
celery_app.autodiscover_tasks(["tasks"])


@celery_app.task(bind=True, name="tasks.ping")
def ping(self):
    """Health check task."""
    return {"status": "pong", "task_id": self.request.id}
