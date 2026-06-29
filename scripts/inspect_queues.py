"""
scripts/inspect_queues.py
Show all Celery queues on the Redis broker and how many messages each holds.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import redis
from tasks.celery_app import celery_app

broker = celery_app.conf.broker_url
print(f"Broker: {broker}\n")

r = redis.Redis.from_url(broker)

# In Celery's Redis transport, each queue is a Redis LIST keyed by the queue name.
# Reserved/active (in-flight) tasks live under 'unacked' (a hash) per worker.
print("── Pending messages per queue (Redis LISTs) ──")
found = False
for key in r.scan_iter("*"):
    try:
        if r.type(key) == b"list":
            name = key.decode("utf-8", "ignore")
            count = r.llen(key)
            if count > 0 or not name.startswith("_kombu"):
                print(f"  {name:<24} {count}")
                found = True
    except Exception:
        continue
if not found:
    print("  (no list-type queues found)")

print("\n── In-flight (reserved but unacked) tasks ──")
try:
    unacked = r.hlen("unacked")
    print(f"  unacked: {unacked}")
except Exception as e:
    print(f"  could not read 'unacked': {e}")

print("\n── Live worker inspection ──")
insp = celery_app.control.inspect()
reserved = insp.reserved() or {}
active = insp.active() or {}
if not reserved and not active:
    print("  No workers responded (worker may be offline).")
for w, tasks in reserved.items():
    print(f"  [{w}] reserved: {len(tasks)}")
for w, tasks in active.items():
    print(f"  [{w}] active: {len(tasks)}")
