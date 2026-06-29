"""Scan all Redis broker keys: type, size, and any unacked task payloads."""
import sys, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import redis
from tasks.celery_app import celery_app

r = redis.Redis.from_url(celery_app.conf.broker_url)

print("── All broker keys ──")
for key in r.scan_iter("*"):
    name = key.decode("utf-8", "ignore")
    ktype = r.type(key).decode()
    if ktype == "list":
        size = r.llen(key)
    elif ktype == "hash":
        size = r.hlen(key)
    elif ktype == "zset":
        size = r.zcard(key)
    elif ktype == "set":
        size = r.scard(key)
    else:
        size = "-"
    print(f"  {name:<28} {ktype:<6} {size}")

# Show unacked payloads (where a grabbed-but-unfinished task lives)
print("\n── 'unacked' hash contents (in-flight tasks) ──")
try:
    h = r.hgetall("unacked")
    if not h:
        print("  (empty)")
    for field, raw in h.items():
        try:
            payload = json.loads(raw)
            # redis transport stores [message, exchange, routing_key]
            msg = payload[0] if isinstance(payload, list) else payload
            headers = msg.get("headers", {})
            print(f"  {field.decode()[:12]}  task={headers.get('task')}  args={headers.get('argsrepr')}")
        except Exception as e:
            print(f"  {field.decode()[:12]}  (unparsed: {e})")
except Exception as e:
    print(f"  could not read: {e}")
