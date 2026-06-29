import sys
import os
import json
import sqlite3
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from flask import Flask, render_template, request, jsonify
from retrieval.sql_search import SQLSearch

app = Flask(__name__)

# Pass GOOGLE_CLIENT_ID to all templates
@app.context_processor
def inject_google_client_id():
    return {"GOOGLE_CLIENT_ID": os.getenv("GOOGLE_CLIENT_ID", "")}

RESULTS_DB = os.getenv(
    "CELERY_RESULTS_DB",
    str(Path(__file__).parent.parent / "celery_results.sqlite")
)

@app.route("/")
@app.route("/chat")
def chat():
    return render_template("chat.html", active_page="chat")

@app.route("/pipeline")
def pipeline():
    return render_template("pipeline.html", active_page="pipeline")

@app.route("/api/pipeline/tasks")
def api_pipeline_tasks():
    """Return user-uploaded document ingestion tasks from the Celery result backend (Redis).
    Falls back to SQLite if the old database exists."""
    import json as _json

    tasks = []

    # Try Redis result backend first (current configuration)
    try:
        import redis
        r = redis.Redis.from_url(
            f"redis://{os.getenv('REDIS_HOST', 'localhost')}:{os.getenv('REDIS_PORT', '6379')}/{os.getenv('REDIS_DB', '0')}"
        )
        # Celery stores results in Redis as keys: celery-task-meta-{task_id}
        for key in r.scan_iter("celery-task-meta-*"):
            try:
                raw = r.get(key)
                if not raw:
                    continue
                meta = _json.loads(raw)
                result = meta.get("result")
                status = meta.get("status", "UNKNOWN")

                # Include all ingestion task results (success or error)
                if isinstance(result, dict) and "filename" in result:
                    task_id = meta.get("task_id", key.decode().replace("celery-task-meta-", ""))
                    tasks.append({
                        "task_id": task_id,
                        "status": status,
                        "result": _json.dumps(result) if isinstance(result, dict) else str(result),
                        "date_done": meta.get("date_done", ""),
                        "traceback": meta.get("traceback", ""),
                        "name": "tasks.document_ingestion_tasks.ingest_document",
                    })
            except Exception:
                continue
    except Exception as e:
        print(f"[Pipeline] Redis query failed: {e}")

    # Fallback: try SQLite if Redis had no results (legacy data)
    if not tasks and Path(RESULTS_DB).exists():
        try:
            conn = sqlite3.connect(RESULTS_DB)
            conn.row_factory = sqlite3.Row
            cur = conn.execute(
                "SELECT task_id, status, result, date_done, traceback, name, kwargs "
                "FROM celery_taskmeta "
                "WHERE name = 'tasks.document_ingestion_tasks.ingest_document' "
                "  AND kwargs LIKE '%user_data%' "
                "ORDER BY date_done DESC LIMIT 200"
            )
            rows = [dict(r) for r in cur.fetchall()]
            for row in rows:
                row.pop("kwargs", None)
            conn.close()
            tasks = rows
        except Exception:
            pass

    # Sort by date descending
    tasks.sort(key=lambda t: t.get("date_done", ""), reverse=True)
    return jsonify({"tasks": tasks[:200]})

@app.route("/api/auth/google", methods=["POST"])
def proxy_google_auth():
    """Proxy Google auth to FastAPI backend (avoids CORS issues)."""
    import requests as http_requests
    api_url = os.getenv("API_BASE_URL", "http://localhost:8000")
    try:
        resp = http_requests.post(
            f"{api_url}/auth/google",
            json=request.get_json(),
            headers={"Content-Type": "application/json"},
            timeout=10,
        )
        try:
            data = resp.json()
        except Exception:
            data = {"detail": f"Backend returned non-JSON (status {resp.status_code}): {resp.text[:200]}"}
        return jsonify(data), resp.status_code
    except http_requests.exceptions.ConnectionError:
        return jsonify({"detail": "Cannot connect to FastAPI backend. Is it running?"}), 502
    except Exception as e:
        return jsonify({"detail": f"Proxy error: {type(e).__name__}: {str(e)}"}), 502


@app.route("/api/auth/register", methods=["POST"])
def proxy_register():
    """Proxy register to FastAPI backend."""
    import requests as http_requests
    api_url = os.getenv("API_BASE_URL", "http://localhost:8000")
    try:
        resp = http_requests.post(
            f"{api_url}/auth/register",
            json=request.get_json(),
            headers={"Content-Type": "application/json"},
            timeout=10,
        )
        try:
            data = resp.json()
        except Exception:
            data = {"detail": f"Backend error (status {resp.status_code})"}
        return jsonify(data), resp.status_code
    except http_requests.exceptions.ConnectionError:
        return jsonify({"detail": "Cannot connect to FastAPI backend."}), 502
    except Exception as e:
        return jsonify({"detail": f"Proxy error: {str(e)}"}), 502


@app.route("/api/auth/login", methods=["POST"])
def proxy_login():
    """Proxy login to FastAPI backend."""
    import requests as http_requests
    api_url = os.getenv("API_BASE_URL", "http://localhost:8000")
    try:
        resp = http_requests.post(
            f"{api_url}/auth/login",
            json=request.get_json(),
            headers={"Content-Type": "application/json"},
            timeout=10,
        )
        try:
            data = resp.json()
        except Exception:
            data = {"detail": f"Backend error (status {resp.status_code})"}
        return jsonify(data), resp.status_code
    except http_requests.exceptions.ConnectionError:
        return jsonify({"detail": "Cannot connect to FastAPI backend."}), 502
    except Exception as e:
        return jsonify({"detail": f"Proxy error: {str(e)}"}), 502


if __name__ == "__main__":
    app.run(port=5000, debug=True)
