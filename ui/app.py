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
    """Return only user-uploaded document ingestion tasks (chat uploads).
    Filters by task name AND kwargs containing 'user_data' to exclude
    local/scraping/background ingestion tasks."""
    if not Path(RESULTS_DB).exists():
        return jsonify({"tasks": []})
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
        # Remove raw kwargs from response (not needed by frontend)
        for row in rows:
            row.pop("kwargs", None)
        conn.close()
        return jsonify({"tasks": rows})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

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


if __name__ == "__main__":
    app.run(port=5000, debug=True)
