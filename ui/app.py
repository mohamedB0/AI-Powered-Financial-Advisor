import sys
import os
import json
import sqlite3
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from flask import Flask, render_template, request, jsonify
from retrieval.sql_search import SQLSearch

app = Flask(__name__)

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

if __name__ == "__main__":
    app.run(port=5000, debug=True)
