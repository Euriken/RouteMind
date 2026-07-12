"""
main.py — Flask orchestrator entry point.

Endpoints:
    POST /solve   — route a task through the RouteMind pipeline
    GET  /health  — liveness check: verifies Flask is up and probes Ollama
                    (Day 4 hardening, SPEC.md §4 Tier 3 item 15)
"""

import time
import os
import logging
import atexit

import requests as _requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

from router import route
from logger import log_run
from stats import get_stats
from semantic_cache import cache as _cache

load_dotenv()
logging.basicConfig(level=logging.INFO)

@atexit.register
def save_cache_on_exit():
    logging.info("Flask server shutting down. Saving semantic cache to disk...")
    try:
        _cache.save_to_disk()
        logging.info("Semantic cache successfully saved to disk.")
    except Exception as e:
        logging.error("Failed to save semantic cache on exit: %s", e)

app = Flask(__name__)
CORS(app)  # allow cross-origin requests from the browser frontend

_OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama:11434")


# ---------------------------------------------------------------------------
# /health — liveness + Ollama reachability probe
# ---------------------------------------------------------------------------

@app.route("/health", methods=["GET"])
def health():
    """
    Returns {"status": "ok", "ollama_reachable": true/false}.
    Always returns HTTP 200 — the caller interprets ollama_reachable.
    """
    ollama_reachable = False
    try:
        resp = _requests.get(f"{_OLLAMA_HOST}/api/tags", timeout=3)
        ollama_reachable = resp.status_code == 200
    except Exception:
        pass

    return jsonify({"status": "ok", "ollama_reachable": ollama_reachable})


# ---------------------------------------------------------------------------
# /stats — telemetry and savings metrics (Tier 1)
# ---------------------------------------------------------------------------

@app.route("/stats", methods=["GET"])
def stats():
    """
    Returns routing statistics, token counts, percentages, and cache savings.
    """
    return jsonify(get_stats())


# ---------------------------------------------------------------------------
# /solve — main routing endpoint
# ---------------------------------------------------------------------------

@app.route("/solve", methods=["POST"])
def solve():
    data = request.get_json(force=True)
    task = data.get("task", "").strip()

    if not task:
        return jsonify({"error": "Missing 'task' field"}), 400

    t_start = time.monotonic()

    try:
        result = route(task)
    except Exception as exc:
        latency_ms = int((time.monotonic() - t_start) * 1000)
        logging.error("Route failed for task=%r: %s", task[:80], exc)
        return jsonify({"error": f"Routing failed: {exc}"}), 502

    latency_ms = int((time.monotonic() - t_start) * 1000)

    log_run(
        task=task,
        route=result["route"],
        tokens=result["tokens"],
        confidence=result["confidence"],
        latency_ms=latency_ms,
        difficulty=result["difficulty"],
        task_type=result["task_type"],
        used_gemma=result.get("used_gemma", False),
    )

    return jsonify({
        "answer":      result["answer"],
        "route":       result["route"],
        "tokens":      result["tokens"],
        "confidence":  result["confidence"],
        "difficulty":  result["difficulty"],
        "task_type":   result["task_type"],
        "used_gemma":  result.get("used_gemma", False),
        "was_compressed": result.get("was_compressed", False),
        "original_word_count": result.get("original_word_count", 0),
        "compressed_word_count": result.get("compressed_word_count", 0),
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
