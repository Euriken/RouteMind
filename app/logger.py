"""
logger.py — Append-only JSONL run logger.

Each call to log_run() writes one JSON line to logs/runs.jsonl.
The logs/ directory is created automatically if it does not exist.

Schema per line:
    {
        "ts":          ISO-8601 UTC timestamp,
        "task":        first 200 chars of the task (truncated for readability),
        "route":       "local" | "cache" | "fireworks-small" | "fireworks-gemma" | "fireworks-large",
        "tokens":      int,
        "confidence":  float,
        "latency_ms":  int,
        "difficulty":  int (1-5),
        "task_type":   str,
        "used_gemma":  bool  — True when the Gemma tier was the final model used
                              (Day 4: bonus track tracking, SPEC.md §4 Tier 2 item 12)
    }
"""

import json
import os
from datetime import datetime, timezone

# Path is relative to the project root; works inside Docker because the
# working directory is /app and logs/ is mounted or created at runtime.
_LOG_DIR  = os.path.join(os.path.dirname(__file__), "..", "logs")
_LOG_FILE = os.path.join(_LOG_DIR, "runs.jsonl")


def _ensure_log_dir() -> None:
    os.makedirs(_LOG_DIR, exist_ok=True)


def log_run(
    task: str,
    route: str,
    tokens: int,
    confidence: float,
    latency_ms: int,
    difficulty: int = 0,
    task_type: str = "",
    used_gemma: bool = False,
) -> None:
    """
    Append a single JSON line to logs/runs.jsonl.

    Args:
        task:        The original task string (truncated to 200 chars in the log).
        route:       Routing decision, e.g. "local", "fireworks-gemma".
        tokens:      Total tokens billed for this request (0 if local/cache).
        confidence:  Self-consistency score from confidence.py [0.0, 1.0].
        latency_ms:  Wall-clock time from request received to answer returned.
        difficulty:  Classifier difficulty score 1-5.
        task_type:   Classifier task type string.
        used_gemma:  True when the Gemma tier was used (Day 4 bonus tracking).
    """
    _ensure_log_dir()

    record = {
        "ts":          datetime.now(timezone.utc).isoformat(),
        "task":        task[:200],
        "route":       route,
        "tokens":      tokens,
        "confidence":  confidence,
        "latency_ms":  latency_ms,
        "difficulty":  difficulty,
        "task_type":   task_type,
        "used_gemma":  used_gemma,
    }

    with open(_LOG_FILE, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")
