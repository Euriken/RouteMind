"""
confidence.py — Self-consistency confidence estimator.

Runs the local Ollama model n_runs times on the same task, compares
answers with a lightweight string-similarity approach, and returns the
most common answer along with a confidence score [0.0, 1.0].

Day 4 hardening (SPEC.md §4 Tier 3 item 15):
  • Split connect vs. read timeouts: 5 s connect (fast-fail if Ollama is down)
    and OLLAMA_READ_TIMEOUT_S (default 120 s) for actual inference.
  • If Ollama is unreachable or all calls time out, raises OllamaUnavailableError
    so the router can auto-escalate to Fireworks instead of crashing.

Note: a 1.5B parameter model doing CPU inference can take 20-60 s for a
non-trivial response — the read timeout must be generous enough to allow
that, while the connect timeout keeps us from hanging on a dead container.
"""

import re
import os
import logging
from collections import Counter
from difflib import SequenceMatcher

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Timeout configuration.
#   OLLAMA_CONNECT_TIMEOUT_S  — time to detect a down/unreachable Ollama (5 s)
#   OLLAMA_READ_TIMEOUT_S     — max wall-clock time for a single inference call
# The requests timeout arg accepts (connect, read) as a tuple.
# ---------------------------------------------------------------------------
OLLAMA_CONNECT_TIMEOUT_S: int = int(os.getenv("OLLAMA_CONNECT_TIMEOUT_S", "5"))
OLLAMA_READ_TIMEOUT_S: int    = int(os.getenv("OLLAMA_READ_TIMEOUT_S", "120"))


class OllamaUnavailableError(RuntimeError):
    """Raised when Ollama is unreachable or all calls timed out."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalise(text: str) -> str:
    """Strip whitespace/punctuation and lowercase for comparison."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def _similarity(a: str, b: str) -> float:
    """SequenceMatcher ratio between two normalised strings (0-1)."""
    return SequenceMatcher(None, _normalise(a), _normalise(b)).ratio()


def _extract_core_answer(text: str) -> str | None:
    """
    Extract a clean numeric or short factual core (e.g., standalone number or short entity)
    from short/single-sentence answers.
    Returns None for open-ended or multi-sentence answers so they fall back to SequenceMatcher.
    """
    cleaned = text.strip()
    # Do not apply to multi-sentence or long open-ended answers (> 20 words or multiple periods)
    if len(cleaned.split()) > 20 or cleaned.count(".") > 1 or "\n" in cleaned:
        return None

    # Check for explicit answer indicators like "is 4", "= 4", "equals 4", "answer is 4", ": 16"
    match = re.search(r"(?:is|equals|=|answer\s+is|result\s+is|->|:)\s*(-?\d+(?:\.\d+)?)\b", cleaned, re.IGNORECASE)
    if match:
        return match.group(1)

    # Try to match a standalone number if there's exactly one in the answer
    nums = re.findall(r"\b-?\d+(?:\.\d+)?\b", cleaned)
    if len(nums) == 1:
        return nums[0]

    # Try to match a short, direct noun phrase/entity if the answer is very short (<= 5 words)
    if len(cleaned.split()) <= 5:
        return _normalise(cleaned)

    return None


def _cluster_answers(answers: list[str], threshold: float = 0.75) -> list[list[str]]:
    """
    Group answers into clusters where every pair has similarity >= threshold
    OR identical extracted numeric/factual core answers.
    Simple greedy approach: good enough for n_runs <= 5.
    """
    clusters: list[list[str]] = []
    for ans in answers:
        placed = False
        ans_core = _extract_core_answer(ans)
        for cluster in clusters:
            cluster_rep = cluster[0]
            rep_core = _extract_core_answer(cluster_rep)

            sim_ok = _similarity(ans, cluster_rep) >= threshold
            core_ok = bool(ans_core is not None and rep_core is not None and ans_core == rep_core)

            if sim_ok or core_ok:
                cluster.append(ans)
                placed = True
                break
        if not placed:
            clusters.append([ans])
    return clusters


def _call_ollama(prompt: str) -> str:
    """
    Single blocking call to Ollama.

    Uses a (connect, read) timeout tuple:
      - connect: OLLAMA_CONNECT_TIMEOUT_S (5 s) — fast-fail on dead container
      - read:    OLLAMA_READ_TIMEOUT_S (120 s) — allow full CPU inference time

    Raises:
        OllamaUnavailableError — on connection error or connect-phase timeout.
    """
    host  = os.getenv("OLLAMA_HOST",  "http://ollama:11434")
    model = os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b")

    try:
        resp = requests.post(
            f"{host}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=(OLLAMA_CONNECT_TIMEOUT_S, OLLAMA_READ_TIMEOUT_S),
        )
        resp.raise_for_status()
        return resp.json().get("response", "")

    except (requests.exceptions.Timeout, requests.exceptions.ConnectionError, requests.exceptions.RequestException) as exc:
        logger.warning("Ollama unreachable or timed out: %s", exc)
        raise OllamaUnavailableError(
            f"Ollama unreachable or timed out: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def estimate_confidence(
    task: str,
    n_runs: int = 3,
) -> tuple[str, float]:
    """
    Run the local model n_runs times and measure self-consistency.

    Raises:
        OllamaUnavailableError — if every run fails due to Ollama being down.
            Router catches this and auto-escalates to Fireworks.

    Returns:
        (best_answer, confidence_score)
        - best_answer: the answer from the largest agreement cluster
        - confidence_score: fraction of runs that agreed with the best answer
    """
    answers: list[str] = []
    last_unavailable: OllamaUnavailableError | None = None

    for _ in range(n_runs):
        try:
            ans = _call_ollama(task)
            answers.append(ans)
        except OllamaUnavailableError as exc:
            # Track but keep trying remaining runs (Ollama may recover)
            last_unavailable = exc
        except Exception:
            # Any other error (HTTP 5xx etc.) — skip this run
            pass

    if not answers:
        # All runs failed — propagate so the router can escalate
        if last_unavailable:
            raise last_unavailable
        raise OllamaUnavailableError("All Ollama calls failed with unknown errors")

    if len(answers) == 1:
        return (answers[0], 0.5)   # single run — moderate uncertainty

    clusters = _cluster_answers(answers)
    best_cluster = max(clusters, key=len)
    confidence   = len(best_cluster) / len(answers)
    best_answer  = min(best_cluster, key=len)   # shortest = most representative

    return (best_answer, round(confidence, 4))
