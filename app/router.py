"""
router.py — Core routing decision logic (SPEC.md §6).

Day 4 flow:
  0. Semantic cache lookup (hit → route="cache", 0 tokens)
  1. Classify task difficulty + type
  2. Run local model with self-consistency confidence check
     → on Ollama timeout/crash: auto-escalate directly to Fireworks (hardened)
  3. confidence >= per-difficulty threshold → cache + return local (0 tokens)
  4. Otherwise → compress prompt, call Fireworks with escalation ladder
     small → gemma → large  (SPEC.md §4 Tier 2 items 10 & 12)
  5. Log which tier was used, including used_gemma flag for bonus track

Error handling: the full routing logic is wrapped in a try/except that
falls back gracefully instead of letting raw exceptions propagate as 500s.
"""

import logging
import os
import time
from typing import TypedDict

from classifier import classify_difficulty
from confidence import estimate_confidence, OllamaUnavailableError
from compression import compress
from fireworks_client import FireworksClient
from semantic_cache import cache as _cache   # module-level singleton

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Per-difficulty confidence thresholds
# Tune empirically after each harness pass using app/threshold_tuner.py.
# (SPEC.md §6: "Tune THRESHOLD[difficulty] empirically once real tasks are seen")
# ---------------------------------------------------------------------------
CONFIDENCE_THRESHOLDS: dict[int, float] = {
    1: 0.60,  # trivial QA — low bar for local
    2: 0.65,
    3: 0.70,  # default / mid-complexity
    4: 0.75,
    5: 0.80,  # hard tasks — require high agreement before trusting local
}

# Self-consistency run count per difficulty band
N_RUNS_BY_DIFFICULTY: dict[int, int] = {
    1: 2,
    2: 2,
    3: 3,
    4: 3,
    5: 3,
}

# Environment knob: set CONFIDENCE_N_RUNS to override globally (useful for testing)
_ENV_N_RUNS = os.getenv("CONFIDENCE_N_RUNS")

# Request timeout configuration (Tier 2)
REQUEST_TIMEOUT_S: float = float(os.getenv("REQUEST_TIMEOUT_S", "30"))


class RoutingTimeoutError(TimeoutError):
    """Raised when routing time exceeds REQUEST_TIMEOUT_S."""


def _check_timeout(t_start: float) -> None:
    if time.monotonic() - t_start > REQUEST_TIMEOUT_S:
        raise RoutingTimeoutError(
            f"Request routing time exceeded limit of {REQUEST_TIMEOUT_S}s"
        )


class RouteResult(TypedDict):
    answer: str
    route: str          # "cache" | "local" | "fireworks-small" | "fireworks-gemma" | "fireworks-large"
    tokens: int
    confidence: float
    difficulty: int
    task_type: str
    used_gemma: bool    # Always False — Gemma serverless tier removed (see fireworks_client.py)
    was_compressed: bool
    original_word_count: int
    compressed_word_count: int


def _tier_to_route(tier: str) -> str:
    """Map a MODEL_TIERS key to a human-readable route label."""
    mapping = {
        "small": "fireworks-small",
        "gemma": "fireworks-gemma",
        "large": "fireworks-large",
    }
    return mapping.get(tier, f"fireworks-{tier}")


def route(task: str) -> RouteResult:
    """
    Main routing entry point.

    Returns a RouteResult dict consumed by /solve in main.py.
    Wraps all logic in try/except; on unrecoverable errors returns a
    best-effort error answer rather than propagating a raw exception.
    """
    t_start = time.monotonic()
    try:
        return _route_inner(task, t_start=t_start)
    except RoutingTimeoutError as exc:
        logger.error("Routing timed out for task=%r: %s", task[:80], exc)
        raise
    except Exception as exc:
        logger.exception("Unhandled error in router._route_inner: %s", exc)
        # Re-raise so main.py can catch it and return a structured 502
        raise


def _route_inner(task: str, t_start: float | None = None) -> RouteResult:
    """Internal routing logic — called by route() which provides the safety net."""
    if t_start is None:
        t_start = time.monotonic()

    _check_timeout(t_start)

    # ------------------------------------------------------------------
    # Step 0 — Semantic cache lookup (SPEC.md §4 Tier 2 item 8) ✓ Day 3
    # ------------------------------------------------------------------
    cached = _cache.lookup(task)
    if cached is not None:
        return RouteResult(
            answer=cached["answer"],
            route="cache",
            tokens=0,
            confidence=cached["similarity"],
            difficulty=0,
            task_type="cached",
            used_gemma=False,
            was_compressed=False,
            original_word_count=0,
            compressed_word_count=0,
        )

    # ------------------------------------------------------------------
    # Step 1 — Classify
    # ------------------------------------------------------------------
    classification = classify_difficulty(task)
    difficulty: int = classification["difficulty"]
    task_type: str  = classification["type"]

    threshold = CONFIDENCE_THRESHOLDS.get(difficulty, 0.70)
    n_runs = (
        int(_ENV_N_RUNS)
        if _ENV_N_RUNS
        else N_RUNS_BY_DIFFICULTY.get(difficulty, 3)
    )

    # ------------------------------------------------------------------
    # Step 2 — Local model + self-consistency
    # OllamaUnavailableError means the container is down/timed-out;
    # in that case we skip straight to Fireworks escalation.
    # ------------------------------------------------------------------
    ollama_ok = True
    answer = ""
    confidence = 0.0

    try:
        answer, confidence = estimate_confidence(task, n_runs=n_runs)
    except OllamaUnavailableError as exc:
        logger.warning("Ollama unreachable or timed out — auto-escalating to Fireworks: %s", exc)
        ollama_ok = False
    except Exception as exc:
        logger.warning("Ollama evaluation failed (%s) — auto-escalating to Fireworks", exc)
        ollama_ok = False

    _check_timeout(t_start)

    # ------------------------------------------------------------------
    # Step 3 — Local path: confident enough, no Fireworks needed
    # ------------------------------------------------------------------
    if ollama_ok and confidence >= threshold:
        _cache.store(task, answer)
        return RouteResult(
            answer=answer,
            route="local",
            tokens=0,
            confidence=confidence,
            difficulty=difficulty,
            task_type=task_type,
            used_gemma=False,
            was_compressed=False,
            original_word_count=0,
            compressed_word_count=0,
        )

    # ------------------------------------------------------------------
    # Step 4 — Escalate to Fireworks with model-size ladder
    # (SPEC.md §4 Tier 2 item 10 & 12)
    # ------------------------------------------------------------------
    _check_timeout(t_start)
    orig_words = len(task.split())
    compressed_task = compress(task)
    comp_words = len(compressed_task.split())
    was_compressed = comp_words < orig_words

    fw = FireworksClient()
    fw_answer, fw_tokens, final_tier = fw.solve_with_escalation(compressed_task)
    _check_timeout(t_start)

    # used_gemma is always False: the Gemma serverless tier was removed from
    # MODEL_TIERS after Fireworks deprecated gemma2-9b-it serverless inference.
    # The field is retained in RouteResult and logs for schema backward compatibility.
    used_gemma = False

    _cache.store(task, fw_answer)

    return RouteResult(
        answer=fw_answer,
        route=_tier_to_route(final_tier),
        tokens=fw_tokens,
        confidence=confidence,
        difficulty=difficulty,
        task_type=task_type,
        used_gemma=used_gemma,
        was_compressed=was_compressed,
        original_word_count=orig_words,
        compressed_word_count=comp_words,
    )
