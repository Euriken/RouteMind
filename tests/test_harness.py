#!/usr/bin/env python3
"""
tests/test_harness.py — Self-evaluation harness for RouteMind (Day 3).

Usage:
    # With the stack running (docker-compose up):
    python tests/test_harness.py

    # Point at a different host:
    ROUTEMIND_URL=http://localhost:5000 python tests/test_harness.py

Covers a representative mix of task types per SPEC.md §7:
    2 × QA (factual)
    1 × reasoning (multi-step logic)
    1 × summarization
    2 × code

Each task has a keyword/substring correctness check — not full semantic
grading, but enough to catch badly wrong answers and measure route efficiency.

Self-eval harness (Day 3). Full adaptive threshold tuning happens Day 4
per SPEC.md §8.
"""

import os
import sys
import time
import json
import requests

BASE_URL = os.getenv("ROUTEMIND_URL", "http://localhost:5000")
SOLVE_URL = f"{BASE_URL}/solve"
REQUEST_TIMEOUT = 180   # seconds — local model can be slow

# ---------------------------------------------------------------------------
# Task suite
# Each entry: task text + list of keywords that must appear in the answer
# (case-insensitive substring match). ANY match = pass.
# ---------------------------------------------------------------------------
TASKS = [
    {
        "id": "qa-1",
        "type": "QA",
        "task": "What is the capital city of Japan?",
        "expect_any": ["tokyo"],
    },
    {
        "id": "qa-2",
        "type": "QA",
        "task": "Who wrote the play Romeo and Juliet?",
        "expect_any": ["shakespeare", "william shakespeare"],
    },
    {
        "id": "reasoning-1",
        "type": "reasoning",
        "task": (
            "Alice has twice as many apples as Bob. Bob has 3 more apples than Carol. "
            "Carol has 5 apples. How many apples does Alice have? "
            "Show your step-by-step reasoning."
        ),
        "expect_any": ["16"],
    },
    {
        "id": "summarization-1",
        "type": "summarization",
        "task": (
            "Summarize the following paragraph in one sentence:\n\n"
            "The Python programming language was created by Guido van Rossum and first "
            "released in 1991. It emphasizes code readability and uses significant "
            "indentation. Python supports multiple programming paradigms, including "
            "structured, object-oriented, and functional programming. It is dynamically "
            "typed and garbage-collected, and has a large standard library often "
            "described as 'batteries included'."
        ),
        "expect_any": ["python", "guido", "1991", "programming", "readability"],
    },
    {
        "id": "code-1",
        "type": "code",
        "task": (
            "Write a Python function called `fibonacci` that takes an integer n "
            "and returns the nth Fibonacci number (0-indexed, so fibonacci(0)=0, "
            "fibonacci(1)=1, fibonacci(6)=8). Use iteration, not recursion."
        ),
        "expect_any": ["def fibonacci", "fibonacci(", "return"],
    },
    {
        "id": "code-2",
        "type": "code",
        "task": (
            "The following Python function has a bug — it should return the "
            "second-largest number in a list, but it always returns the largest. "
            "Fix it:\n\n"
            "def second_largest(nums):\n"
            "    return sorted(nums)[-1]\n"
        ),
        "expect_any": ["[-2]", "sorted(nums)[-2]", "second", "-2"],
    },
]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_task(entry: dict) -> dict:
    """POST one task to /solve and return a result record."""
    t0 = time.monotonic()
    try:
        resp = requests.post(
            SOLVE_URL,
            json={"task": entry["task"]},
            timeout=REQUEST_TIMEOUT,
        )
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.RequestException as exc:
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        return {
            "id":          entry["id"],
            "type":        entry["type"],
            "task":        entry["task"],   # stored for threshold_tuner.py
            "difficulty":  0,              # unknown on error
            "route":       "ERROR",
            "tokens":      0,
            "confidence":  0.0,
            "latency_ms":  elapsed_ms,
            "correct":     False,
            "error":       str(exc),
        }

    answer = data.get("answer", "")
    correct = any(kw in answer.lower() for kw in entry["expect_any"])

    return {
        "id":          entry["id"],
        "type":        entry["type"],
        "task":        entry["task"],              # stored for threshold_tuner.py
        "difficulty":  data.get("difficulty", 0), # from /solve response
        "route":       data.get("route", "?"),
        "tokens":      data.get("tokens", 0),
        "confidence":  data.get("confidence", 0.0),
        "latency_ms":  elapsed_ms,
        "correct":     correct,
        "answer_snip": answer[:120].replace("\n", " "),
    }


def print_table(results: list[dict]) -> None:
    col_w = [10, 14, 18, 8, 12, 12, 8]
    headers = ["id", "type", "route", "tokens", "confidence", "latency_ms", "correct"]
    sep = "  ".join("-" * w for w in col_w)

    def row(vals):
        return "  ".join(str(v)[:w].ljust(w) for v, w in zip(vals, col_w))

    print("\n" + "=" * 90)
    print("  RouteMind Self-Eval Harness — Results")
    print("=" * 90)
    print(row(headers))
    print(sep)
    for r in results:
        vals = [
            r["id"],
            r["type"],
            r.get("route", "?"),
            r["tokens"],
            f"{r['confidence']:.3f}",
            r["latency_ms"],
            "PASS" if r["correct"] else "FAIL",
        ]
        print(row(vals))

    print(sep)
    total   = len(results)
    passed  = sum(1 for r in results if r["correct"])
    errors  = sum(1 for r in results if r.get("route") == "ERROR")
    total_t = sum(r["tokens"] for r in results)
    avg_lat = int(sum(r["latency_ms"] for r in results) / total) if total else 0

    print(f"\n  Tasks: {total}  |  Passed: {passed}/{total}  |  "
          f"Errors: {errors}  |  Total tokens: {total_t}  |  Avg latency: {avg_lat} ms")

    routes = {}
    for r in results:
        routes[r.get("route", "?")] = routes.get(r.get("route", "?"), 0) + 1
    print(f"  Route distribution: {routes}")
    print("=" * 90 + "\n")


def main() -> None:
    print(f"[harness] Connecting to {SOLVE_URL} ...")
    try:
        requests.get(BASE_URL, timeout=5)
    except requests.exceptions.ConnectionError:
        print(f"[harness] ERROR: Cannot reach {BASE_URL}. Is the stack running?")
        sys.exit(1)

    results = []
    for entry in TASKS:
        print(f"[harness] Running {entry['id']} ({entry['type']}) ...", end=" ", flush=True)
        result = run_task(entry)
        status = "PASS" if result["correct"] else "FAIL"
        print(f"{status}  route={result['route']}  tokens={result['tokens']}  {result['latency_ms']}ms")
        results.append(result)

    print_table(results)

    # Write results to a JSON file for later analysis
    out_path = os.path.join(os.path.dirname(__file__), "..", "logs", "harness_results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"[harness] Full results written to {os.path.normpath(out_path)}\n")


if __name__ == "__main__":
    main()
