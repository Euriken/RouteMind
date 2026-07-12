#!/usr/bin/env python3
"""
threshold_tuner.py — Adaptive confidence threshold analysis tool.

SPEC.md §4 Tier 2 item 11: "Adaptive threshold tuning — track running accuracy
vs token spend, tighten threshold if comfortably above floor, loosen if close
to floor."

NOT part of the live API.  Run this manually after each harness pass:

    python app/threshold_tuner.py
    python app/threshold_tuner.py --log logs/runs.jsonl --harness logs/harness_results.json
    python app/threshold_tuner.py --floor 0.90 --step 0.05

How it works
------------
1. Load ground-truth pass/fail from logs/harness_results.json (written by
   tests/test_harness.py).  The harness knows the expected answers.
2. Join against logs/runs.jsonl on the task prefix to get difficulty level.
3. Per difficulty level, compute:
       accuracy  = correct / total tasks at that difficulty
       avg_tokens = mean tokens spent per task at that difficulty
4. Suggest threshold adjustments:
       accuracy > ACCURACY_FLOOR + COMFORT_BAND  → tighten (raise) by STEP
       accuracy < ACCURACY_FLOOR                 → loosen  (lower) by STEP
       else                                      → keep current threshold
5. Print a comparison table.  Does NOT write back to router.py automatically.

Thresholds to copy-paste back into router.py after review are printed at the
end of the output.
"""

import argparse
import json
import os
import sys

# ---------------------------------------------------------------------------
# Tunables — change via CLI flags or edit defaults here
# ---------------------------------------------------------------------------
ACCURACY_FLOOR: float = 0.85   # minimum acceptable accuracy per difficulty level
COMFORT_BAND:   float = 0.08   # if accuracy > floor + band  → tighten threshold
STEP:           float = 0.05   # amount to raise or lower threshold per round

# Mirror of router.py CONFIDENCE_THRESHOLDS — kept in sync manually.
CURRENT_THRESHOLDS: dict[int, float] = {
    1: 0.60,
    2: 0.65,
    3: 0.70,
    4: 0.75,
    5: 0.80,
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _load_json(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _task_prefix(task: str, n: int = 60) -> str:
    """Normalised prefix for fuzzy-joining harness results to run logs (fallback only)."""
    return task.strip().lower()[:n]


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyse(
    harness_results: list[dict],
    run_logs: list[dict],
    accuracy_floor: float,
    comfort_band: float,
    step: float,
) -> None:
    """
    Compute per-difficulty accuracy and suggest threshold adjustments.
    Prints a formatted table.  Does not modify any files.

    Primary source for difficulty: the "difficulty" field in harness_results.json
    (populated by test_harness.py from the /solve API response).
    Falls back to a prefix-join against runs.jsonl if the field is missing.
    """

    # Build a prefix → difficulty lookup from run logs (fallback)
    prefix_to_diff: dict[str, int] = {}
    for rec in run_logs:
        task_text = rec.get("task", "")
        diff      = rec.get("difficulty", 0)
        if diff and task_text:
            prefix_to_diff[_task_prefix(task_text)] = diff

    # Aggregate per difficulty: {diff: {"correct": int, "total": int, "tokens": int}}
    stats: dict[int, dict] = {}

    for res in harness_results:
        # Use difficulty stored in harness result; fall back to run-log join
        diff = res.get("difficulty", 0)
        if not diff:
            task_text = res.get("task", "")
            diff = prefix_to_diff.get(_task_prefix(task_text), 0)

        if diff not in stats:
            stats[diff] = {"correct": 0, "total": 0, "tokens": 0}

        stats[diff]["total"]  += 1
        stats[diff]["correct"] += int(bool(res.get("correct", False)))
        stats[diff]["tokens"]  += res.get("tokens", 0)

    if not stats:
        print("[tuner] No data to analyse.  Run the harness first.")
        return

    # -----------------------------------------------------------------------
    # Print table
    # -----------------------------------------------------------------------
    col = [10, 10, 10, 12, 14, 14, 10]
    headers = ["diff", "correct", "total", "accuracy", "avg_tokens",
               "threshold", "action"]
    sep = "  ".join("-" * w for w in col)

    def _row(vals):
        return "  ".join(str(v)[:w].ljust(w) for v, w in zip(vals, col))

    print()
    print("=" * 84)
    print("  RouteMind Threshold Tuner")
    print(f"  Accuracy floor: {accuracy_floor:.0%}  |  "
          f"Comfort band: +{comfort_band:.0%}  |  "
          f"Step size: {step:.2f}")
    print("=" * 84)
    print(_row(headers))
    print(sep)

    suggested: dict[int, float] = {}

    for diff in sorted(stats):
        d        = stats[diff]
        accuracy = d["correct"] / d["total"] if d["total"] else 0.0
        avg_tok  = d["tokens"] / d["total"]  if d["total"] else 0.0
        cur_thr  = CURRENT_THRESHOLDS.get(diff, 0.70)

        if accuracy > accuracy_floor + comfort_band:
            action       = "TIGHTEN ↑"
            new_thr      = min(round(cur_thr + step, 2), 0.95)
        elif accuracy < accuracy_floor:
            action       = "LOOSEN  ↓"
            new_thr      = max(round(cur_thr - step, 2), 0.30)
        else:
            action       = "keep"
            new_thr      = cur_thr

        suggested[diff] = new_thr

        vals = [
            diff,
            d["correct"],
            d["total"],
            f"{accuracy:.1%}",
            f"{avg_tok:.0f}",
            f"{cur_thr:.2f} → {new_thr:.2f}",
            action,
        ]
        print(_row(vals))

    print(sep)
    print()

    # -----------------------------------------------------------------------
    # Print ready-to-paste dict for router.py
    # -----------------------------------------------------------------------
    print("  Suggested CONFIDENCE_THRESHOLDS for router.py:")
    print("  (copy-paste into router.py after manual review — not auto-applied)\n")
    print("  CONFIDENCE_THRESHOLDS: dict[int, float] = {")
    for diff in sorted(suggested):
        label = {
            1: "trivial QA",
            2: "moderate QA",
            3: "reasoning / code",
            4: "multi-step",
            5: "hard / open-ended",
        }.get(diff, f"difficulty {diff}")
        print(f"      {diff}: {suggested[diff]:.2f},  # {label}")
    print("  }")
    print()
    print("=" * 84)
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyse RouteMind logs and suggest confidence threshold adjustments."
    )
    parser.add_argument(
        "--log",
        default=os.path.join("logs", "runs.jsonl"),
        help="Path to runs.jsonl (default: logs/runs.jsonl)",
    )
    parser.add_argument(
        "--harness",
        default=os.path.join("logs", "harness_results.json"),
        help="Path to harness_results.json (default: logs/harness_results.json)",
    )
    parser.add_argument(
        "--floor",
        type=float,
        default=ACCURACY_FLOOR,
        help=f"Minimum acceptable accuracy per difficulty (default: {ACCURACY_FLOOR})",
    )
    parser.add_argument(
        "--band",
        type=float,
        default=COMFORT_BAND,
        help=f"Comfort margin above floor to trigger tightening (default: {COMFORT_BAND})",
    )
    parser.add_argument(
        "--step",
        type=float,
        default=STEP,
        help=f"Amount to raise/lower threshold per pass (default: {STEP})",
    )
    args = parser.parse_args()

    # Validate inputs
    missing = []
    if not os.path.exists(args.harness):
        missing.append(f"harness results: {args.harness}")
    if not os.path.exists(args.log):
        missing.append(f"run log: {args.log}")

    if missing:
        print("[tuner] Missing required files:")
        for m in missing:
            print(f"  • {m}")
        print("\n  Run the self-eval harness first:")
        print("    python tests/test_harness.py")
        sys.exit(1)

    harness_results = _load_json(args.harness)
    run_logs        = _load_jsonl(args.log)

    print(f"[tuner] Loaded {len(harness_results)} harness results, "
          f"{len(run_logs)} run log entries.")

    analyse(
        harness_results=harness_results,
        run_logs=run_logs,
        accuracy_floor=args.floor,
        comfort_band=args.band,
        step=args.step,
    )


if __name__ == "__main__":
    main()
