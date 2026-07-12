"""
stats.py — Routing telemetry and savings statistics (Tier 1).

Reads logs/runs.jsonl and computes:
  • total requests
  • total tokens actually spent
  • total tokens that WOULD have been spent if every non-cache request
    went straight to Fireworks large-tier
  • percentage of requests served locally vs cached vs escalated
  • tokens saved by caching specifically
"""

import json
import os
from typing import Any

# Path is relative to project root /app/logs/runs.jsonl
_LOG_DIR  = os.path.join(os.path.dirname(__file__), "..", "logs")
_LOG_FILE = os.path.join(_LOG_DIR, "runs.jsonl")

# Reasonable fixed estimate if no Fireworks calls exist in logs yet
DEFAULT_ESTIMATED_TOKENS_PER_REQUEST = 350.0


def get_stats(log_file: str | None = None) -> dict[str, Any]:
    """
    Compute routing statistics from the JSONL log file.

    Returns a dict containing total requests, tokens spent, estimated savings,
    and route percentages.
    """
    if log_file is None:
        log_file = _LOG_FILE

    if not os.path.exists(log_file):
        return {
            "total_requests": 0,
            "total_tokens_spent": 0,
            "estimated_tokens_without_routing": 0,
            "estimated_tokens_if_all_large_tier": 0,
            "tokens_saved_by_caching": 0,
            "local_requests": 0,
            "cached_requests": 0,
            "escalated_requests": 0,
            "gemma_calls": 0,
            "local_percentage": 0.0,
            "cached_percentage": 0.0,
            "escalated_percentage": 0.0,
            "gemma_call_percentage": 0.0,
            "counts": {
                "local": 0,
                "cached": 0,
                "escalated": 0,
                "gemma": 0,
            },
            "percentages": {
                "local": 0.0,
                "cached": 0.0,
                "escalated": 0.0,
                "gemma": 0.0,
            },
        }

    total_requests = 0
    total_tokens_spent = 0
    local_count = 0
    cache_count = 0
    escalated_count = 0
    gemma_count = 0

    fw_call_count = 0
    fw_token_sum = 0

    with open(log_file, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            total_requests += 1
            tokens = int(record.get("tokens", 0))
            total_tokens_spent += tokens
            route = str(record.get("route", ""))

            if route == "cache":
                cache_count += 1
            elif route == "local":
                local_count += 1
            else:
                # Escalated to Fireworks (fireworks-small, fireworks-gemma, fireworks-large, etc.)
                escalated_count += 1
                fw_call_count += 1
                fw_token_sum += tokens

            if bool(record.get("used_gemma")):
                gemma_count += 1

    # Calculate average tokens per Fireworks request
    if fw_call_count > 0:
        avg_fw_tokens = float(fw_token_sum) / float(fw_call_count)
    else:
        avg_fw_tokens = DEFAULT_ESTIMATED_TOKENS_PER_REQUEST

    non_cache_count = local_count + escalated_count
    est_tokens_if_all_large_tier = int(round(non_cache_count * avg_fw_tokens))

    # Calculate average token cost avoided per cache hit
    if non_cache_count > 0:
        avg_cost_avoided = float(total_tokens_spent) / float(non_cache_count)
    else:
        avg_cost_avoided = 0.0

    tokens_saved_by_caching = int(round(cache_count * avg_cost_avoided))

    if total_requests > 0:
        local_pct = round((local_count / total_requests) * 100.0, 2)
        cached_pct = round((cache_count / total_requests) * 100.0, 2)
        escalated_pct = round((escalated_count / total_requests) * 100.0, 2)
        gemma_pct = round((gemma_count / total_requests) * 100.0, 2)
    else:
        local_pct = cached_pct = escalated_pct = gemma_pct = 0.0

    return {
        "total_requests": total_requests,
        "total_tokens_spent": total_tokens_spent,
        "estimated_tokens_without_routing": est_tokens_if_all_large_tier,
        "estimated_tokens_if_all_large_tier": est_tokens_if_all_large_tier,
        "tokens_saved_by_caching": tokens_saved_by_caching,
        "local_requests": local_count,
        "cached_requests": cache_count,
        "escalated_requests": escalated_count,
        "gemma_calls": gemma_count,
        "local_percentage": local_pct,
        "cached_percentage": cached_pct,
        "escalated_percentage": escalated_pct,
        "gemma_call_percentage": gemma_pct,
        "counts": {
            "local": local_count,
            "cached": cache_count,
            "escalated": escalated_count,
            "gemma": gemma_count,
        },
        "percentages": {
            "local": local_pct,
            "cached": cached_pct,
            "escalated": escalated_pct,
            "gemma": gemma_pct,
        },
    }
