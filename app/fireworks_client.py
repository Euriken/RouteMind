"""
fireworks_client.py — REST wrapper for the Fireworks AI inference API.

Day 4 additions (SPEC.md §4 Tier 2 item 10):
  • MODEL_TIERS — ordered escalation ladder: small → large
  • solve_with_escalation() — tries the cheapest tier first, escalates only if
    the response shows low-confidence signals (short / empty / error-like).

Note: the originally-planned Gemma middle rung was removed after Fireworks
deprecated serverless inference for gemma2-9b-it (returns 404). The ladder
is now two tiers: gpt-oss-20b → gpt-oss-120b.

Reads FIREWORKS_API_KEY from the environment.
Retries up to MAX_RETRIES times with exponential backoff on transient failures.
"""

import os
import re
import time
import requests

# ---------------------------------------------------------------------------
# Model tier → Fireworks model ID mapping
#
# Ordered from cheapest/smallest to most capable/expensive.
# The list order defines the escalation sequence used by solve_with_escalation.
#
# Note: gemma2-9b-it was removed — serverless inference is no longer available
# for that model on this account (returns 404). Two tiers remain.
# ---------------------------------------------------------------------------
MODEL_TIERS: dict[str, str] = {
    "small":  "accounts/fireworks/models/gpt-oss-20b",
    "large":  "accounts/fireworks/models/gpt-oss-120b",
}

# Escalation order — cheapest first (two tiers: small → large).
ESCALATION_ORDER: list[str] = ["small", "large"]

FIREWORKS_API_URL = "https://api.fireworks.ai/inference/v1/chat/completions"
MAX_RETRIES = 3
BACKOFF_BASE = 1.5  # seconds; multiplied by 2^attempt each retry

# Minimum token count below which we consider an answer "too short / low-confidence".
_MIN_ANSWER_TOKENS = 10

# Patterns that indicate a model couldn't answer properly.
_FAILURE_PATTERNS: list[re.Pattern] = [
    re.compile(r"^\s*$"),                          # empty / whitespace only
    re.compile(r"i (don'?t|cannot|can'?t) know", re.I),
    re.compile(r"(i'?m|i am) (not sure|unable|sorry)", re.I),
    re.compile(r"^\s*(sorry|apologies)[,.]?\s*$", re.I),
    re.compile(r"error|exception|traceback", re.I),
]


def _is_low_confidence(text: str) -> bool:
    """
    Return True if the response looks uncertain or empty — a signal to escalate.

    Heuristics:
      • fewer than _MIN_ANSWER_TOKENS whitespace-separated tokens
      • matches one of the _FAILURE_PATTERNS
    """
    if len(text.split()) < _MIN_ANSWER_TOKENS:
        return True
    return any(p.search(text) for p in _FAILURE_PATTERNS)


class FireworksClient:
    def __init__(self) -> None:
        self.api_key = os.getenv("FIREWORKS_API_KEY", "")
        if not self.api_key:
            raise EnvironmentError(
                "FIREWORKS_API_KEY is not set. Add it to your .env file."
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _post_with_retry(self, payload: dict) -> dict:
        """POST to Fireworks with up to MAX_RETRIES on transient errors."""
        last_exc: Exception | None = None

        for attempt in range(MAX_RETRIES):
            try:
                resp = requests.post(
                    FIREWORKS_API_URL,
                    headers=self._headers(),
                    json=payload,
                    timeout=60,
                )

                if resp.status_code >= 500:
                    raise requests.HTTPError(
                        f"Fireworks returned {resp.status_code}", response=resp
                    )

                resp.raise_for_status()
                return resp.json()

            except (requests.exceptions.Timeout, requests.HTTPError) as exc:
                last_exc = exc
                time.sleep(BACKOFF_BASE * (2 ** attempt))

            except requests.exceptions.RequestException as exc:
                last_exc = exc
                time.sleep(BACKOFF_BASE * (2 ** attempt))

        raise RuntimeError(
            f"Fireworks request failed after {MAX_RETRIES} attempts: {last_exc}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve(
        self,
        prompt: str,
        model: str = "small",
        max_tokens: int = 512,
    ) -> tuple[str, int]:
        """
        Call Fireworks AI and return (answer_text, tokens_used).

        Args:
            prompt:     The full task prompt to send.
            model:      Key from MODEL_TIERS ("small", "large") or a
                        raw Fireworks model ID string.
            max_tokens: Upper bound on generated tokens.

        Returns:
            (answer_text, total_tokens_used)
        """
        model_id = MODEL_TIERS.get(model, model)

        payload = {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.2,
        }

        data = self._post_with_retry(payload)

        answer = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
        )

        tokens_used = data.get("usage", {}).get("total_tokens", 0)
        return (answer, tokens_used)

    def solve_with_escalation(
        self,
        prompt: str,
        max_tokens: int = 512,
    ) -> tuple[str, int, str]:
        """
        Try each model tier in ESCALATION_ORDER (smallest/cheapest first).
        Escalate to the next tier only if the current tier's answer fails the
        low-confidence check (_is_low_confidence).

        Returns:
            (answer, total_tokens_used, final_tier_used)
            where final_tier_used is a key from MODEL_TIERS (e.g. "large").

        Token accounting: accumulates tokens across all tiers tried, so the
        caller always sees the true cost of the escalation chain.
        """
        total_tokens = 0
        last_answer = ""
        last_tier = ESCALATION_ORDER[0]

        for tier in ESCALATION_ORDER:
            last_tier = tier
            answer, tokens = self.solve(prompt, model=tier, max_tokens=max_tokens)
            total_tokens += tokens
            last_answer = answer

            if not _is_low_confidence(answer):
                # Good enough — stop escalating.
                return (answer, total_tokens, tier)

            # Low confidence — try next tier (loop continues).

        # Exhausted all tiers; return whatever the last one produced.
        return (last_answer, total_tokens, last_tier)
