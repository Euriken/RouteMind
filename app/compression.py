"""
compression.py — Lightweight prompt compression before Fireworks escalation.

Two-stage approach:
  1. Deterministic cleaning: strip redundant whitespace, common boilerplate
     phrases that add no semantic content (e.g. "Please", "Could you please").
  2. If the cleaned task still exceeds max_tokens words, call the local Ollama
     model to summarize it, explicitly asking it to preserve all technical detail.

Returns the original (cleaned) task unchanged if it is already short enough.
"""

import re
import os


# ---------------------------------------------------------------------------
# Boilerplate phrases to strip (case-insensitive prefix / infix)
# Extend this list as you see common padding in real hackathon tasks.
# ---------------------------------------------------------------------------
_BOILERPLATE = [
    r"please\s+",
    r"could you please\s+",
    r"could you\s+",
    r"can you please\s+",
    r"can you\s+",
    r"i want you to\s+",
    r"i need you to\s+",
    r"i would like you to\s+",
    r"kindly\s+",
    r"as an? (?:ai|language model|assistant)[,.]?\s*",
    r"note[:\s]+this is for a hackathon[^\n]*\n?",
]

_BOILERPLATE_RE = re.compile(
    "|".join(_BOILERPLATE),
    flags=re.IGNORECASE,
)


def _clean(text: str) -> str:
    """Strip boilerplate phrases and normalise whitespace."""
    text = _BOILERPLATE_RE.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text)   # collapse 3+ blank lines
    text = re.sub(r"[ \t]+", " ", text)       # collapse inline spaces
    return text.strip()


def _word_count(text: str) -> int:
    return len(text.split())


def _call_ollama_summary(text: str) -> str:
    """Ask the local model to summarize text, preserving technical details."""
    import requests  # local import to keep module lightweight

    host  = os.getenv("OLLAMA_HOST", "http://ollama:11434")
    model = os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b")

    prompt = (
        "Summarize the following task as concisely as possible. "
        "Preserve every technical detail, constraint, and requirement. "
        "Do not add any explanation or preamble — output only the summarized task.\n\n"
        f"TASK:\n{text}"
    )

    resp = requests.post(
        f"{host}/api/generate",
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json().get("response", text).strip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compress(task: str, max_tokens: int = 200) -> str:
    """
    Compress a task string before sending it to Fireworks.

    Steps:
      1. Remove boilerplate phrases and normalise whitespace.
      2. If the result exceeds max_tokens words, call the local model for
         a concise summarization that preserves all technical details.
      3. Return cleaned (or summarized) text.

    Args:
        task:       Raw task string from the user.
        max_tokens: Word-count ceiling before triggering local summarization.
                    Default 200 words ≈ ~270 tokens at average tokenization.

    Returns:
        Compressed task string (never longer than the input).
    """
    cleaned = _clean(task)

    if _word_count(cleaned) <= max_tokens:
        return cleaned

    # Long task — summarize locally before billing Fireworks tokens
    try:
        summarized = _call_ollama_summary(cleaned)
        # Safety: don't return something longer than the cleaned version
        if _word_count(summarized) < _word_count(cleaned):
            return summarized
    except Exception:
        pass  # If local model fails, fall back to the cleaned text

    return cleaned
