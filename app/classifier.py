"""
classifier.py — Heuristic task classifier.

Returns difficulty 1-5 and task type without any model call.
Difficulty is used by router.py to select the confidence threshold.
"""

import re
from typing import Literal

TaskType = Literal["qa", "reasoning", "code", "summarization", "extraction"]

# ---------------------------------------------------------------------------
# Keyword signal tables
# ---------------------------------------------------------------------------

_CODE_SIGNALS = {
    "write a function", "implement", "debug", "refactor", "code", "class",
    "algorithm", "program", "script", "def ", "return ", "variable",
    "loop", "recursion", "compile", "syntax", "lambda", "async", "await",
    "import ", "module", "package",
}

_REASONING_SIGNALS = {
    "why", "explain why", "reason", "cause", "effect", "compare",
    "difference between", "pros and cons", "evaluate", "argue",
    "step by step", "solve the following", "prove", "deduce",
    "if ... then", "given that", "therefore", "logic", "puzzle",
    "math", "calculate", "equation", "probability", "statistics",
}

_SUMMARIZATION_SIGNALS = {
    "summarize", "summary", "tldr", "in brief", "briefly describe",
    "condense", "main points", "key takeaways", "overview",
}

_EXTRACTION_SIGNALS = {
    "extract", "list all", "find all", "identify", "what are the",
    "enumerate", "names in", "dates in", "entities", "parse",
    "from the following", "from the text", "from the passage",
}

# Hard tasks: multi-step, long reasoning, adversarial
_HARD_SIGNALS = {
    "multi-step", "complex", "detailed analysis", "comprehensive",
    "in depth", "thorough", "novel", "creative writing",
    "design a system", "architect",
}


def _contains_any(text: str, signals: set[str]) -> bool:
    lower = text.lower()
    return any(sig in lower for sig in signals)


def _classify_type(task: str) -> TaskType:
    """Return the dominant task type based on keyword signals."""
    if _contains_any(task, _CODE_SIGNALS):
        return "code"
    if _contains_any(task, _SUMMARIZATION_SIGNALS):
        return "summarization"
    if _contains_any(task, _EXTRACTION_SIGNALS):
        return "extraction"
    if _contains_any(task, _REASONING_SIGNALS):
        return "reasoning"
    return "qa"


def _classify_difficulty(task: str, task_type: TaskType) -> int:
    """
    Score 1-5:
      1 — trivial factual QA, very short task
      2 — moderate QA or simple extraction
      3 — standard reasoning / summarization / code
      4 — multi-step reasoning or non-trivial code
      5 — open-ended, creative, or highly complex
    """
    score = 1
    length = len(task.split())

    # Length heuristic
    if length > 200:
        score += 2
    elif length > 80:
        score += 1

    # Type-based baseline bump
    if task_type == "code":
        score += 1
    elif task_type == "reasoning":
        score += 1

    # Hard-signal bump
    if _contains_any(task, _HARD_SIGNALS):
        score += 1

    # Presence of numbers/equations suggests harder math
    if re.search(r"\d[\d\s]*[\+\-\*/\^=]", task):
        # Distinguish simple arithmetic (single operation between two short numbers)
        # from complex math (chained operations, exponents, equations, algebra).
        is_simple_arithmetic = bool(
            re.search(r"\b\d+(?:\.\d+)?\s*[\+\-\*/]\s*\d+(?:\.\d+)?\b", task)
            and not re.search(r"[\^=]|\*\*", task)
            and not re.search(r"\b[xXyYzZ]\b", task, re.IGNORECASE)
            and len(re.findall(r"[\+\-\*/]", task)) == 1
            and len(task.split()) <= 15
        )
        if not is_simple_arithmetic:
            score = max(score, 3)

    return min(score, 5)


def classify_difficulty(task: str) -> dict:
    """
    Main entry point.

    Returns:
        {
            "difficulty": int (1-5),
            "type": "qa" | "reasoning" | "code" | "summarization" | "extraction"
        }
    """
    task_type = _classify_type(task)
    difficulty = _classify_difficulty(task, task_type)
    return {"difficulty": difficulty, "type": task_type}
