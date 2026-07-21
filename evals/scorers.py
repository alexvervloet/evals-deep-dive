"""
evals/scorers.py: code-based scorers.

A *scorer* takes the system's output (and the Example it came from) and returns a
`Score`: did it meet the bar, as a 0–1 number plus a human-readable detail.

The scorers here are **code-based**: plain Python, deterministic, free, and
offline. Reach for these first; they're cheaper and more reliable than asking a
model to grade. Only when a question genuinely needs judgement ("is this summary
faithful?") do you escalate to an LLM judge (see judges.py).

A scorer is just `Callable[[str, Example], Score]`. Some take no configuration
(`exact_match`); others are *factories* that return a configured scorer
(`matches_regex(r"...")`, `numeric_close(0.5)`), a small, useful Python pattern.
"""

import json
import re
from dataclasses import dataclass

from .dataset import Example


@dataclass
class Score:
    """The result of one scorer on one output. `score` is 0–1; `passed` is the
    pass/fail view of it; `detail` explains why (great for debugging failures)."""

    passed: bool
    score: float
    detail: str = ""


def exact_match(output: str, example: Example) -> Score:
    """Output equals the expected answer exactly (after trimming whitespace).
    The strictest scorer, perfect for labels and classification."""
    want = (example.expected or "").strip()
    got = output.strip()
    ok = got == want
    return Score(ok, 1.0 if ok else 0.0, f"{got!r} {'==' if ok else '!='} {want!r}")


def contains_expected(output: str, example: Example) -> Score:
    """The expected text appears somewhere in the output (case-insensitive).
    Looser than exact match, good when the answer is embedded in prose."""
    want = (example.expected or "").lower()
    ok = want in output.lower()
    return Score(ok, 1.0 if ok else 0.0, f"expected {'found' if ok else 'MISSING'}: {want!r}")


def matches_regex(pattern: str):
    """Factory: returns a scorer that passes if `pattern` is found in the output.
    Use for format checks ('starts with a verb', 'contains a date')."""
    rx = re.compile(pattern)

    def scorer(output: str, example: Example) -> Score:
        ok = bool(rx.search(output))
        return Score(ok, 1.0 if ok else 0.0, f"/{pattern}/ {'matched' if ok else 'no match'}")

    return scorer


def is_valid_json(output: str, example: Example) -> Score:
    """Output parses as JSON. The cheapest, most common structured-output check."""
    try:
        json.loads(output)
        return Score(True, 1.0, "valid JSON")
    except (ValueError, TypeError) as e:
        return Score(False, 0.0, f"invalid JSON: {e}")


def json_has_keys(keys: list[str]):
    """Factory: output is a JSON object containing all of `keys`. Partial credit
    is the fraction of keys present, which makes 'almost right' visible."""

    def scorer(output: str, example: Example) -> Score:
        try:
            data = json.loads(output)
        except (ValueError, TypeError) as e:
            return Score(False, 0.0, f"invalid JSON: {e}")
        if not isinstance(data, dict):
            return Score(False, 0.0, "JSON is not an object")
        missing = [k for k in keys if k not in data]
        ok = not missing
        score = (len(keys) - len(missing)) / len(keys) if keys else 1.0
        return Score(ok, score, "all keys present" if ok else f"missing {missing}")

    return scorer


def numeric_close(tolerance: float = 0.0):
    """Factory: the first number in the output is within `tolerance` of the
    expected number. For math/extraction where exact string match is too strict
    ('42' vs '42.0' vs 'The answer is 42')."""

    def scorer(output: str, example: Example) -> Score:
        got = _first_number(output)
        try:
            want = float(example.expected)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return Score(False, 0.0, "example has no numeric expected value")
        if got is None:
            return Score(False, 0.0, "no number found in output")
        ok = abs(got - want) <= tolerance
        return Score(ok, 1.0 if ok else 0.0, f"{got} vs {want} (tol {tolerance})")

    return scorer


def _first_number(text: str) -> float | None:
    m = re.search(r"-?\d+(?:\.\d+)?", text)
    return float(m.group()) if m else None
