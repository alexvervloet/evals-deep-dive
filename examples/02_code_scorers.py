"""
Example 02: code-based scorers (offline, no API call).

Before reaching for an LLM judge, reach for code. Code-based scorers are
deterministic, instant, and free, and a surprising amount of what you want to
check is code-checkable: exact labels, required substrings, formats, valid JSON,
numbers within tolerance.

This example runs each scorer in evals/scorers.py on a couple of crafted outputs
so you can see exactly what passes, what fails, and the `detail` each returns.

Run it:

    python examples/02_code_scorers.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import evals
from evals import Example


def show(name: str, scorer, output: str, example: Example) -> None:
    s = scorer(output, example)
    mark = "PASS" if s.passed else "FAIL"
    print(f"  [{mark}] {name:<18} score={s.score:.2f}  ({s.detail})")
    print(f"         output={output!r}")


print("exact_match: output must equal expected exactly:")
show(
    "exact_match", evals.exact_match, "positive", Example(input="", expected="positive")
)
show(
    "exact_match",
    evals.exact_match,
    "Positive!",
    Example(input="", expected="positive"),
)

print("\ncontains_expected: expected appears somewhere in the output:")
show(
    "contains_expected",
    evals.contains_expected,
    "The capital is Paris.",
    Example(input="", expected="Paris"),
)
show(
    "contains_expected",
    evals.contains_expected,
    "It is Lyon.",
    Example(input="", expected="Paris"),
)

print("\nmatches_regex: output matches a pattern (here: a 4-digit year):")
year = evals.matches_regex(r"\b\d{4}\b")
show("matches_regex", year, "Released in 1969.", Example(input=""))
show("matches_regex", year, "Released last year.", Example(input=""))

print("\nis_valid_json: output parses as JSON:")
show("is_valid_json", evals.is_valid_json, '{"ok": true}', Example(input=""))
show("is_valid_json", evals.is_valid_json, "{ok: true}", Example(input=""))

print("\njson_has_keys: JSON object contains required keys (partial credit):")
keys = evals.json_has_keys(["name", "email"])
show("json_has_keys", keys, '{"name": "Jane", "email": "j@x.com"}', Example(input=""))
show("json_has_keys", keys, '{"name": "Jane"}', Example(input=""))

print("\nnumeric_close: first number is within tolerance of expected:")
close = evals.numeric_close(tolerance=0.5)
show("numeric_close", close, "About 3.0 inches", Example(input="", expected="3.2"))
show("numeric_close", close, "About 9 inches", Example(input="", expected="3.2"))

print(
    "\nNote the partial score on json_has_keys, and that 'Positive!' fails exact "
    "match but would pass contains. Choosing the right scorer is choosing what "
    "'correct' means for your task."
)
