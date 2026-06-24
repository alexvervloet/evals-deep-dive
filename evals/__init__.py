"""
evals — a small, from-scratch library for evaluating LLM apps.

Everything here is built to be *read*, not just used. The pieces map to the four
parts of any eval (dataset -> task -> scorer -> report):

  dataset.py    — examples to test against (a "golden set")
  scorers.py    — code-based scorers: did the output meet a checkable bar?
  judges.py     — model-graded scorers: LLM-as-judge (pointwise + pairwise)
  runner.py     — the loop: run a task over a dataset, score it, build a Report
  metrics.py    — turn raw results into numbers (accuracy, F1, pass@k, CIs)
  providers.py  — the ONLY provider-specific file: generate()

Import what you need, e.g.:

    from evals import run_eval, exact_match
"""

from .dataset import Example, load_jsonl
from .judges import judge_pairwise, judge_pointwise
from .metrics import (
    accuracy,
    compare,
    confidence_interval,
    mean_std,
    pass_at_k,
    pass_rate,
    precision_recall_f1,
)
from .providers import describe, ensure_ready, generate, provider_name
from .runner import Report, Result, run_eval
from .scorers import (
    Score,
    contains_expected,
    exact_match,
    is_valid_json,
    json_has_keys,
    matches_regex,
    numeric_close,
)

__all__ = [
    "Example",
    "load_jsonl",
    "Score",
    "exact_match",
    "contains_expected",
    "matches_regex",
    "is_valid_json",
    "json_has_keys",
    "numeric_close",
    "judge_pointwise",
    "judge_pairwise",
    "run_eval",
    "Result",
    "Report",
    "accuracy",
    "pass_rate",
    "precision_recall_f1",
    "pass_at_k",
    "mean_std",
    "confidence_interval",
    "compare",
    "generate",
    "provider_name",
    "describe",
    "ensure_ready",
]
