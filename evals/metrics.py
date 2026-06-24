"""
evals/metrics.py — turn raw results into numbers you can act on.
================================================================

Scorers tell you about one output. Metrics aggregate across the whole dataset
into the numbers you actually report and compare:

  - pass_rate / accuracy — the headline "how often is it right?"
  - precision / recall / F1 — for classification, where *which kind* of error
    matters (a spam filter that flags everything has perfect recall, awful
    precision).
  - pass@k — for tasks you sample several times: did at least one of k tries pass?
  - mean_std / confidence_interval — because LLM outputs vary run to run, a single
    number is a point estimate; you want the spread.
  - compare — the question that matters in practice: is run B *really* better than
    run A, or is the gap just noise?

All pure math, no API calls — this module is offline.
"""

import math
from statistics import mean, stdev


def pass_rate(scores) -> float:
    """Fraction that passed. Accepts a list of Score objects or of bools."""
    bools = [s.passed if hasattr(s, "passed") else bool(s) for s in scores]
    return sum(bools) / len(bools) if bools else 0.0


def accuracy(predicted: list, expected: list) -> float:
    """Fraction of predictions exactly equal to the expected label."""
    pairs = list(zip(predicted, expected))
    return sum(p == e for p, e in pairs) / len(pairs) if pairs else 0.0


def precision_recall_f1(predicted: list, expected: list, positive_label) -> dict:
    """Precision, recall, and F1 for one label (one-vs-rest).

    precision = of the ones we *called* positive, how many were? (penalizes false
    alarms). recall = of the *actually* positive ones, how many did we catch?
    (penalizes misses). F1 is their harmonic mean — one number balancing both.
    """
    tp = sum(p == positive_label and e == positive_label for p, e in zip(predicted, expected))
    fp = sum(p == positive_label and e != positive_label for p, e in zip(predicted, expected))
    fn = sum(p != positive_label and e == positive_label for p, e in zip(predicted, expected))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def pass_at_k(attempts_per_example: list[list[bool]], k: int) -> float:
    """pass@k: fraction of examples where at least one of the first k attempts
    passed. The metric for "give the model a few tries" tasks (e.g. code gen).

    `attempts_per_example[i]` is the list of pass/fail booleans for example i.
    (This is the simple empirical estimator; the rigorous unbiased version from
    the Codex paper corrects for the number of samples — a refinement, not a
    different idea.)"""
    if not attempts_per_example:
        return 0.0
    solved = sum(any(attempts[:k]) for attempts in attempts_per_example)
    return solved / len(attempts_per_example)


def mean_std(values: list[float]) -> tuple[float, float]:
    """Mean and (sample) standard deviation. std is 0 for a single value."""
    if not values:
        return 0.0, 0.0
    m = mean(values)
    s = stdev(values) if len(values) > 1 else 0.0
    return m, s


def confidence_interval(values: list[float], z: float = 1.96) -> tuple[float, float]:
    """An approximate confidence interval for the mean (default ~95%, z=1.96).

    interval = mean ± z * (std / sqrt(n)). Wider with fewer samples or more
    spread — which is exactly why running an eval once and trusting the number is
    risky. (Normal approximation; fine for teaching, use a t-distribution or
    bootstrap for small-n rigor.)"""
    if len(values) < 2:
        m = values[0] if values else 0.0
        return m, m
    m, s = mean_std(values)
    half = z * s / math.sqrt(len(values))
    return m - half, m + half


def compare(a_values: list[float], b_values: list[float], z: float = 1.96) -> dict:
    """Is B different from A, or is the gap just noise?

    Returns the means, the difference (B − A), the margin of error on that
    difference, and `likely_real`: True when the difference is larger than the
    margin (its confidence interval excludes zero). This is the honest way to
    decide whether a prompt change helped, instead of cheering a +2% that's within
    the noise."""
    ma, sa = mean_std(a_values)
    mb, sb = mean_std(b_values)
    na, nb = len(a_values), len(b_values)
    se = math.sqrt((sa**2 / na if na else 0) + (sb**2 / nb if nb else 0))
    diff = mb - ma
    margin = z * se
    return {
        "mean_a": ma,
        "mean_b": mb,
        "diff": diff,
        "margin": margin,
        "likely_real": abs(diff) > margin and margin > 0,
    }
