"""
Example 12 — online evals: A/B testing on live traffic. (offline)
=================================================================

Everything so far was an *offline* eval: a fixed dataset you score before shipping.
But a passing offline score doesn't guarantee real users are better off. The
complement is the **online eval** — measure a change on *live traffic*, by splitting
users into variant A (control) and variant B (the change) and comparing an outcome
metric you actually care about: thumbs-up rate, task completion, retention.

The discipline is the same one from example 09 (nondeterminism) and the `compare()`
significance test from example 05's metrics: **a difference is only real if it's
bigger than the noise.** Ship B over A only when the gap clears the margin of error
— and only if your *guardrail* metrics (latency, refusal rate, cost) didn't regress.

This script simulates outcomes for two variants, runs the significance test, shows a
guardrail check, and demonstrates why **sample size** decides whether you can
conclude anything. Pure arithmetic — offline and free. (In production the outcome
booleans come from real user signals, not a simulation.)

Run it:

    python examples/12_online_eval.py
"""

import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evals.metrics import compare


def simulate(true_rate: float, n: int, seed: int) -> list[float]:
    """One outcome per request: 1.0 = user was satisfied, 0.0 = not.

    Stands in for live signal (a thumbs-up, a completed task). The *true* rate is
    unknown in reality — that's what the A/B test is trying to detect.
    """
    rng = random.Random(seed)
    return [1.0 if rng.random() < true_rate else 0.0 for _ in range(n)]


def report_ab(label: str, a: list[float], b: list[float]) -> None:
    c = compare(a, b)
    verdict = "SHIP B" if c["likely_real"] and c["diff"] > 0 else "keep A (gap is within the noise)"
    print(f"[{label}]  n={len(a)} per arm")
    print(f"  A satisfaction: {c['mean_a']:.1%}    B satisfaction: {c['mean_b']:.1%}")
    print(f"  difference (B-A): {c['diff']:+.1%}  ± {c['margin']:.1%} margin of error")
    print(f"  -> {verdict}\n")


if __name__ == "__main__":
    print("A/B test: variant B's prompt truly satisfies 68% of users vs A's 60%.\n")

    # 1. Too few samples: the real +8% gap is buried in noise -> can't conclude.
    report_ab("small sample", simulate(0.60, 80, 1), simulate(0.68, 80, 2))

    # 2. Enough samples: the same +8% gap now clears the margin -> ship it.
    report_ab("large sample", simulate(0.60, 1500, 1), simulate(0.68, 1500, 2))

    # 3. A real win on the headline metric, but a GUARDRAIL regressed.
    print("Guardrail check (a metric that must NOT get worse):")
    refusal_a = sum(simulate(0.02, 1500, 3)) / 1500   # A refuses 2% of the time
    refusal_b = sum(simulate(0.09, 1500, 4)) / 1500   # B refuses 9% — worse!
    print(f"  refusal rate  A={refusal_a:.1%}  B={refusal_b:.1%}")
    print("  -> B wins on satisfaction but refuses far more often. Do NOT ship — a\n"
          "     headline win that regresses a guardrail is a regression.\n")

    print(
        "Takeaway: online evals measure the change on real traffic, where it counts.\n"
        "Two rules carry over from offline: judge a difference against its margin of\n"
        "error (small samples can't prove a real gap), and never ship a headline win\n"
        "that quietly regresses a guardrail (latency, refusals, cost)."
    )
