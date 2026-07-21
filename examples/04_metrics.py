"""
Example 04: metrics: from pass/fail to numbers that decide (offline math).

A pile of scored outputs isn't actionable until you aggregate it. This example
works through the metrics in evals/metrics.py on small hand-made inputs so the
math is visible, with no API calls.

What each answers:
  - accuracy:           how often exactly right? (the headline)
  - precision/recall/F1: for classification, *which kind* of mistake? (false
                         alarms vs misses)
  - pass@k:             with several tries, did at least one pass?
  - confidence interval: how much would this number wobble on a re-run?
  - compare:            is run B really better than A, or is it noise?

Run it:

    python examples/04_metrics.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import evals

# A tiny classification result: what a model predicted vs the truth.
expected = ["spam", "ham", "spam", "ham", "ham", "spam"]
predicted = ["spam", "ham", "ham", "ham", "spam", "spam"]  # 2 mistakes

print("Classification of 6 emails (positive label = 'spam'):")
print(f"  expected:  {expected}")
print(f"  predicted: {predicted}\n")

print(f"  accuracy: {evals.accuracy(predicted, expected):.0%}")
prf = evals.precision_recall_f1(predicted, expected, positive_label="spam")
print(
    f"  precision: {prf['precision']:.2f}  (of those we flagged spam, how many were?)"
)
print(f"  recall:    {prf['recall']:.2f}  (of real spam, how much we caught)")
print(f"  f1:        {prf['f1']:.2f}  (the balance of the two)")
print(f"  tp={prf['tp']} fp={prf['fp']} fn={prf['fn']}")

# pass@k: each row is the pass/fail of several attempts at one problem.
attempts = [
    [False, True, False],  # solved on the 2nd try
    [False, False, False],  # never solved
    [True, False, True],  # solved on the 1st try
]
print("\npass@k on 3 problems with 3 attempts each:")
print(
    f"  pass@1 = {evals.pass_at_k(attempts, 1):.0%}   pass@3 = {evals.pass_at_k(attempts, 3):.0%}"
)

# Variance: the same eval re-run gives slightly different scores.
runs = [0.82, 0.79, 0.85, 0.80, 0.83]
lo, hi = evals.confidence_interval(runs)
print(f"\nFive re-runs of an eval: {runs}")
print(f"  mean ≈ {sum(runs) / len(runs):.3f}, 95% CI ≈ [{lo:.3f}, {hi:.3f}]")

# Comparing two systems honestly.
baseline = [0.80, 0.78, 0.81, 0.79]
candidate = [0.83, 0.82, 0.85, 0.84]
cmp = evals.compare(baseline, candidate)
verdict = "a REAL improvement" if cmp["likely_real"] else "within the noise"
print(
    f"\nCandidate vs baseline: {cmp['mean_b']:.3f} vs {cmp['mean_a']:.3f} "
    f"(diff {cmp['diff']:+.3f} ± {cmp['margin']:.3f}) -> {verdict}"
)

print(
    "\nThat last line is the discipline: don't ship a '+2%' that's inside the "
    "margin of error. A difference you can't distinguish from noise isn't a win."
)
