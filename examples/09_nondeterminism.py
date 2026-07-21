"""
Example 09: nondeterminism: one run is a point estimate.

LLM outputs vary between runs (especially above temperature 0). So a single eval
number is a sample, not the truth. Re-run the same eval and it moves. This
example makes that visible and then does the honest thing with it.

  1. Run the same classifier eval several times at temperature 0.9 and watch the
     pass rate wobble. Report the mean with a confidence interval, not one number.
  2. Compare two prompts across several runs each, and use `compare()` to decide
     whether the difference is REAL or just noise.

This is the most expensive example (many calls). It uses all 10 rows so the hard
(sarcasm/mixed-signal) rows produce genuine variance; turn `N` down to spend less,
or up for tighter confidence intervals.

Run it:

    secrun python examples/09_nondeterminism.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import evals
from dotenv import load_dotenv

load_dotenv()
evals.ensure_ready()
print(f"Provider: {evals.describe()}\n")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dataset = evals.load_jsonl(os.path.join(ROOT, "datasets", "sentiment.jsonl"))

ROWS = dataset  # all 10 rows; hard rows at the end are genuinely ambiguous (sarcasm, mixed signals)
N = 4  # runs per prompt
TEMP = 0.9  # higher temp amplifies variance on uncertain rows

LABELS = ("positive", "negative", "neutral")


def to_label(text: str) -> str:
    t = text.strip().lower()
    return next((lbl for lbl in LABELS if lbl in t), t)


def run_once(system: str) -> float:
    """One full eval pass; returns the pass rate."""

    def task(text: str) -> str:
        return to_label(evals.generate(system, text, temperature=TEMP))

    report = evals.run_eval(task, ROWS, {"exact": evals.exact_match})
    return report.pass_rate("exact")


PROMPT_A = "You are a sentiment classifier. Reply with one word: positive, negative, or neutral."
PROMPT_B = (
    "You are a careful sentiment classifier. Watch for sarcasm and mixed signals. "
    "Reply with one word: positive, negative, or neutral."
)

print(f"Running each prompt {N}x over {len(ROWS)} rows at temperature {TEMP}...\n")

a_runs = [run_once(PROMPT_A) for _ in range(N)]
b_runs = [run_once(PROMPT_B) for _ in range(N)]

a_lo, a_hi = evals.confidence_interval(a_runs)
b_lo, b_hi = evals.confidence_interval(b_runs)
print(f"Prompt A pass rates: {[f'{x:.0%}' for x in a_runs]}")
print(f"  mean {sum(a_runs) / N:.0%}, 95% CI [{a_lo:.0%}, {a_hi:.0%}]")
print(f"Prompt B pass rates: {[f'{x:.0%}' for x in b_runs]}")
print(f"  mean {sum(b_runs) / N:.0%}, 95% CI [{b_lo:.0%}, {b_hi:.0%}]\n")

cmp = evals.compare(a_runs, b_runs)
verdict = (
    "a REAL difference" if cmp["likely_real"] else "within the noise (inconclusive)"
)
print(f"B − A = {cmp['diff']:+.0%} ± {cmp['margin']:.0%}  ->  {verdict}")

print(
    "\nIf you'd run each prompt once, you might have 'seen' a winner that's pure "
    "sampling noise. With small datasets and few runs the interval is wide, which "
    "is the honest message: to claim an improvement, run enough to separate it "
    "from the wobble."
)
