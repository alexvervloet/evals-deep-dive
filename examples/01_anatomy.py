"""
Example 01: the anatomy of an eval (offline, no API call).

Every eval, no matter how fancy, is four parts:

    dataset  ->  task  ->  scorer  ->  report

  - dataset: examples to test on (here, labelled sentiment reviews)
  - task:    the system under test, a function from input to output
  - scorer:  did the output meet the bar? (here, does it match the label?)
  - report:  the results, aggregated into numbers

The key insight this example makes concrete: **the task can be anything**. To
prove the loop needs no model at all, our "system" is a ten-line rule-based
sentiment classifier. It's offline and free, and, as you'll see, not very good,
which is exactly why we measure. Example 05 swaps in an LLM and we compare.

Run it:

    python examples/01_anatomy.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import evals

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "datasets", "sentiment.jsonl")

GOOD = {"love", "best", "fantastic", "great", "good", "perfect", "excellent", "amazing"}
BAD = {"broke", "terrible", "avoid", "bad", "dealbreaker", "buggy", "worst", "awful"}


def rule_based_sentiment(text: str) -> str:
    """A deliberately dumb baseline: count good vs bad words. This is the 'task'."""
    words = set(text.lower().replace(".", " ").replace("!", " ").split())
    good, bad = len(words & GOOD), len(words & BAD)
    if good > bad:
        return "positive"
    if bad > good:
        return "negative"
    return "neutral"


dataset = evals.load_jsonl(DATA)
print(f"Dataset: {len(dataset)} labelled examples\n")

# The whole eval, in one call: run the task over the dataset, score each output.
report = evals.run_eval(
    task=rule_based_sentiment,
    dataset=dataset,
    scorers={"exact_match": evals.exact_match},
)

report.print_summary()

# The most useful part of any report is the failures: that's where you learn.
print("\nWhere the rule-based baseline got it wrong:")
for r in report.failures("exact_match"):
    print(
        f"  predicted {r.output:<9} want {r.example.expected:<9} | {r.example.input[:55]}..."
    )

print(
    "\nA keyword baseline catches the easy cases and whiffs the subtle ones. That "
    "gap is the whole reason to measure, and the bar example 05's LLM must beat."
)
