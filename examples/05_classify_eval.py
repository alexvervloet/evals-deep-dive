"""
Example 05 — evaluating an LLM classifier.
==========================================

Now the system under test is a real LLM. Same loop as example 01 — only the task
changed from a rule to a model call. We classify the sentiment dataset and report
accuracy plus per-class precision/recall/F1, so you can see not just *how often*
it's right but *which kind* of mistakes it makes.

Compare the accuracy here to the rule-based baseline from example 01 on the same
data: that's the entire point of an eval — a number that says whether one system
beats another.

Run it:

    python examples/05_classify_eval.py
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

SYSTEM = (
    "You are a sentiment classifier. Classify the user's text as exactly one of: "
    "positive, negative, neutral. Reply with only that single word."
)
LABELS = ("positive", "negative", "neutral")


def to_label(text: str) -> str:
    """LLMs sometimes add punctuation or extra words — pin the reply to a label."""
    t = text.strip().lower()
    for label in LABELS:
        if label in t:
            return label
    return t  # unrecognized — counts as wrong, which is the honest outcome


def classify(text: str) -> str:
    return to_label(evals.generate(SYSTEM, text, temperature=0.0))


report = evals.run_eval(classify, dataset, {"exact_match": evals.exact_match})

predicted = [r.output for r in report.results]
expected = [r.example.expected for r in report.results]

print(
    f"Accuracy: {evals.accuracy(predicted, expected):.0%}  ({len(dataset)} examples)\n"
)
print("Per-class precision / recall / F1:")
for label in LABELS:
    prf = evals.precision_recall_f1(predicted, expected, positive_label=label)
    print(
        f"  {label:<9} P={prf['precision']:.2f}  R={prf['recall']:.2f}  F1={prf['f1']:.2f}"
    )

print("\nMisclassified:")
for r in report.failures("exact_match"):
    print(
        f"  predicted {r.output:<9} want {r.example.expected:<9} | {r.example.input[:50]}..."
    )

print(
    "\nThe 'hard' rows (sarcasm, mixed signals) are where errors cluster — and "
    "where your labels are themselves debatable. That ambiguity is real; a good "
    "eval surfaces it instead of hiding it behind one number."
)
