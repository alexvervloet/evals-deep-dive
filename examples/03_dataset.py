"""
Example 03: the dataset is the hard part (offline, no API call).

Clever scorers and metrics get the attention, but the quality of an eval is
capped by the quality of its dataset. This example loads the three golden sets in
datasets/ and looks at what's in them, and what *kind* of eval each enables.

The distinction to internalize:

  - reference-based (has `expected`): you can score against a known answer 
    exact match, F1, numeric tolerance. Precise, but you had to label it.
  - reference-free (no `expected`): you judge the output on its own merits 
    valid JSON? an LLM judge's rating? No labels needed, fuzzier signal.

Run it:

    python examples/03_dataset.py
"""

import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import evals

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASETS = os.path.join(ROOT, "datasets")

for name in ["sentiment.jsonl", "qa.jsonl", "extraction.jsonl"]:
    data = evals.load_jsonl(os.path.join(DATASETS, name))
    has_expected = sum(1 for e in data if e.expected is not None)
    kind = "reference-based" if has_expected == len(data) else "mixed/reference-free"
    print(f"{name}: {len(data)} examples ({kind})")
    sample = data[0]
    print(f"  input:    {sample.input[:60]}...")
    print(f"  expected: {sample.expected!r}")
    if sample.metadata:
        print(f"  metadata: {sample.metadata}")
    print()

# Metadata lets you slice results: e.g. "is accuracy worse on hard examples?"
sentiment = evals.load_jsonl(os.path.join(DATASETS, "sentiment.jsonl"))
by_difficulty = Counter(e.metadata.get("difficulty", "?") for e in sentiment)
print(f"sentiment.jsonl by difficulty: {dict(by_difficulty)}")

print(
    "\nThese sets are tiny on purpose. Ten hand-checked, representative examples "
    "beat a thousand sloppy ones, and the best examples come from real failures "
    "you find later and add back here so they never regress."
)
