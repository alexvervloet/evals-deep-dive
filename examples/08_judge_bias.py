"""
Example 08 — your judge is biased (and how to mitigate it).
===========================================================

An LLM judge is a model, so it has biases. The most notorious is **position
bias**: a tendency to prefer whichever answer is shown *first* (others include
favouring longer answers, and a model preferring its own style). If you trust a
single pairwise verdict, position bias can hand you a "winner" that's really just
"whoever went first."

The test: judge each pair in BOTH orders. If the same underlying answer wins
regardless of position, the verdict is consistent. If the winner flips when you
swap the order, that pair was decided by position, not quality.

The fix is the same as the test: **run both orders and only count a win if the
answer wins both ways** (otherwise call it a tie). This example uses fixed answer
pairs (no generation) so the judge's behaviour is the only variable.

Run it:

    secrun python examples/08_judge_bias.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import evals
from dotenv import load_dotenv

load_dotenv()
evals.ensure_ready()
print(f"Provider: {evals.describe()}\n")

# (question, answer_1, answer_2). We'll always treat answer_1 as the "concise" one
# and answer_2 as the "verbose" one, then see if position changes the verdict.
PAIRS = [
    (
        "What is the capital of France?",
        "Paris.",
        "The capital of France is Paris, which sits on the river Seine.",
    ),
    (
        "How many continents are there?",
        "Seven.",
        "There are seven continents: Africa, Antarctica, Asia, Europe, North America, "
        "Oceania, and South America.",
    ),
    (
        "What is the chemical symbol for gold?",
        "Au.",
        "Gold's chemical symbol is Au, from the Latin word 'aurum'.",
    ),
    (
        "What is the square root of 144?",
        "12.",
        "The square root of 144 is 12, since 12 times 12 equals 144.",
    ),
]

RUBRIC = "which answer is more correct and helpful"
consistent = 0
flipped = 0

print("Judging each pair in both orders:")
for question, concise, verbose in PAIRS:
    # Order 1: concise first. Order 2: verbose first.
    v1 = evals.judge_pairwise(
        question, concise, verbose, rubric=RUBRIC
    )  # A=concise, B=verbose
    v2 = evals.judge_pairwise(
        question, verbose, concise, rubric=RUBRIC
    )  # A=verbose, B=concise

    # Map each verdict back to which underlying answer won.
    winner1 = "concise" if v1 == "A" else "verbose" if v1 == "B" else "tie"
    winner2 = "verbose" if v2 == "A" else "concise" if v2 == "B" else "tie"

    agree = winner1 == winner2 and winner1 != "tie"
    if agree:
        consistent += 1
        verdict = f"consistent ({winner1} wins both ways)"
    else:
        flipped += 1
        verdict = f"POSITION BIAS (order1={winner1}, order2={winner2})"
    print(f"  {verdict:<48} {question[:32]}")

print(
    f"\nConsistent verdicts: {consistent}/{len(PAIRS)}   biased/flipped: {flipped}/{len(PAIRS)}"
)
print(
    "\nEvery flip is a verdict that depended on order, not quality. The mitigation "
    "is built into the test: judge both orders and only count a win when the same "
    "answer wins both. Always sanity-check a judge before trusting its numbers."
)
