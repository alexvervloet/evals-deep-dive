"""
Example 07: pairwise comparison (win-rate).

Absolute scores from an LLM judge are wobbly. Is this answer a 3 or a 4? LLMs are
much more reliable at *relative* judgements: shown two answers, which is better?
So the standard way to compare two systems (two prompts, two models) is a
**pairwise win-rate**: run both on every input, ask a judge which wins, tally.

Here we compare two system prompts on the QA set:

  A: "answer in a single word"
  B: "answer in a full, helpful sentence"

Neither is universally right; the win-rate tells you which the judge prefers for
*this rubric*. Change the rubric and the winner can flip, which is the real
lesson: "better" is defined by your rubric, so write it deliberately.

Run it:

    secrun python examples/07_pairwise.py
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
dataset = evals.load_jsonl(os.path.join(ROOT, "datasets", "qa.jsonl"))

SYSTEM_A = "Answer the question in a single word or number. Nothing else."
SYSTEM_B = "Answer the question in a full, friendly, helpful sentence."
RUBRIC = (
    "which answer is more helpful and clear for a curious person, while still correct"
)

wins = {"A": 0, "B": 0, "TIE": 0}
for ex in dataset:
    answer_a = evals.generate(SYSTEM_A, ex.input, temperature=0.0)
    answer_b = evals.generate(SYSTEM_B, ex.input, temperature=0.0)
    winner = evals.judge_pairwise(ex.input, answer_a, answer_b, rubric=RUBRIC)
    wins[winner] += 1
    print(f"  {winner:<4} {ex.input[:45]}")

n = len(dataset)
print(f"\nWin-rate over {n} questions (rubric: {RUBRIC!r}):")
print(f"  A (one word):    {wins['A']}/{n} = {wins['A'] / n:.0%}")
print(f"  B (full sentence): {wins['B']}/{n} = {wins['B'] / n:.0%}")
print(f"  ties:            {wins['TIE']}/{n}")

print(
    "\nB should win on *this* rubric (helpfulness). Flip the rubric to 'most "
    "concise' and A would take it. The judge measures whatever you tell it to, "
    "so the rubric is the most important sentence in the whole eval."
)
