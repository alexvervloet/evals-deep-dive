"""
Example 08: is your judge biased? Two checks, and what each can't see.

An LLM judge is a model, so it can have biases. The most notorious is **position
bias**: preferring whichever answer is shown first (or last), so a "winner" is
really just "whoever went first." Others include favouring longer answers and a
model preferring its own style.

Check 1, position. Judge each pair in BOTH orders. Use pairs of answers that are
about equally good, because that's where order gets to be the tiebreaker. Each
pair lands in one of three buckets:
  - consistent win: the same answer wins both ways,
  - consistent tie: tie both ways (fine, the judge saw no difference),
  - flipped: the verdict changed with order. That's position bias.

The fix is the same as the test: run both orders and only count a win when the
same answer wins both. How much it matters depends entirely on the judge. Four
near-equal pairs, both orders, five runs each, measured 2026-10-05:

    gpt-5.4-nano       15 of 20 flipped
    gpt-4o-mini         5 of 20 flipped
    gpt-6-luna          0 of 20 flipped (it called 15 of them ties)
    claude-haiku-4-5    0 of 20 flipped (10 ties)

So you may well see zero flips here. That's a result, not a broken demo: it's
the check passing for this judge on this data. Run it again whenever you change
the judge model.

Check 2, length. Now pair a terse correct answer with a longer correct one. Nano,
gpt-4o-mini and Haiku picked the longer answer in both orders 20 times out of 20;
luna did 13 times and called most of the rest ties. The position check calls
those wins "consistent", and they are. Swapping order can't detect a preference that holds
in both orders. Whether "longer wins" is a bias or a fair reading of "helpful"
is a question about your rubric, and only a rubric that says what to reward, or
a human spot-check, can answer it.

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

# Check 1 uses near-equal pairs: two correct answers of similar length and quality.
NEAR_EQUAL = [
    ("What is the capital of France?", "Paris is the capital of France.", "France's capital is Paris."),
    ("How many continents are there?", "There are seven continents.", "Most geographers count seven continents."),
    ("What is the chemical symbol for gold?", "The symbol for gold is Au.", "Gold is written Au on the periodic table."),
    ("What is the square root of 144?", "The square root of 144 is 12.", "It's 12, because 12 x 12 = 144."),
]

# Check 2 pairs a terse correct answer (first) with a longer correct one (second).
CONCISE_VS_VERBOSE = [
    ("What is the capital of France?", "Paris.", "The capital of France is Paris, which sits on the river Seine."),
    (
        "How many continents are there?",
        "Seven.",
        "There are seven continents: Africa, Antarctica, Asia, Europe, North America, "
        "Oceania, and South America.",
    ),
    ("What is the chemical symbol for gold?", "Au.", "Gold's chemical symbol is Au, from the Latin word 'aurum'."),
    ("What is the square root of 144?", "12.", "The square root of 144 is 12, since 12 times 12 equals 144."),
]

RUBRIC = "which answer is more correct and helpful"


def both_orders(question: str, first: str, second: str) -> tuple[str, str]:
    """Judge the pair both ways; return which underlying answer won each time."""
    v1 = evals.judge_pairwise(question, first, second, rubric=RUBRIC)  # A=first
    v2 = evals.judge_pairwise(question, second, first, rubric=RUBRIC)  # A=second
    w1 = "first" if v1 == "A" else "second" if v1 == "B" else "tie"
    w2 = "second" if v2 == "A" else "first" if v2 == "B" else "tie"
    return w1, w2


def classify(w1: str, w2: str) -> str:
    if w1 != w2:
        return "flipped"
    return "consistent tie" if w1 == "tie" else "consistent win"


print("Check 1: position bias, near-equal pairs judged in both orders")
counts = {"consistent win": 0, "consistent tie": 0, "flipped": 0}
for question, a, b in NEAR_EQUAL:
    w1, w2 = both_orders(question, a, b)
    verdict = classify(w1, w2)
    counts[verdict] += 1
    label = f"{verdict} (order1={w1}, order2={w2})"
    print(f"  {label:<46} {question[:32]}")
print(
    f"\n  flipped: {counts['flipped']}/{len(NEAR_EQUAL)}   "
    f"consistent wins: {counts['consistent win']}   consistent ties: {counts['consistent tie']}"
)
print(
    "  A flip is a verdict decided by order, not quality. Zero flips means this judge\n"
    "  passed this check on this data; older judges failed it badly (see the docstring)."
)

print("\nCheck 2: length, a terse correct answer vs a longer correct one")
longer_won = 0
for question, terse, verbose in CONCISE_VS_VERBOSE:
    w1, w2 = both_orders(question, terse, verbose)
    verdict = classify(w1, w2)
    if verdict == "consistent win" and w1 == "second":
        longer_won += 1
    print(f"  {verdict + f' ({w1}, {w2})':<46} {question[:32]}")
print(f"\n  the longer answer won in both orders: {longer_won}/{len(CONCISE_VS_VERBOSE)}")
print(
    "  Where the longer answer wins both ways, the swap test calls it consistent, and\n"
    "  it is. Swapping can't see a preference that holds in both orders. Whether that's\n"
    "  length bias or a fair reading of 'helpful' is up to your rubric. Always\n"
    "  sanity-check a judge before trusting it."
)
