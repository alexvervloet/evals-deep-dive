"""
Example 11: human annotation & inter-annotator agreement. (offline)

"The dataset is the hard part" (example 04). Where do the *gold labels* come from?
Usually humans, and humans disagree. If your two annotators only agree 60% of the
time, your "ground truth" is shaky, and every eval built on it inherits that noise.
So before you trust a labelled set, you **measure how much the annotators agreed**.

Two numbers:
  - Observed agreement: the fraction of items both annotators labelled the same.
    Easy, but misleading: if 90% of items are one class, two people guessing that
    class agree 80%+ of the time *by chance*.
  - Cohen's kappa (κ): agreement corrected for chance. κ=1 perfect, κ=0 no better
    than chance, κ<0 worse than chance. Rough reading: >0.8 great, 0.6–0.8 ok,
    <0.6 your labels (or your labelling guidelines) need work.

Then you **adjudicate** the disagreements into a final gold label, and *those* are
what your eval scores against.

This is pure arithmetic, offline and free.

Run it:

    python examples/11_human_annotation.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Two annotators label the same 12 support tickets: billing / bug / other.
ITEMS = [
    "double charged this month", "app crashes on export", "how do I invite a teammate",
    "refund never arrived", "button does nothing on Safari", "love the new dashboard",
    "invoice is wrong", "login loops forever", "where are the docs",
    "charged after cancelling", "dark mode flickers", "your competitor is cheaper",
]
ANNOTATOR_A = ["billing", "bug", "other", "billing", "bug", "other",
               "billing", "bug", "other", "billing", "bug", "other"]
ANNOTATOR_B = ["billing", "bug", "other", "billing", "bug", "billing",  # disagrees: #6
               "billing", "bug", "bug",   "billing", "bug", "other"]    # disagrees: #9


def observed_agreement(a: list[str], b: list[str]) -> float:
    return sum(x == y for x, y in zip(a, b)) / len(a)


def cohens_kappa(a: list[str], b: list[str]) -> float:
    """Agreement corrected for the agreement expected by chance."""
    n = len(a)
    labels = set(a) | set(b)
    po = observed_agreement(a, b)
    # Expected agreement: sum over labels of P(A picks it) * P(B picks it).
    pe = sum((a.count(L) / n) * (b.count(L) / n) for L in labels)
    return (po - pe) / (1 - pe) if pe != 1 else 1.0


def grade_kappa(k: float) -> str:
    if k > 0.8:
        return "excellent"
    if k > 0.6:
        return "acceptable"
    return "weak: fix the guidelines or retrain annotators"


if __name__ == "__main__":
    po = observed_agreement(ANNOTATOR_A, ANNOTATOR_B)
    kappa = cohens_kappa(ANNOTATOR_A, ANNOTATOR_B)

    print("Two annotators labelled 12 tickets (billing / bug / other).\n")
    print(f"Observed agreement: {po:.0%}")
    print(f"Cohen's kappa (κ):  {kappa:.2f}  -> {grade_kappa(kappa)}\n")

    print("Disagreements that need ADJUDICATION (a third person decides the gold label):")
    for item, a, b in zip(ITEMS, ANNOTATOR_A, ANNOTATOR_B):
        if a != b:
            print(f"  - {item!r}:  A said {a!r}, B said {b!r}")

    print(
        "\nTakeaway: observed agreement looks high, but κ is lower because some\n"
        "agreement is just chance; κ is the honest number. A low κ means your eval's\n"
        "'ground truth' is noisy: tighten the labelling guidelines and re-annotate\n"
        "before you trust any score built on these labels. Adjudicated disagreements\n"
        "become the final gold set your eval (examples 01-09) scores against."
    )
