"""
evals/judges.py — model-graded scorers (LLM-as-judge).
======================================================

Some qualities can't be checked with code: "is this summary faithful?", "is this
answer helpful?", "which of these two replies is better?". For those you use the
model itself as the grader — an *LLM judge*.

Two shapes, both here:

  - pointwise: grade ONE answer against a rubric, returning a score. Simple, but
    absolute scores from an LLM are wobbly ("is this a 3 or a 4?").
  - pairwise: show the judge TWO answers and ask which is better. LLMs are far
    more reliable at *relative* judgements than absolute ones, so pairwise
    win-rate is usually the better tool for comparing two systems.

⚠️ Judges are themselves models, so they have biases — they can favour the first
option, longer answers, or their own style. Treat a judge as a system to be
evaluated, not gospel: calibrate it against human labels, and mitigate known
biases (example 08 shows position bias and the swap-and-average fix). Grading runs
at temperature=0 for stability.
"""

import re

from .providers import generate
from .scorers import Score

_POINTWISE_SYSTEM = (
    "You are a strict, fair grader. Read the QUESTION and the ANSWER, apply the "
    "RUBRIC, and rate the answer from 1 (poor) to 5 (excellent). Reply with ONLY "
    "the integer."
)


def judge_pointwise(question: str, answer: str, rubric: str, reference: str | None = None) -> Score:
    """Grade one answer 1–5 against a rubric; returned as a 0–1 Score.

    Pass a `reference` answer when you have one — the judge grades far more
    consistently when it can compare against a known-good answer than when judging
    in a vacuum.
    """
    user = f"QUESTION:\n{question}\n\nRUBRIC:\n{rubric}\n"
    if reference:
        user += f"\nREFERENCE (a known-good answer):\n{reference}\n"
    user += f"\nANSWER TO GRADE:\n{answer}\n\nScore (1-5):"

    reply = generate(_POINTWISE_SYSTEM, user, temperature=0.0, max_tokens=8)
    m = re.search(r"[1-5]", reply)
    raw = int(m.group()) if m else 1
    return Score(passed=raw >= 4, score=(raw - 1) / 4, detail=f"judge rated {raw}/5")


_PAIRWISE_SYSTEM = (
    "You compare two answers, A and B, to the same question and decide which is "
    "better according to the rubric. Reply with exactly one token: A, B, or TIE."
)


def judge_pairwise(question: str, answer_a: str, answer_b: str, rubric: str = "overall quality and correctness") -> str:
    """Ask the judge which answer is better. Returns 'A', 'B', or 'TIE'.

    Relative judgements are more reliable than absolute scores — but watch for
    *position bias* (a tendency to favour whichever answer is shown first). To
    neutralize it, run both orders and average; see example 08.
    """
    user = (
        f"QUESTION:\n{question}\n\nRUBRIC: {rubric}\n\n"
        f"ANSWER A:\n{answer_a}\n\nANSWER B:\n{answer_b}\n\n"
        f"Which is better? Reply A, B, or TIE."
    )
    reply = generate(_PAIRWISE_SYSTEM, user, temperature=0.0, max_tokens=8).strip().upper()
    if reply.startswith("A"):
        return "A"
    if reply.startswith("B"):
        return "B"
    return "TIE"
