"""
Example 13 — faithfulness: did the answer stay grounded in its context?
=======================================================================

Reference-based correctness (example 05) and the LLM judge (example 06) ask "is
this answer right?". For a RAG system that's not enough, because of a failure they
both miss: a **fluent, even true-sounding answer that asserts something the
retrieved context never said.** That's a hallucination, and it's the single most
important thing to measure about a RAG app — the "faithfulness" (or groundedness)
leg of the RAG eval triad.

Faithfulness needs no gold answer — only the context the model was handed. So it's
a *reference-free* judge: read the CONTEXT and the ANSWER, and score whether every
claim in the answer is supported by the context (true-but-unsupported still fails —
the test is grounding, not truth).

This example runs the SAME questions over the SAME contexts two ways:

  - grounded:  "answer ONLY from the context; if it's not there, say so"
  - loose:     "answer the question" (free to lean on training knowledge)

...and faithfulness-scores both. The contexts deliberately omit a detail or two, so
the loose prompt invents a plausible-sounding answer while the grounded prompt
declines — and only the faithfulness scorer can tell them apart.

Run it (makes a handful of small calls):

    python examples/13_faithfulness.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

import evals

load_dotenv()
evals.ensure_ready()
print(f"Provider: {evals.describe()}\n")

# Each case is a question + the retrieved context. Some contexts intentionally
# LACK the answer — that's where an ungrounded model hallucinates.
CASES = [
    {
        "q": "How much storage does the Free plan include?",
        "context": "Acme Cloud's Free plan includes 1 project and 2 GB of storage.",
    },
    {
        "q": "Does the Pro plan include SSO (single sign-on)?",
        # Context says nothing about SSO — grounded should decline; loose may guess.
        "context": "The Pro plan is $12/mo and includes unlimited projects and priority email support.",
    },
    {
        "q": "How long is a password-reset link valid for?",
        # Context omits the expiry — loose may invent "30 minutes" or "24 hours".
        "context": "To reset your password, open Settings > Security > Reset password and follow the emailed link.",
    },
    {
        "q": "What regions is data stored in?",
        # Context doesn't say — a classic invent-a-fact trap.
        "context": "Acme Cloud encrypts all data at rest and in transit using industry-standard AES-256.",
    },
]

GROUNDED_SYSTEM = (
    "Answer the question using ONLY the provided context. If the context does not "
    "contain the answer, say exactly: \"The context doesn't say.\" Do not use outside "
    "knowledge."
)
# Deliberately pushy: this is the "eager assistant" failure mode — always give a
# confident, specific answer and never hedge. Aligned models resist plain looseness,
# so we make the gap reproducible by instructing it to fill in the blanks. (That
# instruction is itself a realistic anti-pattern people put in their prompts.)
LOOSE_SYSTEM = (
    "You are a confident, decisive assistant. Always give the user a specific, "
    "definitive one-sentence answer. Never say you don't know or that information "
    "is missing — always provide your best concrete answer."
)


def answer(system: str, question: str, context: str) -> str:
    return evals.generate(system, f"Context:\n{context}\n\nQuestion: {question}", temperature=0.0)


grounded_scores: list[float] = []
loose_scores: list[float] = []

print(f"{'faith':>5}  prompt    answer")
print("-" * 78)
for case in CASES:
    for label, system, bucket in [
        ("grounded", GROUNDED_SYSTEM, grounded_scores),
        ("loose", LOOSE_SYSTEM, loose_scores),
    ]:
        out = answer(system, case["q"], case["context"])
        score = evals.judge_faithfulness(case["context"], out)
        bucket.append(score.score)
        flag = "" if score.passed else "  <- unsupported by context"
        print(f"{score.detail.split()[-1]:>5}  {label:<8}  {' '.join(out.split())[:46]}{flag}")
    print(f"        Q: {case['q']}")

mean = lambda xs: sum(xs) / len(xs) if xs else 0.0
print("-" * 78)
print(f"Mean faithfulness — grounded prompt: {mean(grounded_scores):.2f}"
      f"   |   loose prompt: {mean(loose_scores):.2f}")
print(
    "\nThe grounded prompt scores higher because it declines when the context is "
    "silent,\ninstead of inventing a plausible answer. A correctness-only eval would "
    "miss this:\nthe loose answers often *sound* right (and may even be true) — they "
    "just aren't\nsupported by what was retrieved. That gap is exactly what RAG "
    "systems must measure."
)
