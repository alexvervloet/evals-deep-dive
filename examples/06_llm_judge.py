"""
Example 06: LLM-as-judge (pointwise).

Some answers can't be graded by code. "What is the capital of France?" -> "It's
Paris, the City of Light." is correct, but a strict `contains` check for "Paris"
might pass while "seven" vs "7" fails on a different question. When correctness is
about *meaning*, you let a model grade: an LLM judge.

This example answers the QA dataset, then scores each answer two ways at once:

  - contains (code): is the expected string literally present?
  - judge (model):   does an LLM, given the question + a reference, rate the
                     answer 1-5 against a rubric?

Watch where they disagree, usually a correct answer phrased so the literal
string check misses it. That's the gap LLM judges fill. (And remember: the judge
is itself a model with biases (example 08), so calibrate it, don't trust it
blindly.)

Run it:

    secrun python examples/06_llm_judge.py
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

ANSWER_SYSTEM = "Answer the question correctly in a short, natural sentence."
RUBRIC = (
    "Give a high score if the answer is factually correct and addresses the "
    "question, even if phrased differently from the reference. Low score if it is "
    "wrong, evasive, or off-topic."
)


def qa_task(question: str) -> str:
    return evals.generate(ANSWER_SYSTEM, question, temperature=0.0)


def judge_scorer(output: str, example) -> evals.Score:
    """Wrap the pointwise judge as a scorer: (output, example) -> Score."""
    return evals.judge_pointwise(
        example.input, output, RUBRIC, reference=example.expected
    )


# One task call + one judge call per example, both applied to the same answer.
report = evals.run_eval(
    qa_task, dataset, {"contains": evals.contains_expected, "judge": judge_scorer}
)

print(f"{'code':>4} {'judge':>5}  question -> answer")
print("-" * 70)
for r in report.results:
    code = "ok" if r.scores["contains"].passed else "X"
    judge = r.scores["judge"].detail.replace("judge rated ", "")
    answer = " ".join(r.output.split())[:40]
    flag = (
        "  <- disagree"
        if r.scores["contains"].passed != r.scores["judge"].passed
        else ""
    )
    print(f"{code:>4} {judge:>5}  {r.example.input[:34]:<34} -> {answer}...{flag}")

print(
    f"\nLiteral 'contains' pass rate: {report.pass_rate('contains'):.0%}"
    f"   |   judge mean score: {report.mean_score('judge'):.2f}"
)
print(
    "The disagreements are correct answers the strict string check missed, which "
    "is exactly when an LLM judge earns its (real, per-call) cost."
)
