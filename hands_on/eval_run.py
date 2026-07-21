#!/usr/bin/env python3
"""
eval_run.py: the capstone: a tiny eval runner you'd actually use.

Everything in the repo comes together into one command-line tool that runs an
eval *suite* (a task + dataset + scorers), prints a scored report, and, the part
that makes evals a habit instead of a one-off, lets you **save a run, diff a new
run against it, and fail when quality drops**. That last part is how evals become
a regression gate in CI.

Three built-in suites:
  sentiment  LLM classifier vs labels         (code scorer: exact_match)
  qa         LLM answers, graded by a judge    (code + LLM-judge scorers)
  extraction LLM extracts JSON                 (valid-JSON + required-keys)

Examples
--------
  # Run the default suite (sentiment) and print a report
  secrun python hands_on/eval_run.py

  # Run the QA suite a few times to see the score's variance
  secrun python hands_on/eval_run.py qa --runs 3

  # Save a baseline, then later diff a new run against it
  secrun python hands_on/eval_run.py sentiment --save baseline.run.json
  secrun python hands_on/eval_run.py sentiment --baseline baseline.run.json


  # CI gate: exit non-zero if the headline pass rate drops below 0.8
  secrun python hands_on/eval_run.py sentiment --fail-under 0.8
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import evals
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASETS = os.path.join(ROOT, "datasets")
LABELS = ("positive", "negative", "neutral")


def _to_label(text: str) -> str:
    t = text.strip().lower()
    return next((lbl for lbl in LABELS if lbl in t), t)


def sentiment_suite():
    system = (
        "You are a sentiment classifier. Reply with exactly one word: positive, "
        "negative, or neutral."
    )

    def task(text: str) -> str:
        return _to_label(evals.generate(system, text, temperature=0.0))

    return {
        "dataset": evals.load_jsonl(os.path.join(DATASETS, "sentiment.jsonl")),
        "task": task,
        "scorers": {"exact_match": evals.exact_match},
        "primary": "exact_match",
    }


def qa_suite():
    answer_system = "Answer the question correctly in a short, natural sentence."
    rubric = (
        "High score if the answer is factually correct and addresses the question, "
        "even if phrased differently from the reference; low if wrong or evasive."
    )

    def task(question: str) -> str:
        return evals.generate(answer_system, question, temperature=0.0)

    def judge_scorer(output, example):
        return evals.judge_pointwise(
            example.input, output, rubric, reference=example.expected
        )

    return {
        "dataset": evals.load_jsonl(os.path.join(DATASETS, "qa.jsonl")),
        "task": task,
        "scorers": {"contains": evals.contains_expected, "judge": judge_scorer},
        "primary": "judge",
    }


def extraction_suite():
    system = (
        "Extract the person's name, email, and plan from the message. Reply with "
        'ONLY a JSON object with keys "name", "email", and "plan".'
    )

    def task(text: str) -> str:
        return evals.generate(system, text, temperature=0.0)

    return {
        "dataset": evals.load_jsonl(os.path.join(DATASETS, "extraction.jsonl")),
        "task": task,
        "scorers": {
            "valid_json": evals.is_valid_json,
            "has_keys": evals.json_has_keys(["name", "email", "plan"]),
        },
        "primary": "has_keys",
    }


SUITES = {"sentiment": sentiment_suite, "qa": qa_suite, "extraction": extraction_suite}


def parse_args(argv):
    p = argparse.ArgumentParser(
        description="Run an eval suite, with optional baseline diff and CI gate.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "suite",
        nargs="?",
        default="sentiment",
        choices=list(SUITES),
        help="Which suite to run (default: sentiment).",
    )
    p.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Run the suite N times to see score variance (default 1).",
    )
    p.add_argument(
        "--save", metavar="FILE", help="Save the run's full report to FILE (JSON)."
    )
    p.add_argument(
        "--baseline",
        metavar="FILE",
        help="Diff this run against a previously saved report.",
    )
    p.add_argument(
        "--fail-under",
        type=float,
        metavar="X",
        help="Exit non-zero if the primary pass rate is below X (a CI gate).",
    )
    return p.parse_args(argv)


def main(argv) -> int:
    args = parse_args(argv)
    load_dotenv()
    evals.ensure_ready()

    console = Console()
    console.print(f"[dim]Provider: {evals.describe()}  |  Suite: {args.suite}[/dim]\n")

    suite = SUITES[args.suite]()
    primary = suite["primary"]

    # Run once, or N times for variance. Keep the last full report for save/diff.
    primary_rates = []
    report = None
    for i in range(max(1, args.runs)):
        report = evals.run_eval(suite["task"], suite["dataset"], suite["scorers"])
        primary_rates.append(report.pass_rate(primary))
        if args.runs > 1:
            console.print(
                f"[dim]run {i + 1}: {primary} pass rate {primary_rates[-1]:.0%}[/dim]"
            )

    assert report is not None

    # Per-scorer summary for the (last) run.
    table = Table(title=f"Results: {len(report.results)} examples")
    table.add_column("Scorer", style="cyan")
    table.add_column("Pass rate", justify="right")
    table.add_column("Mean score", justify="right")
    for name in report.scorer_names:
        marker = " *" if name == primary else ""
        table.add_row(
            name + marker,
            f"{report.pass_rate(name):.0%}",
            f"{report.mean_score(name):.3f}",
        )
    console.print(table)
    console.print("[dim]* = primary scorer (used for the gate and diff)[/dim]")

    # Variance summary across runs.
    primary_rate = sum(primary_rates) / len(primary_rates)
    if args.runs > 1:
        lo, hi = evals.confidence_interval(primary_rates)
        console.print(
            f"\n{primary} over {args.runs} runs: mean {primary_rate:.0%}, "
            f"95% CI [{lo:.0%}, {hi:.0%}]"
        )

    # Diff against a saved baseline.
    #
    # A note on the "± margin" you'll see here. It's the 95% confidence interval
    # on the *difference* between the two runs' pass rates (see evals.compare).
    # On these teaching datasets it comes out startlingly wide, often ±40% or
    # more, and that is the honest answer, not a defect. Two things blow it up:
    #   1. Tiny n. The margin shrinks as ~1/sqrt(n). With ~10 examples one
    #      example flipping is a 10-point swing, so the band is enormous. At
    #      n=100 it's ~3x tighter; at n=1000, ~10x.
    #   2. Binary scores. Each score is 0 or 1, which carries the most variance
    #      possible. A graded (0.0-1.0) scorer or averaging over --runs is tighter.
    # The lesson: with a handful of examples you *cannot* distinguish a real
    # quality change from noise, so `likely_real` will call almost any diff
    # "within noise". The fix is more data (or more runs), not a smaller margin.
    if args.baseline:
        base = evals.Report.load(args.baseline)
        console.print(f"\n[bold]Diff vs baseline[/bold] ({args.baseline}):")
        if primary in base.scorer_names:
            base_rate = base.pass_rate(primary)
            cmp = evals.compare(base.scores_for(primary), report.scores_for(primary))
            verdict = "REAL change" if cmp["likely_real"] else "within noise"
            console.print(
                f"  {primary}: {base_rate:.0%} -> {primary_rate:.0%} "
                f"({cmp['diff']:+.0%} ± {cmp['margin']:.0%}, {verdict})"
            )
        else:
            console.print(
                f"  [yellow]baseline has no '{primary}' scorer to compare.[/yellow]"
            )

    # Save the full report.
    if args.save:
        report.save(args.save)
        console.print(f"\n[dim]Saved report to {args.save}[/dim]")

    # CI gate.
    if args.fail_under is not None and primary_rate < args.fail_under:
        console.print(
            f"\n[bold red]GATE FAILED[/bold red]: {primary} pass rate "
            f"{primary_rate:.0%} < {args.fail_under:.0%}"
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
