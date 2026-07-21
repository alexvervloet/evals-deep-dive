"""
evals/runner.py: the eval loop, and the Report it produces.

This is the heart of the whole repo, and it's tiny, because an eval *is* tiny
once you see it:

    for each example in the dataset:
        output = task(example.input)        # run the system under test
        for each scorer: score the output   # grade it

`run_eval` does exactly that and returns a `Report` you can summarize, save,
reload, and diff against another run. The "system under test" (`task`) is just a
function `str -> str`: it can be a plain Python function (example 01), a single
LLM call (example 05), or an entire RAG pipeline. The runner doesn't care.

Pure stdlib, so importing it costs nothing.
"""

import json
from dataclasses import dataclass
from typing import Callable

from .dataset import Example
from .scorers import Score

# A scorer is Callable[[output, Example], Score]; a task is Callable[[input], output].
Scorer = Callable[[str, Example], Score]
Task = Callable[[str], str]


@dataclass
class Result:
    """One example's outcome: what the task produced and how each scorer rated it."""

    example: Example
    output: str
    scores: dict[str, Score]


class Report:
    """The results of an eval run, with helpers to summarize and persist it."""

    def __init__(self, results: list[Result], scorer_names: list[str]) -> None:
        self.results = results
        self.scorer_names = scorer_names

    def passes_for(self, scorer: str) -> list[bool]:
        return [r.scores[scorer].passed for r in self.results]

    def scores_for(self, scorer: str) -> list[float]:
        return [r.scores[scorer].score for r in self.results]

    def pass_rate(self, scorer: str) -> float:
        passes = self.passes_for(scorer)
        return sum(passes) / len(passes) if passes else 0.0

    def mean_score(self, scorer: str) -> float:
        scores = self.scores_for(scorer)
        return sum(scores) / len(scores) if scores else 0.0

    def summary(self) -> dict:
        """Per-scorer pass rate and mean score: the numbers you compare runs on."""
        return {
            name: {"pass_rate": self.pass_rate(name), "mean_score": self.mean_score(name)}
            for name in self.scorer_names
        }

    def print_summary(self) -> None:
        """Plain-text summary (no dependencies). The capstone prints a richer view."""
        print(f"{len(self.results)} examples, {len(self.scorer_names)} scorer(s)")
        for name in self.scorer_names:
            print(f"  {name:<20} pass {self.pass_rate(name):.0%}   mean {self.mean_score(name):.3f}")

    def failures(self, scorer: str) -> list[Result]:
        """The examples that failed a given scorer: where you go to learn."""
        return [r for r in self.results if not r.scores[scorer].passed]

    def save(self, path: str) -> None:
        data = {
            "scorer_names": self.scorer_names,
            "results": [
                {
                    "input": r.example.input,
                    "expected": r.example.expected,
                    "metadata": r.example.metadata,
                    "output": r.output,
                    "scores": {
                        n: {"passed": s.passed, "score": s.score, "detail": s.detail}
                        for n, s in r.scores.items()
                    },
                }
                for r in self.results
            ],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "Report":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        results = [
            Result(
                example=Example(input=r["input"], expected=r["expected"], metadata=r.get("metadata", {})),
                output=r["output"],
                scores={n: Score(**s) for n, s in r["scores"].items()},
            )
            for r in data["results"]
        ]
        return cls(results, data["scorer_names"])


def run_eval(task: Task, dataset: list[Example], scorers: dict[str, Scorer]) -> Report:
    """Run `task` over every example and apply every scorer. Returns a Report.

    `scorers` is a name -> scorer mapping so each metric is labelled in the
    report. Keep tasks deterministic (temperature=0) unless you're deliberately
    studying variance (example 09).
    """
    results: list[Result] = []
    for example in dataset:
        output = task(example.input)
        scored = {name: fn(output, example) for name, fn in scorers.items()}
        results.append(Result(example=example, output=output, scores=scored))
    return Report(results, list(scorers.keys()))
