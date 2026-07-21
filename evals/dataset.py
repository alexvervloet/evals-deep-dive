"""
evals/dataset.py: the examples you test against (a "golden set").

An eval is only as good as the data you run it on. A dataset here is just a list
of `Example`s, each with:

  - input:    what you feed the system under test (a question, a document, ...)
  - expected: the reference answer or label, if you have one (optional!)
  - metadata: anything else (a category, difficulty, source) for slicing results

Two flavors of eval fall out of whether `expected` is present:

  - reference-based: you know the right answer, so a scorer can compare to it
    (exact match, F1, numeric tolerance). Precise, but you have to label the data.
  - reference-free: you don't have a gold answer, so you judge the output on its
    own merits (is it valid JSON? does an LLM judge rate it well?).

Building a good dataset is the hard, unglamorous, highest-leverage part of evals 
more than any clever metric. Start tiny and real (10 hand-checked examples beats
1000 sloppy ones), cover the cases you actually care about, and grow it from real
failures you find in production.
"""

import json
from dataclasses import dataclass, field


@dataclass
class Example:
    """One test case."""

    input: str
    expected: str | None = None
    metadata: dict = field(default_factory=dict)


def load_jsonl(path: str) -> list[Example]:
    """Load a dataset from a JSON-Lines file (one JSON object per line).

    Each line needs an `input`; `expected` and `metadata` are optional. JSONL is
    the de-facto format for eval sets because it's diffable, appendable, and
    streams line by line.
    """
    examples: list[Example] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            examples.append(
                Example(
                    input=row["input"],
                    expected=row.get("expected"),
                    metadata=row.get("metadata", {}),
                )
            )
    return examples
