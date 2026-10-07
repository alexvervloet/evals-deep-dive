"""
evals/hillclimb.py: improve a prompt against an eval without fooling yourself.

Once an eval exists, the obvious next step is to use it as a target: look at what
fails, change the prompt, keep the change if the score goes up, repeat. Tools now do
this automatically. The loop is sound. The trap is which score you climb.

If you choose edits by looking at failures and keep them because the same examples
improved, you are fitting those examples. An edit can raise that score by learning
something real, by learning a coincidence in your data, or by copying the answers
into the prompt, and the score can't tell those apart. So the examples are split
three ways, frozen in the data file so every run compares to every other:

    train       the failures the optimizer is allowed to look at
    validation  decides whether an edit is kept; the optimizer never sees why
    test        scored once, at the end, to report what you'd actually ship

`climb()` runs the same edits under two acceptance policies, so you can see the
difference. The offline model is a stand-in that follows keyword rules and repeats
any example pasted into its prompt word for word. That's crude, and it's exactly how
real models behave toward those two edits, only more obviously: few-shot examples
copied from the eval mostly help the eval.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable

from evals.dataset import Example

LABELS = ("billing", "bug", "account", "other")


@dataclass(frozen=True)
class Prompt:
    """A classification prompt as data: keyword rules plus pasted examples."""

    rules: tuple[tuple[tuple[str, ...], str], ...] = ()
    examples: tuple[tuple[str, str], ...] = ()

    def render(self) -> str:
        """The prompt text a real model receives (see --live in example 15)."""
        lines = [
            "Classify the support ticket into exactly one of: " + ", ".join(LABELS) + ".",
            "Answer with the category name only.",
        ]
        for keywords, label in self.rules:
            lines.append(f"- A ticket that mentions {' or '.join(repr(k) for k in keywords)} is {label}.")
        if self.examples:
            lines.append("Examples:")
            lines += [f"  {text!r} -> {label}" for text, label in self.examples]
        # No "otherwise answer other" here. Next to a partial rule list, a real model
        # takes that literally and files every unmatched ticket as other; an earlier
        # version of this file did exactly that and halved the model's accuracy.
        return "\n".join(lines)


def offline_model(prompt: Prompt, text: str) -> str:
    """A transparent stand-in for a model: copied examples first, then rules in order."""
    for example_text, label in prompt.examples:
        if example_text == text:
            return label
    low = text.lower()
    for keywords, label in prompt.rules:
        if any(k in low for k in keywords):
            return label
    return "other"


Classify = Callable[[Prompt, str], str]


def score(prompt: Prompt, examples: list[Example], classify: Classify) -> tuple[float, list[Example]]:
    """Accuracy, plus the examples it got wrong."""
    wrong = [e for e in examples if classify(prompt, e.input).strip().lower() != e.expected]
    return 1 - len(wrong) / len(examples), wrong


@dataclass(frozen=True)
class Edit:
    """One proposed change. `apply` sees the current prompt and its TRAIN failures."""

    name: str
    apply: Callable[[Prompt, list[Example]], Prompt]


def add_rule(name: str, keywords: tuple[str, ...], label: str) -> Edit:
    return Edit(name, lambda p, _failures: replace(p, rules=p.rules + ((keywords, label),)))


def paste_failures(name: str) -> Edit:
    """Copy the current train failures, with their answers, into the prompt."""
    return Edit(
        name,
        lambda p, failures: replace(
            p, examples=p.examples + tuple((e.input, e.expected) for e in failures)
        ),
    )


@dataclass
class Step:
    edit: str
    train: float
    validation: float
    accepted: bool


@dataclass
class Climb:
    policy: str
    steps: list[Step] = field(default_factory=list)
    final: Prompt = Prompt()


def climb(
    start: Prompt,
    edits: list[Edit],
    train: list[Example],
    validation: list[Example],
    classify: Classify,
    policy: str,
    margin: float,
) -> Climb:
    """Try each edit on the current best prompt and keep it or not.

    policy="train":      keep an edit if the train score rises. The naive loop.
    policy="validation": keep it only if validation rises by more than `margin`.
    Neither policy looks at test.
    """
    if policy not in ("train", "validation"):
        raise ValueError(f"policy must be 'train' or 'validation', not {policy!r}")
    result = Climb(policy=policy, final=start)
    best_train, failures = score(start, train, classify)
    best_validation, _ = score(start, validation, classify)
    for edit in edits:
        candidate = edit.apply(result.final, failures)
        cand_train, cand_failures = score(candidate, train, classify)
        cand_validation, _ = score(candidate, validation, classify)
        if policy == "train":
            keep = cand_train > best_train
        else:
            keep = cand_validation > best_validation + margin
        result.steps.append(Step(edit.name, cand_train, cand_validation, keep))
        if keep:
            result.final = candidate
            best_train, best_validation, failures = cand_train, cand_validation, cand_failures
    return result


def proposed_edits() -> list[Edit]:
    """What an optimizer reading the train failures would plausibly propose, in order."""
    return [
        add_rule("rule: refund/charged/invoice/payment/card -> billing",
                 ("refund", "charged", "invoice", "payment", "card"), "billing"),
        add_rule("rule: crash/error/freez/sync -> bug", ("crash", "error", "freez", "sync"), "bug"),
        add_rule("rule: password/log in/login/2fa/email address/account -> account",
                 ("password", "log in", "login", "2fa", "email address", "account"), "account"),
        # Every train ticket that mentions Monday happens to be a bug. A coincidence
        # in 24 examples looks exactly like a pattern.
        add_rule("rule: monday -> bug (a train coincidence)", ("monday",), "bug"),
        paste_failures("paste the remaining train failures as examples"),
    ]
