"""
Example 15: hill-climbing a prompt against an eval, and the score that lies.

Once you have an eval, you can use it as a target: read the failures, edit the
prompt, keep the edit if the score rises, repeat. Tools now automate the loop. It
works, and it has one classic way to fool you: climbing the same examples you read
the failures from. This example runs one list of proposed edits under two policies.

  train      keep an edit if the train score rises (the naive loop)
  validation keep it only if a separate validation split rises by more than one
             example; the optimizer never sees why validation moved

Test is scored once, at the end, for both. The splits are frozen in
datasets/tickets.jsonl (24 train, 16 validation, 20 test), so every run, and every
person, compares against the same numbers.

Two of the five edits are traps. In the train split, every ticket that mentions
Monday happens to be a bug, so "monday -> bug" fixes three train failures and learns
nothing true. And pasting the remaining train failures into the prompt as examples
gets train to 100% by giving the model the answers.

Offline (default) the model is a transparent stand-in (evals/hillclimb.py), so the
numbers are exact: the train-only loop reports 100% and scores 75% on test. The
validation gate scores 80%, one ticket better, while looking worse on train. The
number to remember is the first pair: 100% reported, 75% delivered.

With --live the same loop runs on the real model, and it's a different picture worth
seeing. Measured twice on gpt-6-luna (2026-10-07): with no rules at all the model
already scores 96% on train and 100% on validation, so most edits have nothing left
to fix. The train-only loop still kept the Monday rule both times, because it took
train from 96% to 100%. Its prompt scored 90% on test in one run and 100% in the
other; the gated prompt scored 100% both times. On 20 test tickets one ticket is 5
points, so a single run can't tell those apart. That's not a flaw in the demo; it's
what example 14's sample-size arithmetic says about any eval this small.

Run it:

    python examples/15_hill_climbing.py
    secrun python examples/15_hill_climbing.py --live
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

from evals import hillclimb
from evals.dataset import load_jsonl

load_dotenv()
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
examples = load_jsonl(os.path.join(ROOT, "datasets", "tickets.jsonl"))
splits = {name: [e for e in examples if e.metadata["split"] == name] for name in ("train", "validation", "test")}
margin = 1 / len(splits["validation"])  # one example's worth; smaller moves are noise

if "--live" in sys.argv:
    from evals import providers

    providers.ensure_ready()
    print(f"Provider: {providers.describe()}\n")
    cache: dict[tuple[str, str], str] = {}

    def classify(prompt, text):
        key = (prompt.render(), text)
        if key not in cache:
            cache[key] = providers.generate(key[0], text, temperature=0.0, max_tokens=10)
        return cache[key]
else:
    print("Model: the offline stand-in in evals/hillclimb.py (exact, no key)\n")
    classify = hillclimb.offline_model

results = {}
for policy in ("train", "validation"):
    run = hillclimb.climb(
        hillclimb.Prompt(), hillclimb.proposed_edits(), splits["train"], splits["validation"],
        classify, policy, margin,
    )
    results[policy] = run
    print(f"Policy: keep an edit if {policy} improves")
    print(f"  {'edit':<58}{'train':>7}{'valid':>7}   kept")
    for step in run.steps:
        print(f"  {step.edit[:58]:<58}{step.train:>7.0%}{step.validation:>7.0%}   {'yes' if step.accepted else 'no'}")
    print()

print("Each policy's final prompt, scored on all three splits (test only now):")
print(f"  {'policy':<12}{'train':>7}{'valid':>7}{'test':>7}")
for policy, run in results.items():
    train, _ = hillclimb.score(run.final, splits["train"], classify)
    valid, _ = hillclimb.score(run.final, splits["validation"], classify)
    test, _ = hillclimb.score(run.final, splits["test"], classify)
    print(f"  {policy:<12}{train:>7.0%}{valid:>7.0%}{test:>7.0%}")

naive_train, _ = hillclimb.score(results["train"].final, splits["train"], classify)
naive_test, _ = hillclimb.score(results["train"].final, splits["test"], classify)
gated_test, _ = hillclimb.score(results["validation"].final, splits["test"], classify)
kept = [step.edit for step in results["train"].steps if step.accepted]
baseline_valid, _ = hillclimb.score(hillclimb.Prompt(), splits["validation"], classify)
test_step = 1 / len(splits["test"])

print(f"\nThe train-only loop kept {len(kept)} edit(s) and would report {naive_train:.0%}.")
print(f"On tickets it never saw, its prompt scores {naive_test:.0%}; the gated prompt scores {gated_test:.0%}.")
if naive_train - naive_test > test_step + 1e-9:
    print(
        "That gap is what climbing your own failures buys: edits that fit the examples\n"
        "you read, coincidences and copied answers included."
    )
if abs(naive_test - gated_test) <= test_step + 1e-9:
    print(
        f"The two test scores are within one ticket ({test_step:.0%}) of each other, so this run\n"
        "can't separate the policies. Run it again, or grow the split (example 14)."
    )
if baseline_valid >= 1 - 1 / len(splits["validation"]):
    print(
        f"The unedited prompt already scores {baseline_valid:.0%} on validation, so there's\n"
        "little left to climb. A saturated eval is telling you to write harder cases,\n"
        "not to keep editing the prompt."
    )
print(
    "Either way: choose edits on validation, report test, and score test once. If you\n"
    "peek at test to pick an edit, it has quietly become a second validation set."
)
