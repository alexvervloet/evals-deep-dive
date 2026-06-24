# Exercises — make the learning stick

Reading code teaches you less than *predicting* what it will do and then checking.
This file turns each section of the [README](README.md) into a few quick
active-recall prompts.

How to use it: work the section first, then come back. **Commit to an answer
before you run or reveal** — the prediction is where the learning happens, even
(especially) when you're wrong. Answers are hidden behind ▸ toggles.

> Examples 01–04 are **(offline)** — no API call, no cost. The rest make small,
> cheap calls; example 09 makes the most.

---

## Section 2 — The anatomy of an eval **(offline)**

**Recall.** Name the four parts of every eval, and which one this whole repo
argues is the hardest.

<details><summary>▸ Answer</summary>

dataset → task → scorer → report. The **dataset** is the hard part: clever scorers
and metrics can't rescue an eval built on unrepresentative or mislabeled examples.
</details>

**Do.** In `examples/01_anatomy.py`, the "task" is a rule-based classifier. Add a
word to the `GOOD`/`BAD` sets and rerun. Did accuracy move? Why is a keyword
baseline doomed to miss the "hard" rows?

<details><summary>▸ Answer</summary>

Because sarcasm and mixed sentiment ("Not as bad as the reviews said") have no
reliable keyword signal — the words point the wrong way. That ceiling is the
motivation for an LLM (example 05) and for measuring at all.
</details>

---

## Section 3 — Code-based scorers **(offline)**

**Predict.** Output is `"Positive!"` and the expected label is `"positive"`. Does
`exact_match` pass? Does `contains_expected`?

<details><summary>▸ Answer</summary>

`exact_match` fails (the `!` and capital P make the strings unequal);
`contains_expected` passes (case-insensitive substring). Choosing the scorer is
choosing what "correct" means — and why example 05's task normalizes the label
before scoring.
</details>

---

## Section 4 — Metrics **(offline)**

**Recall.** A spam filter labels *every* email as spam. What are its precision and
recall for the "spam" class? Why does that show accuracy alone can lie?

<details><summary>▸ Answer</summary>

Recall is 100% (it caught all real spam) but precision is terrible (most flagged
emails weren't spam). On an inbox that's mostly ham, plain accuracy could even
look bad — and on mostly-spam data, a "flag everything" model could post high
accuracy while being useless. Precision/recall/F1 expose what one number hides.
</details>

**Do.** In `examples/04_metrics.py`, change `candidate` so its scores barely beat
`baseline`. Does `compare()` still call it a real improvement? What does that
teach about "+2%" claims?

<details><summary>▸ Answer</summary>

Once the gap shrinks into the margin of error, `likely_real` flips to False. A
small improvement that's inside the noise isn't a result you can trust — you'd
need more data or runs to separate it from chance.
</details>

---

## Section 5 — Evaluating an LLM classifier

**Predict, then run.** Will `examples/05_classify_eval.py` score 100%? Where will
its errors cluster, and are those errors the *model's* fault?

<details><summary>▸ Answer</summary>

Almost certainly not 100% — errors cluster on the "hard" rows (sarcasm, mixed
signals). Some of those labels are genuinely debatable, so a "wrong" answer can be
a disagreement with a shaky label, not a model failure. A good eval surfaces that
rather than hiding it.
</details>

---

## Section 6 — LLM-as-judge

**Recall.** When should you reach for an LLM judge instead of a code scorer? Name
one risk you take on when you do.

<details><summary>▸ Answer</summary>

Use a judge when correctness is about *meaning* and code can't check it (is this
summary faithful? is this answer helpful?). The risk: the judge is itself a model
with biases and costs a call per grade — so it must be calibrated, not trusted
blindly.
</details>

---

## Section 7 — Pairwise win-rate

**Predict.** `examples/07_pairwise.py` compares a one-word prompt (A) against a
full-sentence prompt (B) on the rubric "more helpful." Who wins? Now imagine the
rubric is "most concise" — who wins then?

<details><summary>▸ Answer</summary>

B (helpful) under the helpfulness rubric; A (concise) under the conciseness
rubric. Same answers, opposite winners — because the rubric *defines* "better."
The rubric is the most important sentence in the eval.
</details>

---

## Section 8 — Judge bias

**Recall.** What is position bias, and what's the one-line fix used in
`examples/08_judge_bias.py`?

<details><summary>▸ Answer</summary>

Position bias is a judge favouring whichever answer is shown first. The fix: judge
each pair in *both* orders and only count a win if the same answer wins both ways
(otherwise call it a tie). This both detects and neutralizes the bias.
</details>

---

## Section 9 — Nondeterminism & statistics

**Predict, then run.** You run the same eval twice at temperature 0.7. Will you get
the same pass rate? What does that mean for trusting a single eval number?

<details><summary>▸ Answer</summary>

Probably not — the score wobbles run to run. So one number is a point estimate, not
the truth. Report a mean with a confidence interval, and to claim system B beats A,
run enough times that the difference clears the noise (`compare()`).
</details>

---

## Capstone — `eval_run.py`

**Do.** Save a baseline (`--save baseline.run.json`), then run again with
`--baseline baseline.run.json`. The diff says "within noise" even though the
numbers differ slightly — why is that the *right* answer?

<details><summary>▸ Answer</summary>

Because run-to-run variation makes small deltas meaningless. `compare()` only flags
a change as real when it exceeds the margin of error — so the tool refuses to cry
"regression!" (or "improvement!") over noise. That restraint is what makes an
automated gate trustworthy.
</details>

**Stretch.** Wire `python hands_on/eval_run.py sentiment --fail-under 0.7` into a
git pre-commit hook or CI step. Now a prompt change that tanks quality fails the
build — evals as a safety net, not a one-off.

---

### Where to take it next

Invent your own. Take a task you actually care about, write ten honest examples,
pick a scorer, and get a number. The first time an eval stops you from shipping a
"better" prompt that was actually worse, the whole discipline clicks.
