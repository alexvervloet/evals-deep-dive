# Exercises: make the learning stick

Reading code teaches you less than *predicting* what it will do and then checking.
This file turns each section of the [README](README.md) into a few quick
active-recall prompts.

How to use it: work the section first, then come back. **Commit to an answer
before you run or reveal.** The prediction is where the learning happens, even
(especially) when you're wrong. Answers are hidden behind ▸ toggles.

> Examples 01–04, 10–12, and 14 are **(offline)**: no API call, no cost. The
> remaining examples make small calls; example 09 makes the most.

---

## Section 2: The anatomy of an eval **(offline)**

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
reliable keyword signal; the words point the wrong way. That ceiling is the
motivation for an LLM (example 05) and for measuring at all.
</details>

---

## Section 3: Code-based scorers **(offline)**

**Predict.** Output is `"Positive!"` and the expected label is `"positive"`. Does
`exact_match` pass? Does `contains_expected`?

<details><summary>▸ Answer</summary>

`exact_match` fails (the `!` and capital P make the strings unequal);
`contains_expected` passes (case-insensitive substring). Choosing the scorer is
choosing what "correct" means, and why example 05's task normalizes the label
before scoring.
</details>

---

## Section 4: Metrics **(offline)**

**Recall.** A spam filter labels *every* email as spam. What are its precision and
recall for the "spam" class? Why does that show accuracy alone can lie?

<details><summary>▸ Answer</summary>

Recall is 100% (it caught all real spam) but precision is terrible (most flagged
emails weren't spam). On an inbox that's mostly ham, plain accuracy could even
look bad, and on mostly-spam data, a "flag everything" model could post high
accuracy while being useless. Precision/recall/F1 expose what one number hides.
</details>

**Do.** In `examples/04_metrics.py`, change `candidate` so its scores barely beat
`baseline`. Does the fixed-horizon interval still clear zero? Why would either
result remain insufficient to authorize a release?

<details><summary>▸ Answer</summary>

Once the gap shrinks into the margin, `likely_real` flips to False: this screen
cannot distinguish it from sampling variation. If it stays True, the comparison
still discarded per-case pairing and never accounted for a practical threshold,
multiple metrics, or repeated looks. Example 14 supplies that decision contract.
</details>

---

## Section 5: Evaluating an LLM classifier

**Predict, then run.** Will `examples/05_classify_eval.py` score 100%? Where will
its errors cluster, and are those errors the *model's* fault?

<details><summary>▸ Answer</summary>

Almost certainly not 100%. Errors cluster on the "hard" rows (sarcasm, mixed
signals). Some of those labels are genuinely debatable, so a "wrong" answer can be
a disagreement with a shaky label, not a model failure. A good eval surfaces that
rather than hiding it.
</details>

---

## Section 6: LLM-as-judge

**Recall.** When should you reach for an LLM judge instead of a code scorer? Name
one risk you take on when you do.

<details><summary>▸ Answer</summary>

Use a judge when correctness is about *meaning* and code can't check it (is this
summary faithful? is this answer helpful?). The risk: the judge is itself a model
with biases and costs a call per grade, so it must be calibrated, not trusted
blindly.
</details>

---

## Section 7: Pairwise win-rate

**Predict.** `examples/07_pairwise.py` compares a one-word prompt (A) against a
full-sentence prompt (B) on the rubric "more helpful." Who wins? Now imagine the
rubric is "most concise". Who wins then?

<details><summary>▸ Answer</summary>

B (helpful) under the helpfulness rubric; A (concise) under the conciseness
rubric. Same answers, opposite winners, because the rubric *defines* "better."
The rubric is the most important sentence in the eval.
</details>

---

## Section 8: Judge bias

**Recall.** What is position bias, and what's the one-line fix used in
`examples/08_judge_bias.py`?

<details><summary>▸ Answer</summary>

Position bias is a judge favouring whichever answer is shown first. Fix it by judging
each pair in *both* orders and only counting a win if the same answer wins both ways
(otherwise call it a tie). This both detects and neutralizes the bias.
</details>

---

## Section 9: Nondeterminism & statistics

**Predict, then run.** You run the same eval twice at temperature 0.7. Will you get
the same pass rate? What does that mean for trusting a single eval number?

<details><summary>▸ Answer</summary>

Probably not; the score wobbles run to run. So one number is a point estimate, not
the truth. Report a mean with an interval. Here `compare()` is only a fixed-horizon
screen over independent run scores; paired release evidence comes in Example 14.
</details>

---

## Going further

**Recall (trajectory, `10`).** The "lucky" agent scores 100% on the final answer but
0% on tool use. What does that reveal, and why isn't answer-accuracy enough for an
agent?

<details><summary>▸ Answer</summary>

It **guessed**. It reached the right answer without doing the work (never called the
required tool). For an agent, a right answer can hide a broken or unsafe process, so
you grade the **trajectory** (steps + answer) on several axes: correctness, required
tool used, no forbidden tools, within the step budget.
</details>

**Predict (annotation, `11`).** Observed agreement is 83% but Cohen's kappa is only
0.75. Why is kappa lower, and which number should you trust?

<details><summary>▸ Answer</summary>

Some of that 83% is agreement you'd expect **by chance** (especially with imbalanced
classes). Kappa subtracts the chance baseline, so it's the **honest** reliability
number. A low kappa means your gold labels are noisy, so fix the guidelines and
re-annotate before trusting any score built on them.
</details>

**Recall (online eval, `12`, offline).** The headline screen favors B on
satisfaction. Why is that not yet a ship verdict?

<details><summary>▸ Answer</summary>

The fixed-horizon screen only says its interval cleared zero. The effect must also
clear a predeclared practical threshold, every inspected metric and look must share
one error budget, and no guardrail may regress. Example 14 makes the first three
requirements executable; the guardrail remains a separate release condition.
</details>

**Predict (faithfulness, `13`).** A RAG answer says "the reset link is valid for 30
minutes", and that happens to be true, but the retrieved context never mentions an
expiry. Does it pass a *correctness* check? A *faithfulness* check? Why does the gap
matter?

<details><summary>▸ Answer</summary>

It can pass correctness (it's true) but **fails faithfulness** (the context didn't
support it; the model made it up and got lucky). That's the danger: an ungrounded
answer that happens to be right today teaches you to trust a system that's actually
hallucinating. Faithfulness needs no gold answer, only the context, so it catches
exactly what correctness misses. The fix at answer time is the grounded prompt:
answer only from context, decline when it's silent.
</details>

### Decision statistics (`14`, offline)

**Predict, then run.** The first candidate's paired interval is entirely above
zero but entirely below the +3 percentage-point practical threshold. Which
evidence state and decision should appear?

<details><summary>▸ Answer</summary>

`statistical_improvement_only` and **HOLD**. Enough data can measure a tiny effect
precisely; precision does not make the effect worth migration cost or release risk.
</details>

**Calculate.** With variance and error controls held fixed, what happens to MDE
when paired sample size grows from 100 to 400? Why?

<details><summary>▸ Answer</summary>

It halves, because the normal planning approximation scales as
`difference_std / sqrt(pairs)`. Four times the pairs buys twice the sensitivity,
not four times.
</details>

**Predict.** Four independent null metrics each tested at alpha .05 have what
chance of at least one false positive? Why does the example allocate .003125 to
each of sixteen metric-look decisions instead?

<details><summary>▸ Answer</summary>

`1 - 0.95**4 ≈ 18.5%`. Dividing the family alpha `.05` by `4 metrics × 4 looks`
gives `.003125` per decision; Bonferroni's union bound keeps the whole declared
family at or below five percent even when metrics and looks are dependent. It is
conservative, which is the price of the simple guarantee.
</details>

**Break, diagnose, repair.** Shuffle only the candidate score list before calling
`paired_bootstrap`. The function cannot detect that both lists still have equal
length. What invariant did the caller violate, and what is the production repair?

<details><summary>▸ Answer</summary>

Position no longer names the same case in both arms, so the computed differences
are meaningless even though every number is valid. Join control and candidate by
a trusted unique case (or user/session) ID, reject missing and duplicate matches,
then pass the aligned scores. Length validation prevents truncation; it cannot
prove identity alignment the caller discarded.
</details>

---

## Capstone: `eval_run.py`

**Do.** Save a baseline (`--save baseline.run.json`), then run again with
`--baseline baseline.run.json`. The diff says "within noise" even though the
numbers differ slightly. Why is that the *right* answer?

<details><summary>▸ Answer</summary>

Because run-to-run variation makes small deltas weak evidence. `compare()` keeps
this teaching gate from reacting to every numeric change, but it remains an
independent-sample fixed-horizon approximation. A production gate over the same
cases should retain per-case results and apply Example 14's predeclared policy.
</details>

**Stretch.** Wire `secrun python hands_on/eval_run.py sentiment --fail-under 0.7` into a
git pre-commit hook or CI step. Now a prompt change that tanks quality fails the
build: evals as a safety net, not a one-off.

---

### Where to take it next

Invent your own. Take a task you actually care about, write ten honest examples,
pick a scorer, and get a number. The first time an eval stops you from shipping a
"better" prompt that was actually worse, the whole discipline clicks.
