# Evals — A Guided Deep Dive

A hands-on playground for learning how to **evaluate LLM applications** — the skill
that separates "it seems to work" from "I can prove this change made it better."
You'll build a small eval framework from scratch and learn every moving part:
datasets, code-based scorers, LLM-as-judge, the metrics that matter, judge bias,
statistical significance, and regression gates — by building each one yourself. No
promptfoo, no OpenAI Evals, no Ragas; just enough code to *see* how evaluation
works.

This is the fifth of eight core repos in the series. The first four teach you to
*build* LLM apps — the [OpenAI API](https://github.com/Ailuue/openai-api-deep-dive) and
[Claude API](https://github.com/Ailuue/claude-api-deep-dive), [prompt engineering](https://github.com/Ailuue/prompt-engineering-deep-dive), and a [RAG](https://github.com/Ailuue/rag-deep-dive) system on top.
This one teaches you to *measure* them. It's the meta-skill that makes the other
three trustworthy: once you can put a number on quality, you can improve it on
purpose instead of by vibes.

Like its siblings, it's meant to be *walked through*, not just read. Each section
ends with something to run, and the first four run **offline and free**.
[EXERCISES.md](EXERCISES.md) has a predict-then-run prompt for each section.

---

## 0. The one big idea

> **If you can't measure it, you can't improve it — so make your app's quality a
> number you can rerun.**

Everything else is detail on top of that. An eval is always four parts —
**a dataset, a task, a scorer, a report** — and every section below is a variation
on one of them: how to score (code vs a model), what to measure (metrics), whether
to trust the score (bias, statistics), and how to keep it from regressing (gates).
Hold onto that and none of this feels complicated.

---

## 1. Setup (5 minutes)

```bash
# 1. Create an isolated Python environment
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Choose your provider and add your key
cp .env.example .env
#    ...then open .env. Set PROVIDER to "openai" or "claude" and paste the key.

# 4. Confirm everything is wired up (makes no API call, costs nothing)
python check_setup.py
```

Evals are provider-agnostic, so this repo is too — pick whichever stack you set up
in the sibling repos with `PROVIDER` in `.env`:

| `PROVIDER` | Chat model | Key needed |
|------------|-----------|------------|
| `openai` (default) | OpenAI `gpt-4o-mini` | `OPENAI_API_KEY` |
| `claude` | Claude `claude-haiku-4-5` | `ANTHROPIC_API_KEY` |

The only file that knows which provider you picked is
[evals/providers.py](evals/providers.py). (Evals never need embeddings, so unlike
the RAG repo the `claude` stack needs only your Anthropic key — no Voyage.)

> 💡 **Start before spending anything.** Examples 01–04 are completely offline —
> no key, no cost. They cover the dataset → scorer → metric foundations. The rest
> make small, cheap calls; example 09 makes the most.

---

## 2. The anatomy of an eval

Every eval, however fancy, is the same loop: run a **task** over a **dataset**,
**score** each output, aggregate into a **report**. To prove it needs no model at
all, the first example evaluates a ten-line rule-based classifier — offline.

```bash
python examples/01_anatomy.py        # offline
```

The key realization: **the task is just a function from input to output.** Today
it's a Python rule; in Section 6 it's an LLM call; it could be an entire RAG
pipeline. The eval machinery doesn't change. See [evals/runner.py](evals/runner.py)
— `run_eval` is about ten lines, because an eval really is that simple.

---

## 3. Code-based scorers — grade with code first

A **scorer** decides whether an output met the bar. Before paying a model to
grade, reach for code: deterministic, instant, free, and enough for a surprising
amount — exact labels, required substrings, formats, valid JSON, numbers within
tolerance.

```bash
python examples/02_code_scorers.py   # offline
```

[evals/scorers.py](evals/scorers.py) has `exact_match`, `contains_expected`,
`matches_regex`, `is_valid_json`, `json_has_keys`, and `numeric_close`. Choosing a
scorer is really choosing *what "correct" means* for your task — `"Positive!"`
fails exact match but passes a contains check, and which you want is a real
decision.

---

## 4. The dataset is the hard part

Clever scorers get the attention, but an eval's quality is capped by its dataset.
This section looks at the three golden sets in [datasets/](datasets/) and the kind
of eval each enables.

```bash
python examples/03_dataset.py        # offline
```

The key distinction:
- **reference-based** (`expected` is known) — score against the right answer
  (exact match, F1, numeric tolerance). Precise, but you have to label it.
- **reference-free** (no `expected`) — judge the output on its own merits (valid
  JSON? an LLM judge's rating?). No labels, fuzzier signal.

The datasets here are deliberately tiny: ten hand-checked, representative examples
beat a thousand sloppy ones, and the best ones come from real failures you find
later and add back so they never regress.

---

## 5. Metrics — from pass/fail to a decision

A pile of scores isn't actionable until you aggregate it into the numbers you
report and compare.

```bash
python examples/04_metrics.py        # offline math
```

- **accuracy / pass rate** — the headline "how often right?"
- **precision / recall / F1** — for classification, *which kind* of error (false
  alarms vs misses); accuracy alone can lie.
- **pass@k** — for tasks you sample several times: did any of k tries pass?
- **confidence intervals** — how much would this number wobble on a re-run?
- **`compare()`** — the one that matters: is B *really* better than A, or noise?

See [evals/metrics.py](evals/metrics.py). The discipline this buys you: never ship
a "+2%" that sits inside the margin of error.

---

## 6. Evaluating a real LLM

Now the task is a model. Same loop as Section 2 — only the task changed — so we can
classify the sentiment set and report accuracy plus per-class precision/recall/F1.

```bash
python examples/05_classify_eval.py
```

Compare the accuracy to the rule-based baseline from Section 2 on the same data:
that side-by-side number is the entire point of evals.

---

## 7. LLM-as-judge — grading the ungradeable

Some qualities can't be code-checked: "is this answer helpful?", "is this summary
faithful?". For those you use a model as the grader.

```bash
python examples/06_llm_judge.py
```

The example answers the QA set and scores each answer two ways at once — a code
`contains` check and an LLM judge — so you can watch them disagree. The
disagreements are usually correct answers phrased so the literal check misses
them: exactly where a judge earns its (real, per-call) cost. See
[evals/judges.py](evals/judges.py).

---

## 8. Pairwise comparison — win-rate

Absolute scores from a judge are wobbly ("is this a 3 or a 4?"). LLMs are far more
reliable at *relative* judgements, so the standard way to compare two systems is a
**pairwise win-rate**: run both, ask a judge which wins, tally.

```bash
python examples/07_pairwise.py
```

The example pits a terse prompt against a verbose one. The winner depends entirely
on the **rubric** you give the judge — flip "most helpful" to "most concise" and
the result flips. The rubric is the most important sentence in the whole eval.

---

## 9. Your judge is biased

A judge is a model, so it has biases — most notoriously **position bias**, a
tendency to prefer whichever answer came first.

```bash
python examples/08_judge_bias.py
```

The test doubles as the fix: judge each pair in **both** orders and only count a
win if the same answer wins both ways. The example measures how often the verdict
flips on order alone — and why you must sanity-check a judge before trusting its
numbers.

---

## 10. Nondeterminism — one run is a point estimate

LLM outputs vary run to run, so a single eval number is a sample, not the truth.

```bash
python examples/09_nondeterminism.py
```

This runs the same eval several times at temperature 0.7 to watch the score wobble,
reports a mean with a confidence interval instead of one number, and uses
`compare()` to decide whether two prompts *really* differ. (It's the costliest
example — it uses a small slice and a few runs; turn them down to spend less.)

---

## Going further — four more kinds of eval

The core loop scores a single output string. These extend it to the cases you hit in
practice. The first three run **offline and free** (they're about *method*); the
fourth (faithfulness) is a model-graded judge, so it makes small calls.

### Evaluating an agent's trajectory
A right final answer can hide a broken process: the lucky guess, the forbidden tool
call, the 9-step solution to a 2-step task. When the system under test is an *agent*,
grade the **trace** (steps taken + answer) on several axes — answer correctness, did
it use the required tool, did it avoid forbidden ones, did it stay within a step
budget.
```bash
python examples/10_agent_trajectory.py
```

### Human annotation & inter-annotator agreement
Your gold labels usually come from humans, and humans disagree. Before trusting a
labelled set, measure how much annotators agreed — **observed agreement** and
**Cohen's kappa** (agreement corrected for chance). A low kappa means your "ground
truth" is noisy, and every score built on it inherits that noise.
```bash
python examples/11_human_annotation.py
```

### Online evals — A/B testing on live traffic
A passing offline score doesn't prove real users are better off. The complement is
the **online eval**: split live traffic into A (control) and B (the change), compare
an outcome metric you care about, and ship B only if the gap clears the margin of
error *and* no guardrail (latency, refusals, cost) regressed.
```bash
python examples/12_online_eval.py
```

### Faithfulness — did the answer stay grounded in its context?
The eval every RAG system needs and correctness-only evals miss: a fluent,
even *true* answer that asserts something the retrieved context never said — a
hallucination. `judge_faithfulness(context, answer)` is a **reference-free** judge
(it needs only the context, no gold answer) that scores whether every claim is
grounded. The example answers the same questions a *grounded* way and a *loose* way
and shows the loose prompt inventing plausible facts the context doesn't support.
```bash
python examples/13_faithfulness.py
```

---

## 11. The capstone: `eval_run.py`

Everything comes together in one command-line tool that runs an eval **suite**,
prints a scored report, and — the part that turns evals into a habit — saves a run,
diffs a new run against it, and **fails when quality drops**. That last part is how
evals become a regression gate in CI.

```bash
# Run the default suite (sentiment) and print a report:
python hands_on/eval_run.py

# Run the QA suite a few times to see the score's variance:
python hands_on/eval_run.py qa --runs 3

# Save a baseline, then later diff a new run against it:
python hands_on/eval_run.py sentiment --save baseline.run.json
python hands_on/eval_run.py sentiment --baseline baseline.run.json

# CI gate: exit non-zero if the headline pass rate drops below 0.7:
python hands_on/eval_run.py sentiment --fail-under 0.7
```

Three built-in suites exercise the whole repo: `sentiment` (classifier + code
scorer), `qa` (answers + LLM judge), and `extraction` (JSON + key checks). Read
[hands_on/eval_run.py](hands_on/eval_run.py) — the diff uses `compare()` so it only
cries "regression" when a change clears the noise. **Suggested exercise:** wire
`--fail-under` into a pre-commit hook, then watch a quality-tanking prompt change
fail the build.

> ⚠️ **The diff's `± margin` will look huge — often ±40% or more — and that's
> honest, not a bug.** It's the 95% confidence interval on the *difference*
> between two runs' pass rates, and two things blow it up here: the datasets are
> tiny (~10 examples, so the margin scales as ~1/√n and one example flipping is a
> 10-point swing) and each score is binary 0/1 (maximum variance). The takeaway
> *is* the lesson: with a handful of examples you genuinely can't tell a real
> quality change from noise, so `compare()` will call almost any diff "within
> noise." Shrink the margin with more data or more `--runs`, not by trusting a
> smaller sample.

---

## Where to go next

You've built a complete small eval framework. The road to production is more of the
same idea, at more scale and rigor:

- **Eval frameworks** — promptfoo, OpenAI Evals, Inspect, and (for RAG)
  Ragas/DeepEval, instead of hand-rolling the runner.
- **Bigger, better datasets** — more examples, harder cases, stratified by
  category; and generating or mining them from production traffic.
- **Human evaluation** — annotation workflows and inter-annotator agreement, the
  ground truth you calibrate LLM judges against.
- **Online / production evals** — scoring real traffic continuously, plus tracing
  and observability (Langfuse, Braintrust, Arize) to see what actually happened.
- **Guardrails** — turning evals into runtime checks that block bad outputs before
  a user sees them.
- **Eval-driven development** — making "the eval suite passed" the definition of
  done for every prompt or model change, exactly like unit tests for code.

Each is a variation on the one idea you started with: make quality a number you can
rerun.

---

## From teaching code to production

This repo taught you to *measure* a change. In production the measurement has to
run automatically and *block* bad changes — and the harness that runs it needs
the same operational care as the app it grades:

| This repo's teaching shortcut | In production |
|-------------------------------|---------------|
| You run `eval_run.py` by hand and read the score | An **eval gate** in CI — a threshold that fails the build and blocks the merge |
| The judge model is called bare | Judge calls wrapped in **retries** and counted against a **cost budget** (judging at scale isn't free) |
| Scores printed to the terminal | Results **logged and traced** over time, so you can see drift, not just today's number |
| The dataset and judge prompt are files you edit in place | **Versioned** datasets and judge prompts, so a graded run is reproducible and diffable |
| One run is a point estimate (Section 10) | The gate decides on **aggregates with thresholds**, run on every change, not a single sample |

These shortcuts are right for learning and wrong for production. All seven
concerns — observability, cost, reliability, caching, guardrails, prompt
versioning, and eval gates — are built from scratch and wired into one running
app in **[Production](https://github.com/Ailuue/ai-in-production-deep-dive)** (#8 in the
series), where this repo's eval *gate* sits on the request path. It runs
**offline on a mock provider**, so you can see the whole ops machinery with no key
and no cost.

---

## File map

```
check_setup.py              ← run first: verifies Python, packages, provider, key
README.md                   ← this guide
EXERCISES.md                ← predict-then-run prompts, one per section
evals/                      ← the from-scratch library (read it!)
  providers.py              ← the ONLY provider-specific file: generate()
  dataset.py                ← Example + load_jsonl (golden sets)
  scorers.py                ← code-based scorers + the Score type
  judges.py                 ← LLM-as-judge: pointwise + pairwise
  metrics.py                ← accuracy, precision/recall/F1, pass@k, CIs, compare
  runner.py                 ← run_eval + the Report (save / load / diff)
datasets/                   ← small golden sets (JSONL)
  sentiment.jsonl           ← classification (labels)
  qa.jsonl                  ← short-answer QA
  extraction.jsonl          ← structured extraction (JSON)
hands_on/
  eval_run.py               ← capstone: suite runner + baseline diff + CI gate
examples/
  01_anatomy.py             ← the dataset->task->scorer->report loop (offline)
  02_code_scorers.py        ← deterministic code scorers (offline)
  03_dataset.py             ← reference-based vs reference-free (offline)
  04_metrics.py             ← accuracy, F1, pass@k, confidence, compare (offline)
  05_classify_eval.py       ← evaluating an LLM classifier
  06_llm_judge.py           ← LLM-as-judge (pointwise), vs a code scorer
  07_pairwise.py            ← pairwise win-rate between two prompts
  08_judge_bias.py          ← position bias and how to mitigate it
  09_nondeterminism.py      ← variance, confidence intervals, is-it-real?
  10_agent_trajectory.py    ← grade an agent's steps, not just its answer (offline)
  11_human_annotation.py    ← annotator agreement & Cohen's kappa (offline)
  12_online_eval.py         ← A/B testing on live traffic; significance + guardrails (offline)
  13_faithfulness.py        ← reference-free groundedness judge for RAG (grounded vs loose)
```

---

## Troubleshooting

Run `python check_setup.py` first — it catches most problems. Then, by symptom:

| What you see | What it means / the fix |
|--------------|-------------------------|
| `PROVIDER=... needs ... in .env` | The active stack is missing its key. Set `PROVIDER` and the matching key in `.env`. |
| `ModuleNotFoundError` (openai / anthropic / rich) | Dependencies aren't installed or the venv isn't active. `source .venv/bin/activate` then `pip install -r requirements.txt`. |
| `AuthenticationError` / 401 | The key is present but wrong — check it matches the `PROVIDER` you set. |
| Scores change every run | Expected above temperature 0 — that's the whole lesson of Section 10. Use `--runs` and confidence intervals; the library's tasks default to temperature 0 for stability. |
| The judge's verdicts seem off | Judges are biased models (Section 9). Calibrate against a few human labels and judge both orders; don't treat a judge as ground truth. |
| `SyntaxError` / odd type errors on startup | You're likely on Python 3.9 or older; this repo needs 3.10+. `check_setup.py` confirms your version. |

Still stuck? Every file is small and self-contained — open it, read the docstring
at the top, and run it directly.

---

## The series

This is one of thirteen standalone, hands-on deep dives into building with LLM APIs — eight core, plus five bonus dives.
Each one stands on its own — its own setup, examples, and capstone — and they all
share the same house style: provider-agnostic, built from scratch (no
frameworks), offline-first examples, and a real capstone. Do them in any order;
this sequence builds naturally:

1. [OpenAI API](https://github.com/Ailuue/openai-api-deep-dive) — the API from zero
2. [Claude API](https://github.com/Ailuue/claude-api-deep-dive) — the same ideas, the Anthropic way
3. [Prompt Engineering](https://github.com/Ailuue/prompt-engineering-deep-dive) — shape model behavior with better prompts (zero/few-shot, chain-of-thought, roles)
4. [RAG](https://github.com/Ailuue/rag-deep-dive) — answer questions over your own documents
5. [Evals](https://github.com/Ailuue/evals-deep-dive) — measure whether a change actually helps
6. [Agents](https://github.com/Ailuue/agents-deep-dive) — give a model tools and a loop so it can act
7. [Prompt Injection & Guardrails](https://github.com/Ailuue/prompt-injection-deep-dive) — attack and defend all of the above
8. [Production](https://github.com/Ailuue/ai-in-production-deep-dive) — operate one app end to end: observability, cost, reliability, caching, guardrails, prompt versioning, eval gates

**Bonus dives** — standalone, slotting in where they're most useful:

- [Context Engineering](https://github.com/Ailuue/context-engineering-deep-dive) — manage what's in the window: memory, compaction, assembly
- [Multimodal](https://github.com/Ailuue/multimodal-deep-dive) — images & audio, not just text
- [Fine-tuning](https://github.com/Ailuue/fine-tuning-deep-dive) — teach a model new behavior by example
- [MCP](https://github.com/Ailuue/mcp-deep-dive) — serve tools, data & prompts to any LLM over a standard protocol
- [Local Models](https://github.com/Ailuue/local-models-deep-dive) — run open-weight models on your own machine
- [Agent Harnesses](https://github.com/Ailuue/agent-harness-deep-dive) — build on the loop: hooks, permissions, sandboxing, subagents
- [Realtime Voice](https://github.com/Ailuue/realtime-voice-deep-dive) — low-latency speech-to-speech agents

**You are here: #5 — Evals.**
