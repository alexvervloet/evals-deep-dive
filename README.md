# Evals: A Guided Deep Dive

A hands-on playground for learning how to **evaluate LLM applications**, the skill that
separates "it seems to work" from "I can prove this change made it better." You'll build
a small eval framework from scratch and learn every moving part by building it yourself.
Datasets, code-based scorers, LLM-as-judge, the metrics that matter, judge bias, paired
decision statistics, and regression gates. No promptfoo, no OpenAI Evals, no Ragas. Just
enough code to see how evaluation works.

This is the fifth of eight core repos in the series. The first four teach you to build
LLM apps: the [OpenAI API](https://github.com/alexvervloet/openai-api-deep-dive) and
[Claude API](https://github.com/alexvervloet/claude-api-deep-dive),
[prompt engineering](https://github.com/alexvervloet/prompt-engineering-deep-dive), and a
[RAG](https://github.com/alexvervloet/rag-deep-dive) system on top. This one teaches you
to measure them, which is what makes the other three trustworthy. Once you can put a
number on quality you can improve it on purpose instead of by vibes.

Like its siblings, walk through it rather than reading it. Each section ends with
something to run, and examples 01–04, 10–12, and 14 run offline and free.
[EXERCISES.md](EXERCISES.md) has a predict-then-run prompt for each section.

---

## 0. The one big idea

> **If you can't measure it you can't improve it, so make your app's quality a number
> you can rerun.**

Everything else is detail on top of that. An eval is always four parts: a dataset, a
task, a scorer, a report. Every section below varies one of them. How to score, with code
or with a model. What to measure, which is metrics. Whether to trust the score, which is
bias and statistics. And how to keep it from regressing, which is gates. Hold onto that
and none of this feels complicated.

---

## 1. Setup (5 minutes)

```bash
# 1. Create an isolated Python environment
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Choose your provider (set PROVIDER in .env); your key loads separately
cp .env.example .env
#    Your API key does NOT go in .env. Store it in your OS keychain and run
#    lessons with `secrun`: 2-minute setup in ../docs/SECRETS.md.

# 4. Confirm everything is wired up (makes no API call, costs nothing)
secrun python check_setup.py       # secrun injects your key so the check can see it
```

Evals are provider-agnostic, so this repo is too. Pick whichever stack you set up in the
sibling repos with `PROVIDER` in `.env`.

| `PROVIDER` | Chat model | Key needed |
|------------|-----------|------------|
| `openai` (default) | OpenAI `gpt-5.4-nano` | `OPENAI_API_KEY` |
| `claude` | Claude `claude-haiku-4-5` | `ANTHROPIC_API_KEY` |

The only file that knows which provider you picked is
[evals/providers.py](evals/providers.py). Evals never need embeddings, so unlike the RAG
repo the `claude` stack needs only your Anthropic key and no Voyage.

> **Start before spending anything.** Examples 01–04, 10–12, and 14 are completely
> offline, with no key and no cost. They cover foundations, agent and human evaluation,
> online experiments, and release-decision statistics. The remaining examples make small
> calls, and example 09 makes the most.

---

## 2. The anatomy of an eval

Every eval, however fancy, is the same loop. Run a **task** over a **dataset**, **score**
each output, aggregate into a **report**. To prove it needs no model at all, the first
example evaluates a ten-line rule-based classifier, offline.

```bash
python examples/01_anatomy.py        # offline
```

The task is a function from input to output, and nothing more. Here it's a Python rule.
In Section 6 it's an LLM call. It could be an entire RAG pipeline. The eval machinery
does not change. See [evals/runner.py](evals/runner.py), where `run_eval` is about ten
lines, because an eval really is that simple.

---

## 3. Grade with code first

A **scorer** decides whether an output met the bar. Before paying a model to grade, reach
for code. It's deterministic, instant, free, and enough for a surprising amount: exact
labels, required substrings, formats, valid JSON, numbers within tolerance.

```bash
python examples/02_code_scorers.py   # offline
```

[evals/scorers.py](evals/scorers.py) has `exact_match`, `contains_expected`,
`matches_regex`, `is_valid_json`, `json_has_keys`, and `numeric_close`. Choosing a scorer
is really choosing what "correct" means for your task. `"Positive!"` fails exact match
and passes a contains check, and which one you want is a real decision.

---

## 4. The dataset is the hard part

Clever scorers get the attention, and an eval's quality is capped by its dataset. This
section looks at the three golden sets in [datasets/](datasets/) and the kind of eval
each one enables.

```bash
python examples/03_dataset.py        # offline
```

One distinction matters most.
- **Reference-based** sets know the `expected` answer, so you score against it with exact
  match, F1, or numeric tolerance. Precise, but you have to label it.
- **Reference-free** sets have no `expected`, so you judge the output on its own merits.
  Is it valid JSON? What does an LLM judge rate it? No labels, fuzzier signal.

The datasets here are deliberately tiny. Ten hand-checked, representative examples beat a
thousand sloppy ones, and the best ones come from real failures you find later and add
back so they never regress.

---

## 5. From pass/fail to a decision

A pile of scores isn't actionable until you aggregate it into the numbers you report and
compare.

```bash
python examples/04_metrics.py        # offline math
```

- **Accuracy, or pass rate.** The headline "how often right?"
- **Precision, recall, F1.** For classification, which kind of error you're making: false
  alarms against misses. Accuracy alone can lie.
- **pass@k.** For tasks you sample several times: did any of k tries pass?
- **Confidence intervals.** How much would this number wobble on a rerun?
- **`compare()`.** A first fixed-horizon, independent-sample screen for whether an
  interval clears zero.

See [evals/metrics.py](evals/metrics.py). It teaches the shape of uncertainty and it is
not a release decision. It discards pairing and takes no account of a practical
threshold, multiple metrics, or repeated looks. Example 14 adds those.

---

## 6. Evaluating a real LLM

Now the task is a model. Same loop as Section 2, with only the task changed, so we can
classify the sentiment set and report accuracy plus per-class precision, recall, and
F1.

```bash
secrun python examples/05_classify_eval.py
```

Compare the accuracy to the rule-based baseline from Section 2 on the same data. That
side-by-side number is the entire point of evals.

---

## 7. LLM-as-judge, for grading the ungradeable

Some qualities can't be code-checked. Is this answer helpful? Is this summary faithful?
For those you use a model as the grader.

```bash
secrun python examples/06_llm_judge.py
```

The example answers the QA set and scores each answer two ways at once, with a code
`contains` check and an LLM judge, so you can watch them disagree. The disagreements are
usually correct answers phrased so the literal check misses them, which is exactly where
a judge pays back its real per-call cost. See [evals/judges.py](evals/judges.py).

---

## 8. Pairwise comparison and win-rate

Absolute scores from a judge are wobbly. Is this a 3 or a 4? LLMs are far more reliable
at relative judgements, so the standard way to compare two systems is a **pairwise
win-rate**. Run both, ask a judge which wins, tally.

```bash
secrun python examples/07_pairwise.py
```

The example pits a terse prompt against a verbose one. The winner depends entirely on
the rubric you give the judge. Flip "most helpful" to "most concise" and the result
flips. The rubric is the most important sentence in the whole eval.

---

## 9. Your judge is biased

A judge is a model, so it has biases. The most notorious is position bias, a tendency to
prefer whichever answer came first.

```bash
secrun python examples/08_judge_bias.py
```

The test doubles as the fix. Judge each pair in both orders and only count a win if the
same answer wins both ways. The example measures how often the verdict flips on order
alone, which is why you have to sanity-check a judge before trusting its numbers.

---

## 10. One run is a point estimate

LLM outputs vary run to run, so a single eval number is a sample rather than the truth.

```bash
secrun python examples/09_nondeterminism.py
```

This runs the same eval several times at temperature 0.7 to watch the score wobble,
reports a mean with a confidence interval instead of one number, and uses `compare()` as
a fixed-horizon screen over independent run-level scores. It is the costliest example,
using a small slice and a few runs, so turn them down to spend less. Example 14 handles
the paired per-case release decision.

---

## Going further: five more kinds of eval

The core loop scores a single output string. These extend it to the cases you hit in
practice. Four of them run offline and free, because they are about method. Faithfulness
is a model-graded judge, so it makes small calls.

### Evaluating an agent's trajectory
A right final answer can hide a broken process. The lucky guess, the forbidden tool
call, the 9-step solution to a 2-step task. When the system under test is an agent, grade
the trace, meaning the steps taken plus the answer, on several axes. Answer correctness.
Did it use the required tool. Did it avoid forbidden ones. Did it stay within a step
budget.
```bash
python examples/10_agent_trajectory.py
```

### Human annotation and inter-annotator agreement
Your gold labels usually come from humans, and humans disagree. Before trusting a
labelled set, measure how much the annotators agreed, using observed agreement and
Cohen's kappa, which corrects for chance. A low kappa means your ground truth is noisy,
and every score built on it inherits that noise.
```bash
python examples/11_human_annotation.py
```

### Online evals, or A/B testing on live traffic
A passing offline score doesn't prove real users are better off. The complement is the
online eval. Split live traffic into A (control) and B (the change), compare an outcome
metric you care about, and screen whether the fixed-horizon gap clears its margin while
checking that no guardrail regressed on latency, refusals, or cost.
```bash
python examples/12_online_eval.py
```

### Faithfulness, or did the answer stay grounded in its context?
This is the eval every RAG system needs and correctness-only evals miss. A fluent, even
true answer that asserts something the retrieved context never said is a hallucination.
`judge_faithfulness(context, answer)` is a reference-free judge, needing only the context
and no gold answer, that scores whether every claim is grounded. The example answers the
same questions a grounded way and a loose way, and shows the loose prompt inventing
plausible facts the context doesn't support.
```bash
secrun python examples/13_faithfulness.py
```

### Decision statistics, or when is a candidate ready to release?

Run control and candidate on the same cases, then keep those pairs. Example 14 uses a
paired bootstrap interval, separates statistical evidence from a predeclared minimum
practical effect, plans minimum detectable effect and sample size at a target power,
allocates one family error budget across metrics and interim looks, and demonstrates a
conservative sequential stop. Its first candidate is precisely better and still too small
to ship. The second continues at 100 and 250 pairs before the full interval clears the
practical boundary at 500.

```bash
python examples/14_decision_statistics.py        # offline
```

See [evals/decision.py](evals/decision.py). The teaching implementation uses a percentile
bootstrap, normal planning intervals, and Bonferroni spending. Those choices are readable
and conservative rather than universal. Clustered users, adaptive traffic, rare outcomes,
and regulated decisions all need a design validated for their own sampling process. The
sequential looks refuse any schedule starting below 30 pairs, because at two pairs a
nominal 95% interval actually covers about 70% of the time, and a module about spending
an error budget honestly should not hand you a six-times overspend with the declared
number still printed on it.

---

## 11. The capstone: `eval_run.py`

Everything comes together in one command-line tool. It runs an eval suite, prints a
scored report, and then does the thing that turns evals into a habit: it saves a run,
diffs a new run against it, and fails when quality drops. That last piece is how evals
become a regression gate in CI.

```bash
# Run the default suite (sentiment) and print a report:
secrun python hands_on/eval_run.py

# Run the QA suite a few times to see the score's variance:
secrun python hands_on/eval_run.py qa --runs 3

# Save a baseline, then later diff a new run against it:
secrun python hands_on/eval_run.py sentiment --save baseline.run.json
secrun python hands_on/eval_run.py sentiment --baseline baseline.run.json

# CI gate: exit non-zero if the headline pass rate drops below 0.7:
secrun python hands_on/eval_run.py sentiment --fail-under 0.7
```

Three built-in suites exercise the whole repo. `sentiment` pairs a classifier with a
code scorer, `qa` pairs answers with an LLM judge, and `extraction` pairs JSON with key
checks. Read [hands_on/eval_run.py](hands_on/eval_run.py). The diff uses `compare()` as a
fixed-horizon screen, so it does not cry regression over every numeric change.
**Suggested exercise:** wire `--fail-under` into a pre-commit hook, then watch a
quality-tanking prompt change fail the build.

> **The diff's `± margin` will look huge, often ±40% or more. That is expected rather
> than a bug.** It is the normal-approximation interval on the difference between two
> runs' pass rates, and two things blow it up here. The datasets are tiny, around ten
> examples, so the margin scales as roughly 1/√n and one example flipping is a 10-point
> swing. And each score is binary 0 or 1, which is maximum variance. That is the lesson.
> With a handful of examples this screen cannot separate a small change from noise.
> Shrink the margin with more data or more `--runs`, not by trusting a smaller sample.
> For a real release comparison over the same cases, keep per-case scores and use the
> predeclared policy from Example 14.

---

## Where to go next

You've built a complete small eval framework. The road to production is more of the same
idea, at more scale and more rigor.

- **Eval frameworks.** promptfoo, OpenAI Evals, Inspect, and for RAG, Ragas or DeepEval,
  instead of hand-rolling the runner.
- **Bigger, better datasets.** More examples, harder cases, stratified by category, plus
  generating or mining them from production traffic.
- **Human evaluation.** Annotation workflows and inter-annotator agreement, the ground
  truth you calibrate LLM judges against.
- **Online and production evals.** Scoring real traffic continuously, plus tracing and
  observability (Langfuse, Braintrust, Arize) to see what actually happened. Running a
  sampled judge over live traffic to catch quality regressions over time, and mining
  production failures back into this gold set, is its own bonus dive:
  [Observability](https://github.com/alexvervloet/observability-deep-dive).
- **Guardrails.** Turning evals into runtime checks that block bad outputs before a user
  sees them.
- **Eval-driven development.** Making "the eval suite passed" the definition of done for
  every prompt or model change, exactly like unit tests for code.

Every one of these is a variation on the idea you started with. Make quality a number you
can rerun.

---

## From teaching code to production

This repo taught you to measure a change. In production the measurement has to run
automatically and block bad changes, and the rig that runs it needs the same operational
care as the app it grades.

| This repo's teaching shortcut | In production |
|-------------------------------|---------------|
| You run `eval_run.py` by hand and read the score | An **eval gate** in CI: a threshold that fails the build and blocks the merge |
| The judge model is called bare | Judge calls wrapped in **retries** and counted against a **cost budget** (judging at scale isn't free) |
| Scores printed to the terminal | Results **logged and traced** over time, so you can see drift, not just today's number |
| The dataset and judge prompt are files you edit in place | **Versioned** datasets and judge prompts, so a graded run is reproducible and diffable |
| The capstone diff compares aggregate run rates once | Preserve per-case pairs and predeclare practical effect, power, metric family, and look schedule before the gate sees outcomes |

All seven concerns (observability, cost, reliability, caching, guardrails, prompt
versioning, and eval gates) get built from scratch and wired into one running app in
[Production](https://github.com/alexvervloet/ai-in-production-deep-dive), which is #8 in
the series and where this repo's eval gate sits on the request path. It runs offline on
a mock provider, so you can see the whole ops machinery with no key and no cost.

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
  decision.py               ← paired bootstrap, power, multiplicity, sequential tests
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
  12_online_eval.py         ← fixed-horizon A/B screen + guardrails (offline)
  13_faithfulness.py        ← reference-free groundedness judge for RAG (grounded vs loose)
  14_decision_statistics.py ← paired release evidence, power, multiplicity, looks (offline)
```

---

## Troubleshooting

Run `secrun python check_setup.py` first; it catches most problems. Then, by symptom:

| What you see | What it means / the fix |
|--------------|-------------------------|
| `PROVIDER=... needs ... in the environment` | Set `PROVIDER` in `.env`, then load the key from your keychain by running under `secrun`. See [SECRETS.md](../docs/SECRETS.md). |
| `ModuleNotFoundError` (openai / anthropic / rich) | Dependencies aren't installed or the venv isn't active. `source .venv/bin/activate` then `pip install -r requirements.txt`. |
| `AuthenticationError` / 401 | The key is present but wrong; check it matches the `PROVIDER` you set. |
| Scores change every run | Expected above temperature 0; that's the whole lesson of Section 10. Use `--runs` and confidence intervals; the library's tasks default to temperature 0 for stability. |
| The judge's verdicts seem off | Judges are biased models (Section 9). Calibrate against a few human labels and judge both orders; don't treat a judge as ground truth. |
| `SyntaxError` / odd type errors on startup | You're likely on Python 3.9 or older; this repo needs 3.10+. `check_setup.py` confirms your version. |

Still stuck? Every file is small and self-contained. Open it, read the docstring
at the top, and run it directly.

---

## The series

This is one of the standalone, hands-on deep dives into building with LLM APIs. Eight
core dives, plus the bonus ones listed below. Each one stands on its own, with its own
setup, examples, and capstone, and they all share one house style. Provider-agnostic,
built from scratch with no frameworks, offline-first examples, and a real capstone at
the end. Do them in any order. This sequence builds naturally.

1. [OpenAI API](https://github.com/alexvervloet/openai-api-deep-dive): the API from zero
2. [Claude API](https://github.com/alexvervloet/claude-api-deep-dive): the same ideas, the Anthropic way
3. [Prompt Engineering](https://github.com/alexvervloet/prompt-engineering-deep-dive): shape model behavior with better prompts, using zero-shot and few-shot, chain-of-thought, and roles
4. [RAG](https://github.com/alexvervloet/rag-deep-dive): answer questions over your own documents
5. [Evals](https://github.com/alexvervloet/evals-deep-dive): measure whether a change actually helps
6. [Agents](https://github.com/alexvervloet/agents-deep-dive): give a model tools and a loop so it can act
7. [Prompt Injection & Guardrails](https://github.com/alexvervloet/prompt-injection-deep-dive): attack and defend all of the above
8. [Production](https://github.com/alexvervloet/ai-in-production-deep-dive): operate one app end to end, across observability, cost, reliability, caching, guardrails, prompt versioning, and eval gates

**Bonus dives**, standalone and slotting in where they're most useful:

- [Context Engineering](https://github.com/alexvervloet/context-engineering-deep-dive): manage what's in the window, with memory, compaction, and assembly
- [AI Data Engineering](https://github.com/alexvervloet/ai-data-engineering-deep-dive): the corpus behind the index, with versions, lineage, ACLs, and deletes
- [Multimodal](https://github.com/alexvervloet/multimodal-deep-dive): images and audio as well as text
- [Fine-tuning](https://github.com/alexvervloet/fine-tuning-deep-dive): teach a model new behavior by example
- [MCP](https://github.com/alexvervloet/mcp-deep-dive): serve tools, data, and prompts to any LLM over a standard protocol
- [Local Models](https://github.com/alexvervloet/local-models-deep-dive): run open-weight models on your own machine
- [Agent Harnesses](https://github.com/alexvervloet/agent-harness-deep-dive): build on the loop, adding hooks, permissions, sandboxing, and subagents
- [Realtime Voice](https://github.com/alexvervloet/realtime-voice-deep-dive): low-latency speech-to-speech agents
- [Observability](https://github.com/alexvervloet/observability-deep-dive): watch a running app over time, covering drift, quality, alerting, and the feedback loop
- [Architecture](https://github.com/alexvervloet/architecture-deep-dive): the seams between the components, each decision measured rather than asserted
- [GenAI Security](https://github.com/alexvervloet/genai-security-deep-dive): treat the model as an untrusted principal, and put identity, supply chain, isolation, budgets, and release gates around it
- [Inference Platform Engineering](https://github.com/alexvervloet/inference-platform-deep-dive): turn finite GPU memory and a request queue into latency, throughput, and a fleet size you can defend
- [Testing & Delivery](https://github.com/alexvervloet/testing-and-delivery-deep-dive): decide whether a build is fit to promote, using evidence, gates, staged rollout, and rollback
- [Professional Tools](https://github.com/alexvervloet/professional-tools-deep-dive): rebuild each hand-written piece with the tool professionals reach for, and measure both

And the whole series lands in one codebase in the
[capstone](https://github.com/alexvervloet/deep-dive-capstone): a codebase Q&A tool
built step by step, one tag per dive.

**You are here: #5, Evals.**
