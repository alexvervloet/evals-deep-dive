# Chapter 5: Measurement, or How to Stop Shipping by Vibes

*This is the textbook chapter for the Evals deep dive. The [README](README.md) is the lab manual; this is the lecture. It covers why evaluating language model applications is genuinely hard, where the field's techniques came from, and the statistical honesty that separates a number you can act on from a number that flatters you.*

---

## 5.1 The crisis nobody budgeted for

Software engineering spent fifty years building a culture of testing. Write a function, write a test that calls it with known inputs, assert the exact expected output, run it on every change. The whole edifice rests on one assumption so basic nobody thought to state it: the same input produces the same output, and there is a fact of the matter about whether the output is correct.

Language models break both halves of that assumption at once. The same input can produce different outputs on different runs. And for the interesting tasks (summarize this document, answer this customer, review this code) there is no single correct output to assert against; there is a space of acceptable outputs and a fuzzy boundary around it. Your test suite's most fundamental tool, `assertEqual`, has nothing to grab.

Teams meet this problem in a predictable sequence. First they eyeball: try a few inputs, outputs look good, ship it. Then a prompt change improves the three cases they checked and quietly breaks two categories they didn't, and they discover the failure weeks later, from users. Then someone says "we should really test this," discovers that testing means something different here, and the team either builds an evaluation practice or continues shipping by vibes with extra steps. This dive exists so you can skip to the good ending.

The word the field settled on is **evals**, and the discipline can be stated in a sentence you have already met twice in this series:

> **If you can't measure it, you can't improve it. So make your app's quality a number you can rerun.**

The phrase echoes a century of management proverbs, but the engineering content is real. "Rerun" is the load-bearing word. A one-time quality check is an anecdote. A rerunnable one is an instrument: change the prompt, rerun, compare. Swap the model, rerun, compare. It converts "I think the new prompt is better" into "the new prompt scored 0.84 against 0.71, on the same cases, and here are the ones it fixed." One of those statements survives a design review.

It is also, frankly, a hiring signal. Building a flashy demo with these models is easy, which is exactly why it impresses no one anymore. Knowing how to measure one is rarer, and it is the skill interviewers probe when they ask "how did you know your change helped?"

## 5.2 Four parts, no more

Strip any evaluation system to its skeleton, from a ten-line script to the harnesses at frontier labs, and the same four parts appear. A **dataset**: inputs, sometimes with expected outputs. A **task**: the thing being evaluated, as a function from input to output. A **scorer**: something that decides whether an output met the bar. A **report**: the aggregation you actually read.

The lab's first example makes this point by evaluating something deliberately dumb: a rule-based classifier, ten lines of Python, no model anywhere, fully offline. This is not a warm-up to be skipped; it is the load-bearing insight. The eval machinery does not know or care what the task is. Today it is a rule; in a later section it is an LLM call; in the series capstone it is an entire RAG pipeline. Everything you learn about scoring, metrics, bias, and noise applies unchanged as the task grows, because the task is just a function.

The rule-based baseline earns its keep a second way. When your LLM classifier scores 84%, the number floats free until you know what dumb rules score on the same data. If the answer is 79%, you have learned something important about your task, your dataset, and possibly your budget. Every eval deserves a baseline stupid enough to be embarrassing.

## 5.3 Grade with code until you can't

The current excitement around LLM-as-judge techniques obscures a plain fact: a large fraction of quality checks need no intelligence at all. Is the output valid JSON? Does it contain the required keys? Does it match the expected label? Is the number within tolerance? Is the format right? Code answers these instantly, deterministically, and for free, and a code scorer never has an off day.

The craft hiding in this humble layer is that choosing a scorer is really choosing what "correct" means, and that is a genuine decision with consequences. The output "Positive!" fails an exact match against "positive" and passes a contains check; which one you want depends on whether downstream code parses the label or a human reads it. Too strict, and your eval punishes healthy outputs for trivia (this series' own capstone once docked a model heavily for citing sources in a grouped format the parser rejected; the citations were real, the parser was narrow, and the metric libeled the model). Too lenient, and garbage passes. Writing down what correct means, precisely enough for code to check, is often the moment a team discovers it never actually agreed on the requirements.

So the working rule: reach for code first, and spend model-graded judgment only on qualities code genuinely cannot reach.

## 5.4 The dataset is the product

Here is where this chapter diverges from what you might expect. The glamorous parts of evaluation are the scorers and judges. The part that determines whether your eval is worth anything is the dataset, and it is chronically underinvested everywhere.

An eval can only detect failures its dataset contains. If your golden set is ten easy, similar, well-phrased inputs, your eval will report excellence forever while users suffer, because users do not send well-phrased inputs. The dataset is your eval's imagination, and most datasets imagine too little.

Three practices raise the ceiling. First, smallness done well: ten hand-checked, representative, deliberately varied examples beat a thousand sloppy ones, and they are cheap enough that you will actually maintain them. Second, adversarial coverage: include the sarcastic review, the ambiguous ticket, the question whose answer is not in the corpus, because those are what production sends. Third, and most valuable over time: mine real failures. When production surfaces a case your system botched, fix it and add the case to the golden set, where it stands guard forever against that regression returning. A dataset built this way becomes a fossil record of every hard lesson the team has learned, which is precisely what you want your tests to be.

One distinction organizes everything else in this dive: **reference-based** evaluation has a known expected output to compare against (precise, but someone must produce the labels), while **reference-free** evaluation judges an output on its own merits with no answer key (no labeling cost, fuzzier signal). Most teams need both, and knowing which you are doing tells you which failure modes to worry about.

## 5.5 Metrics: when the headline number lies

Scores aggregate into metrics, and the first metric everyone reaches for, accuracy, has a famous failure mode. If 95% of your support tickets are not urgent, a classifier that answers "not urgent" every single time scores 95% accuracy while being precisely useless, since its whole job was catching the urgent ones. Skewed classes make accuracy a vanity metric.

The repair kit is a century old, borrowed from statistics and information retrieval. **Precision** asks: of the items you flagged, how many were right? **Recall** asks: of the items you should have flagged, how many did you catch? They pull in opposite directions (flag everything and recall is perfect while precision collapses; flag nothing and vice versa), and **F1** balances them. The practical value is diagnostic: precision and recall distinguish false alarms from misses, which are different diseases with different cures and, usually, very different costs. A cancer screen and a spam filter both classify, but they should not balance those errors the same way, and deciding the balance is a product decision, not a math one.

**pass@k** covers tasks where you sample several attempts and success means any of them passing, the standard regime for code generation. And **confidence intervals** get their own section below, because they are where most eval self-deception lives.

## 5.6 The judge that is also a defendant

Some qualities cannot be code-checked. Is this answer helpful? Is this summary faithful to the source? Is this explanation clear? For decades the only grader for such questions was a human, which is why evaluation of this kind was rare: humans are slow and expensive at scale.

Around 2023 the field noticed that the models had become good enough to grade each other, and **LLM-as-judge** went from oxymoron to standard practice almost overnight. Research projects like MT-Bench and the Chatbot Arena leaderboard demonstrated that a strong model's judgments correlate well with human preferences, and suddenly every team could afford a grader that reads every output. It is genuinely one of the most useful techniques in this book.

It is also a technique whose failure modes are your responsibility now, because the judge is a language model, which means the defendant and the judge share a species and its flaws.

**Position bias** is the notorious one: shown two answers, judges systematically favor whichever came first (some models, whichever came last). The lab has you measure this rather than take it on faith, and the mitigation doubles as the test: judge each pair in both orders and only count verdicts that agree both ways. **Verbosity bias**: longer answers score better than their content merits. **Self-preference**: models rate their own style above others'. And beneath the named biases sits something more basic: the judge applies its own interpretation of your rubric, which may not be yours.

This repo hit that last one in the wild, and kept the specimen (the full story is in [AUTHORING-LESSONS.md](../AUTHORING-LESSONS.md)). A faithfulness judge was asked whether answers stayed grounded in provided context. One test answer said a plan does not include SSO; the context never mentioned SSO at all. The rubric's intent: silence is not evidence, so that claim is ungrounded. The judge's reading: "no SSO" is a reasonable inference from silence, verdict faithful, five out of five, reproducibly, at temperature zero. Not a bug you can fix with a retry; a genuine disagreement about what "grounded" means, discovered only because a human looked at the individual verdicts. Hence the standing rule: calibrate a judge against a handful of human labels before trusting its numbers, and spot-check it forever after. A judge is measurement infrastructure, and measurement infrastructure gets audited.

Two more rules complete the toolkit. Prefer **pairwise** comparison over absolute scoring: models are unreliable at "is this a 3 or a 4?" and much steadier at "which of these two is better?", so the standard way to compare two systems is a pairwise win rate. And whatever the format, the **rubric is the most important sentence in the eval**: the lab shows the same two prompts trading victory when the judge's instruction flips from "most helpful" to "most concise." Whoever writes the rubric decides the winner, which is exactly why the rubric belongs in version control next to the code.

Finally, a rule this series' authoring principles state as law: when comparing two systems with a judge, hold the judge constant. Swap the system and the judge between runs and a score delta could be either one moving; the comparison means nothing. And never let a model grade its own answers; that is not evaluation, it is a performance review written by the employee.

## 5.7 One run is an anecdote

Run the same eval twice above temperature zero and you get two different numbers. This unsettles people the first time, and the lab makes you watch it happen on purpose, because the alternative is learning it from a shipped regression.

The statistics involved are older than computing and mercifully light. A single eval score is a **point estimate**, a sample from a distribution of scores that eval could produce. The honest report is a mean with a **confidence interval**: run it several times, report the center and the wobble. And the question that actually matters, "is prompt B really better than A, or is this noise?", is answered by whether the difference clears the combined wobble, which the lab's `compare()` implements.

The capstone forces the uncomfortable version of this lesson. On its tiny ten-example datasets, the confidence interval on a difference between two runs is enormous, often plus or minus forty points, and the tool reports it without cosmetics. That is not a defect in the tool. With ten binary-scored examples, one flipped case moves the score ten points, so you genuinely cannot distinguish a real improvement from luck, and an instrument that admits this is worth more than one that manufactures certainty. The margin shrinks with more examples and more runs, roughly with the square root of the sample size, and there is no other honest way to shrink it.

The industry context makes this sharper. Public model benchmarks routinely tout gains of a point or two, sample sizes are not always what they should be, and benchmark questions have a way of leaking into training data, a contamination problem the field wrestles with continuously. There is also the older trap with a name: Goodhart's law, "when a measure becomes a target, it ceases to be a good measure." Optimize hard against any fixed eval and you will eventually be optimizing the eval rather than the quality it was a proxy for. The defenses are unglamorous: rotate in fresh cases, mine production for new failures, and remember the number stands for something it is not identical to.

## 5.8 Beyond the single string

Four extensions cover the cases a basic string-scoring loop misses, and each introduces an idea bigger than its example.

**Trajectory evaluation** applies when the system under test is an agent (Chapter 6). A correct final answer can hide a broken process: the lucky guess, the forbidden tool call, the nine-step wander through a two-step task. So you grade the trace, not just the destination: did it use the required tool, avoid the forbidden ones, stay within its step budget, and land the answer? Process metrics catch failures that outcome metrics miss, an idea quality engineering has known for decades.

**Inter-annotator agreement** confronts the fact that your "ground truth" comes from humans, and humans disagree. Cohen's kappa measures agreement corrected for chance, and a low kappa is a serious finding: it means your gold labels are noisy, every score computed against them inherits that noise, and no judge can be calibrated against a target that will not hold still. Sometimes the honest fix is not better annotators but a clearer task definition, and discovering that your task was ambiguous is one of evaluation's most valuable outputs.

**Online evaluation** closes the gap between "passed our offline suite" and "real users are better off," which are related but not identical claims. The mechanism is the classic A/B test: split live traffic, compare an outcome you care about, and ship only if the gap clears the noise and no guardrail metric (latency, cost, refusal rate) regressed. The guardrail clause deserves emphasis, because a change that wins the headline metric while quietly doubling latency is a loss wearing a medal.

**Faithfulness** is the eval every RAG system from Chapter 4 needs and correctness-only evals miss: a fluent, plausible, even factually true answer that asserts something the retrieved context never said is a hallucination in the sense that matters, because the system's promise was answers grounded in these documents. A faithfulness judge is reference-free (it needs only the context and the answer, no gold label), which makes it deployable on live traffic where no answer key exists. That property becomes the backbone of the Observability dive.

## 5.9 The gate: where evals grow teeth

Everything above produces numbers. The capstone gives the numbers authority, and it is the difference between having evals and having an evaluation practice.

The tool runs a suite, saves a baseline, diffs new runs against it (using `compare()`, so it only cries regression when a change clears the noise), and, decisively, exits nonzero when quality drops below a threshold. That exit code is the whole trick. It means a CI system can block a merge on it, which converts evaluation from a virtue people are encouraged to practice into a fact about what can reach production. Nobody remembers to run quality checks under deadline; gates do not need to remember.

The endpoint of this road has a name, eval-driven development, and it is deliberately modeled on test-driven development: the eval suite passing is the definition of done for any prompt or model change. Teams that work this way move faster, not slower, for the same reason well-tested codebases move faster: they can change things without fear, because the net is real. When a new model releases, they rerun the suite and know by lunch whether to upgrade. Teams without evals greet each model release with a fresh round of vibes.

## 5.10 Where this chapter leaves you

This dive sits in the middle of the core path, and it faces both directions. Backward: the prompt comparisons of Chapter 3 and the retrieval knobs of Chapter 4 were all "guess until you measure," and now you can measure them. Forward: agents (Chapter 6) will need their trajectories graded, guardrails (Chapter 7) are evals run at request time against hostile input, production (Chapter 8) wires the gate into the release path, and observability (Chapter 16) runs the sampled judge over live traffic and watches the trend. Every remaining chapter leans on this one, which is why it is core rather than bonus.

You leave with a small framework you built yourself (a runner, scorers, judges, metrics, a gate; the runner is about ten lines, because an eval really is that simple) and, more durably, with three reflexes. Baseline everything, including against dumb rules. Distrust single numbers: ask for the interval, the base rate, and what the judge was actually instructed. And when a metric moves, open the failing cases before you write the conclusion, because the real story is usually narrower, and more useful, than the headline.

---

*Lab manual: [README.md](README.md) · Exercises: [EXERCISES.md](EXERCISES.md) · Previous: [RAG](../rag-deep-dive/TEXTBOOK.md) · Next: [Agents](../agents-deep-dive/TEXTBOOK.md)*
