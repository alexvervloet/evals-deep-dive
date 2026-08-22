"""
Example 14: decision statistics: evidence before a release (offline, no key).

A candidate and control usually answer the same eval cases. That pairing is
valuable evidence: a hard case is hard for both systems, so compare each case to
itself instead of pretending the two score lists came from unrelated samples.

This lesson also separates questions that a single “significant?” flag collapses:

1. Is the observed lift distinguishable from zero?
2. Is the entire plausible lift large enough to matter to the product?
3. Could the experiment detect the effect the policy cares about?
4. How much family error did multiple metrics and repeated looks spend?

The policy below is declared before outcomes are generated. The simulator returns
only observable paired scores; the decision functions never receive its hidden
capability setting or an expected verdict.

Predict before running: will a precisely measured +2-point lift ship when product
policy requires at least +3 points? At which planned look will a stronger candidate
clear that practical boundary?

Run it:

    python examples/14_decision_statistics.py
"""

import os
import random
import sys
from statistics import stdev

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import evals


# Release policy: fixed before either candidate's outcomes are generated.
MINIMUM_PRACTICAL_EFFECT = 0.03
FAMILY_ALPHA = 0.05
TARGET_POWER = 0.80
PLANNED_METRICS = 4
PLANNED_LOOKS = (100, 250, 500, 800)


def simulated_eval_scores(
    pairs: int, *, capability_lift: float, seed: int
) -> tuple[list[float], list[float]]:
    """Generate observable rubric scores from shared case difficulty.

    Difficulty changes both systems' behavior; independent response noise changes
    each score. ``capability_lift`` changes the candidate process that produces the
    observations. It is not returned as a label and never enters the statistical
    decision, matching a real experiment where the true effect is unknown.
    """

    rng = random.Random(seed)
    control: list[float] = []
    candidate: list[float] = []
    for _ in range(pairs):
        difficulty = rng.betavariate(2.0, 2.0)
        shared_quality = 0.84 - 0.44 * difficulty
        control.append(
            min(1.0, max(0.0, shared_quality + rng.gauss(0.0, 0.05)))
        )
        candidate.append(
            min(
                1.0,
                max(
                    0.0,
                    shared_quality + capability_lift + rng.gauss(0.0, 0.05),
                ),
            )
        )
    return control, candidate


def percentage_points(value: float) -> str:
    """Render a score difference in explicitly named percentage-point units."""

    return f"{value * 100:+.2f} pp"


print("Predeclared decision policy")
print(f"  minimum practical lift: {percentage_points(MINIMUM_PRACTICAL_EFFECT)}")
print(f"  family alpha:           {FAMILY_ALPHA:.3f}")
print(f"  target power:           {TARGET_POWER:.0%}")
print(f"  metrics × looks:        {PLANNED_METRICS} × {len(PLANNED_LOOKS)}\n")

# A tiny but consistent improvement: enough pairs can make it statistically clear
# even when it remains too small to justify release cost and risk.
control, tiny_candidate = simulated_eval_scores(
    1_200, capability_lift=0.02, seed=14
)
metric_alpha = evals.bonferroni_alpha(FAMILY_ALPHA, PLANNED_METRICS)
paired = evals.paired_bootstrap(
    control,
    tiny_candidate,
    confidence=1.0 - metric_alpha,
    resamples=5_000,
    seed=14,
)
evidence = evals.classify_effect(
    paired, minimum_practical_effect=MINIMUM_PRACTICAL_EFFECT
)
unpaired = evals.compare(control, tiny_candidate)

print("1. Same cases: preserve the pair")
print(
    "  unpaired normal approximation: "
    f"{percentage_points(unpaired['diff'])} ± "
    f"{unpaired['margin'] * 100:.2f} pp (95%)"
)
print(
    f"  paired bootstrap interval:      {percentage_points(paired.estimate)} "
    f"[{percentage_points(paired.lower)}, {percentage_points(paired.upper)}] "
    f"({paired.confidence:.2%}, adjusted across {PLANNED_METRICS} metrics)"
)
print(f"  evidence state: {evidence.value}")
print(
    "  decision: HOLD. The interval clears zero, but not the predeclared "
    "+3.00 pp practical boundary.\n"
)

# Planning consumes pilot variance, not the final candidate estimate. Using the
# full result to choose sample size would turn a prospective guarantee into a
# retrospective story.
pilot_differences = evals.paired_differences(control[:100], tiny_candidate[:100])
pilot_std = stdev(pilot_differences)
planned_pairs = evals.required_sample_size(
    MINIMUM_PRACTICAL_EFFECT,
    pilot_std,
    family_alpha=FAMILY_ALPHA,
    power=TARGET_POWER,
    comparisons=PLANNED_METRICS,
    looks=len(PLANNED_LOOKS),
)
first_look_mde = evals.minimum_detectable_effect(
    PLANNED_LOOKS[0],
    pilot_std,
    family_alpha=FAMILY_ALPHA,
    power=TARGET_POWER,
    comparisons=PLANNED_METRICS,
    looks=len(PLANNED_LOOKS),
)

print("2. Power is a plan, not a post-hoc excuse")
print(f"  pilot paired-difference SD: {pilot_std * 100:.2f} pp")
print(
    f"  approximate MDE at {PLANNED_LOOKS[0]} pairs: "
    f"{percentage_points(first_look_mde)}"
)
print(
    f"  pairs planned for {percentage_points(MINIMUM_PRACTICAL_EFFECT)} "
    f"at {TARGET_POWER:.0%} power: {planned_pairs}\n"
)

naive_family_risk = evals.independent_familywise_risk(0.05, PLANNED_METRICS)
look_alpha = evals.bonferroni_alpha(
    FAMILY_ALPHA, PLANNED_METRICS * len(PLANNED_LOOKS)
)
print("3. Every metric and peek spends the same error budget")
print(
    f"  {PLANNED_METRICS} independent null tests at alpha=.05 would have an "
    f"{naive_family_risk:.1%} chance of at least one false positive."
)
print(
    "  conservative planned allocation: "
    f"alpha={look_alpha:.6f} per metric-look decision\n"
)

# A separate, stronger candidate demonstrates a planned sequential campaign. The
# experiment may stop early only at one of the declared look sizes and only when
# the full interval clears the practical boundary.
sequential_control, strong_candidate = simulated_eval_scores(
    PLANNED_LOOKS[-1], capability_lift=0.05, seed=29
)
looks = evals.sequential_paired_test(
    sequential_control,
    strong_candidate,
    look_sizes=PLANNED_LOOKS,
    minimum_practical_effect=MINIMUM_PRACTICAL_EFFECT,
    family_alpha=FAMILY_ALPHA,
    comparisons=PLANNED_METRICS,
)

print("4. Predeclared sequential looks")
for look in looks:
    print(
        f"  n={look.pairs:>3}: {percentage_points(look.estimate)} "
        f"[{percentage_points(look.lower)}, {percentage_points(look.upper)}] "
        f"-> {look.status.value}"
    )

terminal = looks[-1]
if terminal.status is evals.SequentialStatus.STOP_FOR_BENEFIT:
    print(
        "\nPrimary-metric result: the candidate earned an early stop for practical "
        "benefit. This is not release authorization by itself: the other three "
        "predeclared metrics still have to clear their guardrails."
    )
else:
    print(
        "\nPrimary-metric result: HOLD. The planned looks ended without evidence "
        "that clears the practical boundary."
    )

print(
    "\nTakeaway: pair the same cases, plan sensitivity before collecting the final "
    "sample, spend one error budget across every metric and look, and require the "
    "whole interval—not just a point estimate—to clear the effect worth shipping."
)
