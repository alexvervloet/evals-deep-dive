"""Turn paired eval outcomes into a predeclared release decision.

Aggregate scores hide the most useful experimental structure: control and
candidate usually answer the same cases. This module preserves those pairs,
resamples paired differences for an uncertainty interval, and keeps statistical
evidence separate from the minimum effect a product team considers worth shipping.

The ordering is the lesson. Define the practical threshold, family error budget,
metrics, power target, and interim looks before reading candidate outcomes; plan
sample size from pilot variance; then evaluate the fixed policy. Choosing any of
those controls after seeing the result spends error invisibly.

This is a stdlib teaching implementation, not a universal experiment service.
The percentile bootstrap is approximate, the power and sequential intervals use
a normal approximation, and Bonferroni spending is deliberately conservative.
Clustered users, adaptive traffic, rare outcomes, heavy tails, or regulated
decisions need a design validated for that data-generating process.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from statistics import NormalDist, mean, stdev


class EffectEvidence(str, Enum):
    """State what an interval supports without turning uncertainty into “no effect.”"""

    PRACTICAL_IMPROVEMENT = "practical_improvement"
    STATISTICAL_IMPROVEMENT_ONLY = "statistical_improvement_only"
    PRACTICALLY_EQUIVALENT = "practically_equivalent"
    INCONCLUSIVE = "inconclusive"
    STATISTICAL_REGRESSION_ONLY = "statistical_regression_only"
    PRACTICAL_REGRESSION = "practical_regression"


class SequentialStatus(str, Enum):
    """Describe whether a predeclared interim boundary ended the experiment."""

    CONTINUE = "continue"
    STOP_FOR_BENEFIT = "stop_for_benefit"
    STOP_FOR_HARM = "stop_for_harm"
    COMPLETE_INCONCLUSIVE = "complete_inconclusive"


@dataclass(frozen=True)
class PairedBootstrapInterval:
    """Percentile interval for the mean candidate-minus-control difference."""

    estimate: float
    lower: float
    upper: float
    confidence: float
    pairs: int
    resamples: int


@dataclass(frozen=True)
class SequentialLook:
    """Evidence available at one planned look in a sequential comparison."""

    pairs: int
    estimate: float
    lower: float
    upper: float
    look_alpha: float
    cumulative_family_alpha: float
    status: SequentialStatus


def _probability(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    result = float(value)
    if not 0.0 < result < 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
    return result


def _positive_integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _nonnegative(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def paired_differences(
    control: Sequence[float], candidate: Sequence[float]
) -> tuple[float, ...]:
    """Validate a caller-owned case join and return candidate minus control.

    Position represents case identity, so unequal lengths are rejected rather
    than truncated. Values may be continuous scores or binary booleans, but must
    be finite. A successful return proves only that the numeric pairs are usable;
    the caller still owns the stronger invariant that position ``i`` names the
    same case, user, and sampling unit in both arms.
    """

    if len(control) != len(candidate):
        raise ValueError("control and candidate must contain the same number of pairs")
    if len(control) < 2:
        raise ValueError("at least two paired observations are required")

    differences: list[float] = []
    for index in range(len(control)):
        control_raw = control[index]
        candidate_raw = candidate[index]
        if not isinstance(control_raw, (bool, int, float)) or not isinstance(
            candidate_raw, (bool, int, float)
        ):
            raise TypeError(f"pair {index} must contain numeric scores")
        control_value = float(control_raw)
        candidate_value = float(candidate_raw)
        if not math.isfinite(control_value) or not math.isfinite(candidate_value):
            raise ValueError(f"pair {index} must contain finite scores")
        differences.append(candidate_value - control_value)
    return tuple(differences)


def _percentile(sorted_values: Sequence[float], probability: float) -> float:
    """Linearly interpolate one percentile from an already sorted sample."""

    position = (len(sorted_values) - 1) * probability
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return sorted_values[lower_index]
    weight = position - lower_index
    return (
        sorted_values[lower_index] * (1.0 - weight)
        + sorted_values[upper_index] * weight
    )


def paired_bootstrap(
    control: Sequence[float],
    candidate: Sequence[float],
    *,
    confidence: float = 0.95,
    resamples: int = 10_000,
    seed: int = 0,
) -> PairedBootstrapInterval:
    """Estimate a percentile interval by resampling whole case pairs.

    The inputs are observed outcomes, never expected decisions. Resampling the
    candidate-minus-control differences preserves shared case difficulty; drawing
    the two arms independently would throw that information away and answer a
    different question. ``seed`` makes the approximation reproducible, not exact.
    Runtime is O(pairs * resamples); production systems should vectorize, cluster
    by the real sampling unit, or use an experiment library at scale.
    """

    confidence = _probability("confidence", confidence)
    resamples = _positive_integer("resamples", resamples)
    if resamples < 100:
        raise ValueError("resamples must be at least 100 for a meaningful interval")
    differences = paired_differences(control, candidate)
    pair_count = len(differences)
    rng = random.Random(seed)
    bootstrap_means = [
        sum(differences[rng.randrange(pair_count)] for _ in range(pair_count))
        / pair_count
        for _ in range(resamples)
    ]
    bootstrap_means.sort()
    tail = (1.0 - confidence) / 2.0
    return PairedBootstrapInterval(
        estimate=mean(differences),
        lower=_percentile(bootstrap_means, tail),
        upper=_percentile(bootstrap_means, 1.0 - tail),
        confidence=confidence,
        pairs=pair_count,
        resamples=resamples,
    )


def classify_effect(
    interval: PairedBootstrapInterval,
    *,
    minimum_practical_effect: float,
) -> EffectEvidence:
    """Classify what the full interval supports against a predeclared threshold.

    Statistical evidence means the interval excludes zero. Practical evidence is
    deliberately harder: the entire interval must clear the product's minimum
    worthwhile effect in the same direction. An interval wholly inside the
    practical-equivalence band supports equivalence; merely crossing zero remains
    inconclusive. The threshold must be policy input chosen before this interval.
    """

    threshold = _nonnegative("minimum_practical_effect", minimum_practical_effect)
    if not all(
        math.isfinite(value)
        for value in (interval.estimate, interval.lower, interval.upper)
    ):
        raise ValueError("interval estimate and bounds must be finite")
    if interval.lower > interval.upper:
        raise ValueError("interval lower bound must not exceed its upper bound")
    if interval.lower > threshold:
        return EffectEvidence.PRACTICAL_IMPROVEMENT
    if interval.upper < -threshold:
        return EffectEvidence.PRACTICAL_REGRESSION
    if interval.lower > 0.0:
        return EffectEvidence.STATISTICAL_IMPROVEMENT_ONLY
    if interval.upper < 0.0:
        return EffectEvidence.STATISTICAL_REGRESSION_ONLY
    if interval.lower >= -threshold and interval.upper <= threshold:
        return EffectEvidence.PRACTICALLY_EQUIVALENT
    return EffectEvidence.INCONCLUSIVE


def bonferroni_alpha(family_alpha: float, decisions: int) -> float:
    """Allocate one family error budget across predeclared decisions.

    Bonferroni controls family-wise error by the union bound and therefore does
    not require independent metrics or looks. Its price is conservatism: correlated
    metrics often permit more powerful procedures, but only when that procedure
    and family were chosen before inspecting results.
    """

    return _probability("family_alpha", family_alpha) / _positive_integer(
        "decisions", decisions
    )


def independent_familywise_risk(per_test_alpha: float, hypotheses: int) -> float:
    """Return P(at least one false positive) for independent null tests.

    This is an illustration of multiplicity, not a claim about correlated eval
    metrics. The exact risk for dependent tests differs; Bonferroni's upper bound
    remains valid without the independence assumption.
    """

    alpha = _probability("per_test_alpha", per_test_alpha)
    count = _positive_integer("hypotheses", hypotheses)
    return 1.0 - (1.0 - alpha) ** count


def _planning_quantiles(
    *, family_alpha: float, power: float, comparisons: int, looks: int
) -> tuple[float, float, float]:
    family_alpha = _probability("family_alpha", family_alpha)
    power = _probability("power", power)
    if power <= 0.5:
        raise ValueError("power must be greater than 0.5")
    comparisons = _positive_integer("comparisons", comparisons)
    looks = _positive_integer("looks", looks)
    decision_alpha = bonferroni_alpha(family_alpha, comparisons * looks)
    normal = NormalDist()
    return (
        normal.inv_cdf(1.0 - decision_alpha / 2.0),
        normal.inv_cdf(power),
        decision_alpha,
    )


def minimum_detectable_effect(
    pairs: int,
    difference_std: float,
    *,
    family_alpha: float = 0.05,
    power: float = 0.80,
    comparisons: int = 1,
    looks: int = 1,
) -> float:
    """Plan the smallest mean paired effect detectable under a normal model.

    ``difference_std`` comes from prior or pilot *paired differences*, not the two
    arm standard deviations. The returned MDE is prospective sensitivity at the
    requested family alpha and power after conservative adjustment for every
    planned comparison and look. It is not a promise that a future bootstrap
    interval will have exactly that coverage or a retrospective interpretation of
    a result already observed.
    """

    pairs = _positive_integer("pairs", pairs)
    if pairs < 2:
        raise ValueError("pairs must be at least two")
    difference_std = _nonnegative("difference_std", difference_std)
    z_alpha, z_power, _ = _planning_quantiles(
        family_alpha=family_alpha,
        power=power,
        comparisons=comparisons,
        looks=looks,
    )
    return (z_alpha + z_power) * difference_std / math.sqrt(pairs)


def required_sample_size(
    effect: float,
    difference_std: float,
    *,
    family_alpha: float = 0.05,
    power: float = 0.80,
    comparisons: int = 1,
    looks: int = 1,
) -> int:
    """Plan paired sample size for a predeclared effect under a normal model.

    This algebraically inverts :func:`minimum_detectable_effect` and rounds up so
    the planned sample is never smaller than the approximation requests. The
    variance estimate must come from representative pilot or historical pairs;
    optimistic pilot variance creates an underpowered experiment.
    """

    effect = _nonnegative("effect", effect)
    if effect == 0.0:
        raise ValueError("effect must be positive")
    difference_std = _nonnegative("difference_std", difference_std)
    z_alpha, z_power, _ = _planning_quantiles(
        family_alpha=family_alpha,
        power=power,
        comparisons=comparisons,
        looks=looks,
    )
    planned = math.ceil(((z_alpha + z_power) * difference_std / effect) ** 2)
    return max(2, planned)


def _normal_interval(
    differences: Sequence[float], confidence: float
) -> tuple[float, float, float]:
    estimate = mean(differences)
    standard_error = stdev(differences) / math.sqrt(len(differences))
    critical = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    margin = critical * standard_error
    return estimate, estimate - margin, estimate + margin


def sequential_paired_test(
    control: Sequence[float],
    candidate: Sequence[float],
    *,
    look_sizes: Sequence[int],
    minimum_practical_effect: float,
    family_alpha: float = 0.05,
    comparisons: int = 1,
) -> tuple[SequentialLook, ...]:
    """Run a conservative group-sequential test at predeclared sample sizes.

    The function validates the complete paired dataset first, then visits strictly
    increasing ``look_sizes``. It splits family alpha equally across all metrics
    and looks with Bonferroni, constructs a two-sided normal interval for the mean
    paired difference, and stops only when that interval clears the practical
    benefit or harm boundary. Otherwise the final state is explicitly
    ``complete_inconclusive``—absence of evidence is never reported as no effect.

    This does not authorize arbitrary peeking: changing the schedule after seeing
    outcomes invalidates the declared error budget. Normal intervals are a readable
    large-sample approximation; production sequential testing should use a design
    such as a validated alpha-spending boundary for the actual outcome model.
    """

    differences = paired_differences(control, candidate)
    threshold = _nonnegative("minimum_practical_effect", minimum_practical_effect)
    comparisons = _positive_integer("comparisons", comparisons)
    family_alpha = _probability("family_alpha", family_alpha)

    looks = tuple(look_sizes)
    if not looks:
        raise ValueError("look_sizes must contain at least one planned look")
    for look in looks:
        _positive_integer("look size", look)
    if tuple(sorted(set(looks))) != looks:
        raise ValueError("look_sizes must be unique and strictly increasing")
    if looks[0] < 2:
        raise ValueError("the first look must contain at least two pairs")
    if looks[-1] > len(differences):
        raise ValueError("a look cannot exceed the available paired observations")

    look_alpha = bonferroni_alpha(family_alpha, comparisons * len(looks))
    confidence = 1.0 - look_alpha
    results: list[SequentialLook] = []
    for index, pair_count in enumerate(looks, start=1):
        estimate, lower, upper = _normal_interval(
            differences[:pair_count], confidence
        )
        if lower > threshold:
            status = SequentialStatus.STOP_FOR_BENEFIT
        elif upper < -threshold:
            status = SequentialStatus.STOP_FOR_HARM
        elif index == len(looks):
            status = SequentialStatus.COMPLETE_INCONCLUSIVE
        else:
            status = SequentialStatus.CONTINUE
        results.append(
            SequentialLook(
                pairs=pair_count,
                estimate=estimate,
                lower=lower,
                upper=upper,
                look_alpha=look_alpha,
                cumulative_family_alpha=index * comparisons * look_alpha,
                status=status,
            )
        )
        if status in {
            SequentialStatus.STOP_FOR_BENEFIT,
            SequentialStatus.STOP_FOR_HARM,
        }:
            break
    return tuple(results)
