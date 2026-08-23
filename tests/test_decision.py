"""Regression tests for paired release-decision statistics."""

import math
import unittest

from evals.decision import (
    EffectEvidence,
    PairedBootstrapInterval,
    SequentialStatus,
    bonferroni_alpha,
    classify_effect,
    independent_familywise_risk,
    minimum_detectable_effect,
    paired_bootstrap,
    paired_differences,
    required_sample_size,
    sequential_paired_test,
)


def _interval(lower: float, upper: float) -> PairedBootstrapInterval:
    return PairedBootstrapInterval(
        estimate=(lower + upper) / 2.0,
        lower=lower,
        upper=upper,
        confidence=0.95,
        pairs=100,
        resamples=1_000,
    )


class PairedBootstrapTests(unittest.TestCase):
    def test_constant_paired_effect_has_a_zero_width_interval(self) -> None:
        control = [0.0, 0.25, 0.5, 0.75]
        candidate = [0.125, 0.375, 0.625, 0.875]

        interval = paired_bootstrap(control, candidate, resamples=500, seed=7)

        self.assertEqual(interval.estimate, 0.125)
        self.assertEqual(interval.lower, 0.125)
        self.assertEqual(interval.upper, 0.125)

    def test_a_shared_shift_leaves_the_paired_interval_unchanged(self) -> None:
        control = [0.0, 0.25, 0.5, 0.75]
        candidate = [0.125, 0.5, 0.625, 1.0]
        original = paired_bootstrap(control, candidate, resamples=500, seed=11)
        shifted = paired_bootstrap(
            [value + 16.0 for value in control],
            [value + 16.0 for value in candidate],
            resamples=500,
            seed=11,
        )

        self.assertEqual(original, shifted)

    def test_pair_validation_rejects_truncation_and_invalid_scores(self) -> None:
        with self.assertRaisesRegex(ValueError, "same number"):
            paired_differences([0.1, 0.2], [0.1])
        with self.assertRaisesRegex(ValueError, "at least two"):
            paired_differences([0.1], [0.2])
        with self.assertRaisesRegex(ValueError, "finite"):
            paired_differences([0.1, math.nan], [0.2, 0.3])
        with self.assertRaisesRegex(TypeError, "numeric"):
            paired_differences([0.1, "0.2"], [0.2, 0.3])  # type: ignore[list-item]

    def test_bootstrap_refuses_too_few_resamples(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least 100"):
            paired_bootstrap([0.0, 0.0], [0.1, 0.1], resamples=99)


class EvidenceTests(unittest.TestCase):
    def test_every_effect_state_has_a_separating_interval(self) -> None:
        threshold = 0.03
        cases = {
            _interval(0.04, 0.06): EffectEvidence.PRACTICAL_IMPROVEMENT,
            _interval(0.01, 0.02): EffectEvidence.STATISTICAL_IMPROVEMENT_ONLY,
            _interval(-0.02, 0.02): EffectEvidence.PRACTICALLY_EQUIVALENT,
            _interval(-0.02, 0.04): EffectEvidence.INCONCLUSIVE,
            _interval(-0.02, -0.01): EffectEvidence.STATISTICAL_REGRESSION_ONLY,
            _interval(-0.06, -0.04): EffectEvidence.PRACTICAL_REGRESSION,
        }

        for interval, expected in cases.items():
            with self.subTest(expected=expected):
                self.assertIs(
                    classify_effect(
                        interval, minimum_practical_effect=threshold
                    ),
                    expected,
                )

    def test_invalid_interval_cannot_be_classified(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not exceed"):
            classify_effect(_interval(0.2, 0.1), minimum_practical_effect=0.03)
        with self.assertRaisesRegex(ValueError, "finite"):
            classify_effect(_interval(math.nan, 0.1), minimum_practical_effect=0.03)


class PlanningTests(unittest.TestCase):
    def test_family_risk_and_bonferroni_answer_different_questions(self) -> None:
        self.assertAlmostEqual(independent_familywise_risk(0.05, 20), 0.641514, places=6)
        self.assertEqual(bonferroni_alpha(0.05, 20), 0.0025)

    def test_quadrupling_pairs_halves_the_mde(self) -> None:
        first = minimum_detectable_effect(100, 0.2)
        second = minimum_detectable_effect(400, 0.2)
        self.assertAlmostEqual(second, first / 2.0)

    def test_each_multiplicity_dimension_raises_the_planned_mde(self) -> None:
        fixed = minimum_detectable_effect(400, 0.2)
        more_metrics = minimum_detectable_effect(400, 0.2, comparisons=3)
        more_looks = minimum_detectable_effect(400, 0.2, looks=4)

        self.assertGreater(more_metrics, fixed)
        self.assertGreater(more_looks, fixed)

    def test_required_sample_rounds_up_to_the_requested_sensitivity(self) -> None:
        target = 0.04
        planned = required_sample_size(target, 0.2, comparisons=2, looks=3)

        self.assertLessEqual(
            minimum_detectable_effect(planned, 0.2, comparisons=2, looks=3),
            target,
        )
        self.assertGreater(
            minimum_detectable_effect(planned - 1, 0.2, comparisons=2, looks=3),
            target,
        )

    def test_planning_rejects_impossible_controls(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least two"):
            minimum_detectable_effect(1, 0.2)
        with self.assertRaisesRegex(ValueError, "greater than 0.5"):
            minimum_detectable_effect(100, 0.2, power=0.5)
        with self.assertRaisesRegex(ValueError, "positive"):
            required_sample_size(0.0, 0.2)


class SequentialTests(unittest.TestCase):
    def test_strong_benefit_stops_without_consuming_later_looks(self) -> None:
        control = [0.5] * 200
        candidate = [0.56 + (index % 5 - 2) * 0.002 for index in range(200)]

        looks = sequential_paired_test(
            control,
            candidate,
            look_sizes=(40, 100, 200),
            minimum_practical_effect=0.03,
        )

        self.assertEqual(len(looks), 1)
        self.assertIs(looks[0].status, SequentialStatus.STOP_FOR_BENEFIT)

    def test_noisy_zero_effect_finishes_inconclusive(self) -> None:
        control = [0.5] * 80
        candidate = [0.48 if index % 2 else 0.52 for index in range(80)]

        looks = sequential_paired_test(
            control,
            candidate,
            look_sizes=(30, 50, 80),
            minimum_practical_effect=0.03,
            comparisons=2,
        )

        self.assertEqual(
            [look.status for look in looks],
            [
                SequentialStatus.CONTINUE,
                SequentialStatus.CONTINUE,
                SequentialStatus.COMPLETE_INCONCLUSIVE,
            ],
        )
        self.assertAlmostEqual(looks[0].look_alpha, 0.05 / 6)
        self.assertAlmostEqual(looks[-1].cumulative_family_alpha, 0.05)

    def test_statistical_but_sub_practical_gain_does_not_stop_for_benefit(self) -> None:
        control = [0.5] * 200
        candidate = [0.52 + (index % 5 - 2) * 0.002 for index in range(200)]

        looks = sequential_paired_test(
            control,
            candidate,
            look_sizes=(40, 100, 200),
            minimum_practical_effect=0.03,
        )

        self.assertEqual(len(looks), 3)
        self.assertTrue(all(look.lower > 0.0 for look in looks))
        self.assertEqual(
            [look.status for look in looks],
            [
                SequentialStatus.CONTINUE,
                SequentialStatus.CONTINUE,
                SequentialStatus.COMPLETE_INCONCLUSIVE,
            ],
        )

    def test_strong_harm_has_a_distinct_terminal_state(self) -> None:
        control = [0.5] * 60
        candidate = [0.44 + (index % 3 - 1) * 0.002 for index in range(60)]

        looks = sequential_paired_test(
            control,
            candidate,
            look_sizes=(30, 60),
            minimum_practical_effect=0.03,
        )

        self.assertIs(looks[0].status, SequentialStatus.STOP_FOR_HARM)

    def test_look_schedule_must_be_predeclared_and_available(self) -> None:
        control = [0.5] * 10
        candidate = [0.5] * 10
        invalid_schedules = ((), (4, 4), (6, 4), (4, 11), (1, 4))

        for schedule in invalid_schedules:
            with self.subTest(schedule=schedule), self.assertRaises(ValueError):
                sequential_paired_test(
                    control,
                    candidate,
                    look_sizes=schedule,
                    minimum_practical_effect=0.03,
                )


if __name__ == "__main__":
    unittest.main()
