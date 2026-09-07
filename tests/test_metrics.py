"""Regression tests for the aggregate metrics.

The length-mismatch cases are the point. `zip` truncates to the shorter list, so
before these guards a run that dropped predictions scored *higher* than one that
kept them, and a half-empty run could report a perfect 100%.
"""

import unittest

from evals.metrics import accuracy, pass_rate, precision_recall_f1


class TestAccuracy(unittest.TestCase):
    def test_counts_every_case(self):
        self.assertAlmostEqual(accuracy(["a", "b", "c", "d"], ["a", "b", "x", "y"]), 0.5)

    def test_empty_is_zero(self):
        self.assertEqual(accuracy([], []), 0.0)

    def test_missing_predictions_are_rejected_not_dropped(self):
        """Four correct predictions against eight labels is not 100%."""
        with self.assertRaises(ValueError):
            accuracy(["a", "b", "c", "d"], ["a", "b", "c", "d", "e", "f", "g", "h"])

    def test_extra_predictions_are_rejected(self):
        with self.assertRaises(ValueError):
            accuracy(["a", "b", "c"], ["a", "b"])

    def test_filling_a_missing_prediction_scores_it_wrong(self):
        """The documented fix: pad the gap, and the case counts against you."""
        self.assertAlmostEqual(accuracy(["a", "b", None], ["a", "b", "c"]), 2 / 3)


class TestPrecisionRecallF1(unittest.TestCase):
    def test_known_confusion_matrix(self):
        prf = precision_recall_f1(
            ["spam", "ham", "ham", "spam"],
            ["spam", "ham", "spam", "ham"],
            positive_label="spam",
        )
        self.assertEqual((prf["tp"], prf["fp"], prf["fn"]), (1, 1, 1))
        self.assertAlmostEqual(prf["precision"], 0.5)
        self.assertAlmostEqual(prf["recall"], 0.5)
        self.assertAlmostEqual(prf["f1"], 0.5)

    def test_mismatched_lengths_are_rejected(self):
        with self.assertRaises(ValueError):
            precision_recall_f1(["spam"], ["spam", "ham"], positive_label="spam")


class TestPassRate(unittest.TestCase):
    def test_accepts_bools(self):
        self.assertAlmostEqual(pass_rate([True, True, False, False]), 0.5)

    def test_empty_is_zero(self):
        self.assertEqual(pass_rate([]), 0.0)


if __name__ == "__main__":
    unittest.main()
