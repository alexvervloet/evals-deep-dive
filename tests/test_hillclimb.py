"""Pin the offline hill-climbing result the README and example 15 describe."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evals import hillclimb  # noqa: E402
from evals.dataset import load_jsonl  # noqa: E402

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets", "tickets.jsonl")


def _splits():
    examples = load_jsonl(DATA)
    return {s: [e for e in examples if e.metadata["split"] == s] for s in ("train", "validation", "test")}


def _climb(policy):
    s = _splits()
    run = hillclimb.climb(hillclimb.Prompt(), hillclimb.proposed_edits(), s["train"], s["validation"],
                          hillclimb.offline_model, policy, margin=1 / len(s["validation"]))
    return run, s


class TestSplits(unittest.TestCase):
    def test_frozen_split_sizes(self):
        self.assertEqual({k: len(v) for k, v in _splits().items()}, {"train": 24, "validation": 16, "test": 20})

    def test_no_ticket_appears_in_two_splits(self):
        s = _splits()
        seen = [e.input for split in s.values() for e in split]
        self.assertEqual(len(seen), len(set(seen)))


class TestClimb(unittest.TestCase):
    def test_train_only_keeps_both_traps_and_overstates(self):
        run, s = _climb("train")
        self.assertTrue(all(step.accepted for step in run.steps))
        self.assertEqual(hillclimb.score(run.final, s["train"], hillclimb.offline_model)[0], 1.0)
        self.assertEqual(hillclimb.score(run.final, s["test"], hillclimb.offline_model)[0], 0.75)

    def test_validation_gate_drops_both_traps(self):
        run, s = _climb("validation")
        self.assertEqual([step.accepted for step in run.steps], [True, True, True, False, False])
        self.assertEqual(hillclimb.score(run.final, s["test"], hillclimb.offline_model)[0], 0.80)

    def test_unknown_policy_is_refused(self):
        s = _splits()
        with self.assertRaises(ValueError):
            hillclimb.climb(hillclimb.Prompt(), [], s["train"], s["validation"], hillclimb.offline_model, "test", 0.0)


if __name__ == "__main__":
    unittest.main()
