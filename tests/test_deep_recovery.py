"""Tests for the K20 base + deep recovery merge (synthetic data only)."""

import unittest

import numpy as np
import pandas as pd

from src.pipeline.deep_recovery import base_predictions, evaluate, merge_deep, rule_masks


def setup():
    truth = {"S1-a": {"S2-x", "S2-d"}, "S1-b": {"S3-y"}, "S1-c": set()}
    k20 = pd.DataFrame({"s1_id": ["S1-a", "S1-b", "S1-c"], "cand_id": ["S2-x", "S3-y", "S2-q"],
                        "rank": [1, 1, 1], "label": [1, 1, 0], "prob": [0.9, 0.8, 0.2]})
    deep = pd.DataFrame({"s1_id": ["S1-a", "S1-b", "S1-c", "S1-c", "S1-b"],
                         "cand_id": ["S2-d", "S2-x", "S2-e", "S2-d", "S2-f"],
                         "deep_rank": [21, 22, 21, 22, 23], "prob": [0.95, 0.99, 0.30, 0.95, 0.10],
                         "label": [1, 0, 0, 0, 0]})
    s1 = pd.DataFrame({"entity_id": ["S1-a", "S1-b", "S1-c"], "country": ["India", "US", "US"], "city": ""})
    return truth, k20, deep, s1


class MergeTests(unittest.TestCase):
    def test_base_priority_deep_exclusivity_and_threshold(self):
        truth, k20, deep, s1 = setup()
        base, owner = base_predictions(k20, 0.5, truth)
        self.assertEqual(base, {"S1-a": {"S2-x"}, "S1-b": {"S3-y"}, "S1-c": set()})
        pred, added = merge_deep(base, owner, deep, deep["prob"].to_numpy() >= 0.9)
        # S2-x is S1-a's base match: never given to S1-b even at 0.99.
        # S2-d is claimed by S1-a and S1-c at 0.95: tie -> smallest s1_id (S1-a).
        self.assertEqual(pred, {"S1-a": {"S2-x", "S2-d"}, "S1-b": {"S3-y"}, "S1-c": set()})
        self.assertEqual(added[["s1_id", "cand_id"]].values.tolist(), [["S1-a", "S2-d"]])

    def test_base_matches_never_removed(self):
        truth, k20, deep, s1 = setup()
        base, owner = base_predictions(k20, 0.5, truth)
        pred, _ = merge_deep(base, owner, deep, np.ones(len(deep), dtype=bool))
        for s, v in base.items():
            self.assertTrue(v <= pred[s])
        seen = [c for v in pred.values() for c in v]
        self.assertEqual(len(seen), len(set(seen)))  # still globally exclusive

    def test_highest_probability_wins_deep_conflict(self):
        truth, k20, deep, s1 = setup()
        base, owner = base_predictions(k20, 0.5, truth)
        deep.loc[3, "prob"] = 0.97  # S1-c now claims S2-d more strongly than S1-a (0.95)
        pred, _ = merge_deep(base, owner, deep, deep["prob"].to_numpy() >= 0.9)
        self.assertEqual(pred["S1-c"], {"S2-d"})
        self.assertEqual(pred["S1-a"], {"S2-x"})

    def test_evaluate_counts(self):
        truth, k20, deep, s1 = setup()
        base, owner = base_predictions(k20, 0.5, truth)
        pred, added = merge_deep(base, owner, deep, deep["prob"].to_numpy() >= 0.9)
        r = evaluate(pred, base, truth, s1, added)
        self.assertEqual((r["added"], r["added_tp"], r["added_fp"]), (1, 1, 0))
        self.assertEqual((r["s1_improved"], r["s1_harmed"]), (1, 0))
        self.assertAlmostEqual(r["f05"], 1.0)
        self.assertEqual(r["singletons_empty"], 1.0)

    def test_rule_masks(self):
        f = pd.DataFrame({"prob": [0.9, 0.9, 0.9, 0.5], "num_best_ratio": [0.2, 1.0, -1.0, 1.0],
                          "name_tsort": [0.9, 0.3, 0.9, 0.9], "ph_jacc": [0.0, 0.0, 0.0, 0.0],
                          "cand_addr_missing": [False, False, True, False], "addr_word_jacc": [0.5, 0.5, 0.0, 0.5],
                          "deep_is_best": [True, False, False, True], "deep_second_prob": [0.5, 0.85, 0.2, 0.1]})
        m = rule_masks(f, 0.85)
        self.assertEqual(m["A prob"].tolist(), [True, True, True, False])
        self.assertEqual(m["B prob, no strong number conflict"].tolist(), [False, True, True, False])
        self.assertEqual(m["C prob, strong name/transliteration evidence"].tolist(), [True, False, True, False])
        self.assertEqual(m["D prob, beats the S1's other deep candidates"].tolist(), [True, False, True, False])
        self.assertEqual(m["E prob, name evidence and acceptable address"].tolist(), [True, False, True, False])


if __name__ == "__main__":
    unittest.main()
