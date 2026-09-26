"""Tests for the matcher error-analysis helpers (synthetic data only)."""

import unittest

import numpy as np
import pandas as pd

from src.evaluation.matcher_errors import (breakdown, classify_errors, decide, missing_true_pairs, pair_categories,
                                           patterns, rank_bucket)


def synthetic():
    records = pd.DataFrame({
        "entity_id": ["S1-a", "S1-b", "S1-c", "S2-x", "S2-y", "S3-z", "S3-w"],
        "business_name": ["Pioneer Bakers", "Acme Tools", "Lonely Shop", "PIONEER BAKERS", "पायोनियर बेकर्स",
                          "Acme Tols", "Other"],
        "business_address": ["12 Oak St", "5 Main Rd", np.nan, "12 OAK ST", np.nan, "7 Main Rd", "9 Elm"],
        "country": ["India", "US", "US", "India", "India", "US", "US"],
    })
    pairs = pd.DataFrame({
        "s1_id": ["S1-a", "S1-a", "S1-b", "S1-b", "S1-c", "S1-c"],
        "cand_id": ["S2-x", "S2-y", "S3-z", "S2-x", "S3-w", "S3-z"],
        "rank": [1, 2, 1, 4, 1, 15],
        "label": [1, 0, 1, 0, 0, 0],
    })
    X = pd.DataFrame({
        "num_conflict": [0, 0, 1, 1, 0, 0], "first_num_eq": [1, -1, 0, 0, -1, -1],
        "name_tsort": [1.0, 0.1, 0.95, 0.3, 0.2, 0.1], "addr_tsort": [1.0, 0.0, 0.4, 0.6, 0.0, 0.1],
        "core_tsort": [1.0, 0.1, 0.95, 0.3, 0.2, 0.1], "name_tfidf": [1, 0, .9, .2, .1, .1],
        "addr_tfidf": [1, 0, .3, .5, 0, .1],
    })
    truth = {"S1-a": {"S2-x"}, "S1-b": {"S3-z"}, "S1-c": set()}
    claimed = {"S2-x", "S3-z", "S2-q"}
    return records, pairs, X, truth, claimed


class HelperTests(unittest.TestCase):
    def test_rank_bucket(self):
        self.assertEqual(list(rank_bucket([1, 2, 3, 4, 10, 11, 20, 21])),
                         ["1", "2", "3", "4-10", "4-10", "11-20", "11-20", ">20"])

    def test_classify_errors(self):
        label = np.array([1, 0, 1, 1, 0])
        prob = np.array([0.9, 0.8, 0.3, 0.9, 0.2])
        ex = np.array([0.9, 0.8, 0.3, 0.0, 0.2])  # 4th: above threshold but lost to exclusivity
        self.assertEqual(list(classify_errors(label, prob, ex, 0.5)), ["TP", "FP", "FN2", "FN3", "TN"])

    def test_decide_uses_exclusivity(self):
        pairs = pd.DataFrame({"s1_id": ["S1-a", "S1-b"], "cand_id": ["S2-x", "S2-x"]})
        acc, ex = decide(pairs, [0.9, 0.95], 0.5)
        self.assertEqual(acc.tolist(), [False, True])

    def test_missing_true_pairs(self):
        _, pairs, _, truth, _ = synthetic()
        truth = dict(truth, **{"S1-a": {"S2-x", "S3-q"}})
        self.assertEqual(missing_true_pairs(pairs, truth), [("S1-a", "S3-q")])

    def test_pair_categories(self):
        records, pairs, X, truth, claimed = synthetic()
        cat = pair_categories(pairs, X, records, truth, claimed)
        self.assertEqual(cat["rank_1"].tolist(), [True, False, True, False, True, False])
        self.assertEqual(cat["rank_11-20"].tolist(), [False] * 5 + [True])
        self.assertEqual(cat["India"].tolist(), [True, True, False, False, False, False])
        self.assertEqual(cat["singleton_s1"].tolist(), [False, False, False, False, True, True])
        self.assertEqual(cat["s1_address_missing"].tolist(), [False, False, False, False, True, True])
        self.assertEqual(cat["cand_address_missing"].tolist(), [False, True, False, False, False, False])
        self.assertEqual(cat["both_addresses"].tolist(), [True, False, True, True, False, False])
        # S2-x under S1-b is S1-a's true match; S3-z under S1-c is S1-b's
        self.assertEqual(cat["claimed_by_other_s1"].tolist(), [False, False, False, True, False, True])
        self.assertEqual(cat["multi_s1_candidate"].tolist(), [True, False, True, True, False, True])
        self.assertEqual(cat["nonlatin_candidate"].tolist(), [False, True, False, False, False, False])
        self.assertEqual(cat["translit_phonetic_match"].tolist(), [False, True, False, False, False, False])
        self.assertEqual(cat["name_high_addr_weak"].tolist(), [False, False, True, False, False, False])
        self.assertEqual(cat["S3"].tolist(), [False, False, True, False, True, True])

    def test_breakdown_and_patterns(self):
        records, pairs, X, truth, claimed = synthetic()
        cat = pair_categories(pairs, X, records, truth, claimed)
        mask = np.array([False, True, False, True, True, True])
        prob = np.array([0.9, 0.7, 0.9, 0.8, 0.6, 0.66])
        b = breakdown(mask, cat, prob, pairs["rank"].to_numpy(), X)
        self.assertEqual(b["all"]["count"], 4)
        self.assertAlmostEqual(b["all"]["mean_prob"], np.mean([0.7, 0.8, 0.6, 0.66]), places=4)
        self.assertEqual(b["singleton_s1"]["count"], 2)
        self.assertEqual(b["singleton_s1"]["share"], 0.5)
        self.assertNotIn("rank_2", {k for k, v in b.items() if v["count"] == 0})
        p = patterns(mask, cat)
        self.assertEqual(sum(x["count"] for x in p), 4)
        self.assertTrue(all(" | " in x["pattern"] for x in p))


if __name__ == "__main__":
    unittest.main()
