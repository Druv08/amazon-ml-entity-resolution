"""Tests for the hybrid K20 + deep candidate set (synthetic data only)."""

import unittest

import numpy as np
import pandas as pd

from src.pipeline.hybrid import blocking_metrics, hybrid_candidates, label_pairs, overlap_stats


def runs():
    # S1-a: base has 3 candidates, deep has 5 (one base candidate missing from deep, different order)
    base = pd.DataFrame({"s1_id": ["S1-a", "S1-a", "S1-a", "S1-b"], "cand_id": ["S2-x", "S2-q", "S3-y", "S2-z"],
                         "rank": [1, 2, 3, 1]})
    deep = pd.DataFrame({"s1_id": ["S1-a"] * 5 + ["S1-b"] * 3,
                         "cand_id": ["S3-y", "S2-x", "S2-d1", "S3-d2", "S2-d3", "S2-z", "S2-e1", "S2-e2"],
                         "rank": [1, 2, 3, 4, 5, 1, 2, 3]})
    return base, deep


class HybridTests(unittest.TestCase):
    def test_all_base_candidates_survive_in_base_order(self):
        base, deep = runs()
        h = hybrid_candidates(base, deep, cap=4)
        a = h[h.s1_id == "S1-a"]
        self.assertEqual(a["cand_id"].tolist()[:3], ["S2-x", "S2-q", "S3-y"])  # exact base list, base order
        self.assertEqual(a["origin"].tolist(), ["base", "base", "base", "deep"])
        self.assertTrue(set(zip(base.s1_id, base.cand_id)) <= set(zip(h.s1_id, h.cand_id)))

    def test_deep_only_fill_by_deep_rank_up_to_cap(self):
        base, deep = runs()
        h = hybrid_candidates(base, deep, cap=4)
        self.assertEqual(h[h.s1_id == "S1-a"]["cand_id"].tolist()[3], "S2-d1")  # best deep-only candidate
        self.assertEqual(h[h.s1_id == "S1-b"]["cand_id"].tolist(), ["S2-z", "S2-e1", "S2-e2"])
        self.assertLessEqual(h.groupby("s1_id").size().max(), 4)
        big = hybrid_candidates(base, deep, cap=50)
        self.assertEqual(len(big[big.s1_id == "S1-a"]), 6)  # 3 base + all 3 deep-only

    def test_no_duplicates_and_ranks(self):
        base, deep = runs()
        h = hybrid_candidates(base, deep, cap=50)
        self.assertFalse(h.duplicated(["s1_id", "cand_id"]).any())
        for _, g in h.groupby("s1_id"):
            self.assertEqual(g["hybrid_rank"].tolist(), list(range(1, len(g) + 1)))
        self.assertTrue(np.isnan(h.loc[h.cand_id == "S2-q", "deep_rank"]).all())  # base-only candidate
        self.assertEqual(float(h.loc[(h.s1_id == "S1-a") & (h.cand_id == "S3-y"), "deep_rank"].iloc[0]), 1.0)

    def test_deterministic_and_input_order_independent(self):
        base, deep = runs()
        a = hybrid_candidates(base, deep, cap=4)
        b = hybrid_candidates(base.sample(frac=1, random_state=1), deep.sample(frac=1, random_state=2), cap=4)
        pd.testing.assert_frame_equal(a, b)

    def test_cap_smaller_than_base_keeps_base(self):
        base, deep = runs()
        h = hybrid_candidates(base, deep, cap=2)
        self.assertEqual(h[h.s1_id == "S1-a"]["cand_id"].tolist(), ["S2-x", "S2-q", "S3-y"])  # base never dropped

    def test_duplicate_input_rejected(self):
        base, deep = runs()
        with self.assertRaises(ValueError):
            hybrid_candidates(pd.concat([base, base.iloc[:1]]), deep)

    def test_labels_and_metrics(self):
        base, deep = runs()
        truth = {"S1-a": {"S2-q", "S3-d2"}, "S1-b": {"S2-e2", "S3-never"}}
        h = hybrid_candidates(base, deep, cap=50)
        self.assertEqual(label_pairs(h, truth).sum(), 3)  # S2-q (base only), S3-d2, S2-e2 (deep)
        country = {"S1-a": "India", "S1-b": "US"}
        mb = blocking_metrics(base, truth, country, ["S1-a", "S1-b"])
        mh = blocking_metrics(h, truth, country, ["S1-a", "S1-b"])
        self.assertEqual((mb["true_pairs_found"], mh["true_pairs_found"]), (1, 3))
        self.assertGreaterEqual(mh["ceiling"], mb["ceiling"])
        self.assertEqual(mh["all_true_retained"], 0.5)  # S1-b misses S3-never
        st = overlap_stats(base, deep, truth, country)["overall"]
        self.assertEqual((st["base_only_candidates"], st["base_only_true"]), (1, 1))
        self.assertEqual((st["deep_only_candidates"], st["deep_only_true"]), (5, 2))


if __name__ == "__main__":
    unittest.main()
