"""Tests for the untouched final holdout split (src/evaluation/holdout.py).

These tests only check WHICH S1 are selected. Nothing here reads ground truth or scores a model.
"""

import os
import unittest

import numpy as np
import pandas as pd

from src.blocking.sampling import hash_fraction, in_sample
from src.evaluation.holdout import (HOLDOUT_RATE, HOLDOUT_SALT, P2_VALIDATION_RATE, P3_DEV_RATE, P3_DEV_SALT,
                                    development_masks, overlap_report, select_holdout)

RAW_S1 = "data/raw/train/train_source1.tsv"


def synthetic_s1(n=120_000):
    """Synthetic S1 frame: ids, addresses (some in the P3 dense-sample cities), US/India countries."""
    rng = np.random.default_rng(0)
    ids = [f"S1-{i}" for i in rng.choice(10**9, n, replace=False)]
    country = np.where(rng.random(n) < 0.6, "US", "India")
    city = rng.choice(["Dayton, OH", "Pune, Maharashtra", "Tucson, AZ", "Bhopal, Madhya Pradesh"], n,
                      p=[0.45, 0.45, 0.05, 0.05])
    return pd.DataFrame({"entity_id": ids, "business_address": [f"{i} Main Road, {c}" for i, c in enumerate(city)],
                         "country": country})


class HoldoutSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.s1 = synthetic_s1()
        cls.h = select_holdout(cls.s1)

    def test_new_salt(self):
        self.assertNotIn(HOLDOUT_SALT, ("blocking-validation-v1", "p3-matcher-v1"))

    def test_deterministic_and_order_independent(self):
        self.assertTrue(self.h.equals(select_holdout(self.s1)))
        shuffled = self.s1.sample(frac=1, random_state=1).reset_index(drop=True)
        self.assertEqual(sorted(select_holdout(shuffled)["entity_id"]), sorted(self.h["entity_id"]))

    def test_approximate_size(self):
        expected = len(self.s1) * HOLDOUT_RATE * (1 - P2_VALIDATION_RATE) * (1 - P3_DEV_RATE)
        city_share = self.s1["business_address"].str.contains(r"Tucson|Bhopal").mean()
        expected *= 1 - city_share * 0.5  # only the city with the matching country is excluded
        self.assertLess(abs(len(self.h) - expected), 5 * np.sqrt(expected))

    def test_unique_ids_and_both_countries(self):
        self.assertTrue(self.h["entity_id"].is_unique)
        counts = self.h["country"].value_counts()
        self.assertGreater(counts.get("US", 0), 100)
        self.assertGreater(counts.get("India", 0), 100)

    def test_zero_overlap_with_development_samples(self):
        self.assertEqual(overlap_report(self.s1, self.h),
                         {"p2_blocking_validation": 0, "p3_random_dev": 0, "p3_city_dev": 0})
        ids = set(self.h["entity_id"])
        for i in ids:
            self.assertFalse(in_sample(i, P2_VALIDATION_RATE))  # P2's own sampling function
            self.assertFalse(hash_fraction(i, P3_DEV_SALT) < P3_DEV_RATE)
        city_ids = set(self.s1.loc[development_masks(self.s1)["p3_city_dev"], "entity_id"])
        self.assertTrue(city_ids)
        self.assertFalse(ids & city_ids)

    def test_s1_row_points_to_file_row(self):
        self.assertTrue((self.s1["entity_id"].to_numpy()[self.h["s1_row"]] == self.h["entity_id"]).all())


@unittest.skipUnless(os.path.exists(RAW_S1), "raw training data not available")
class RealDataHoldoutTests(unittest.TestCase):
    def test_real_split_is_disjoint_from_p3_training_sample(self):
        from src.blocking.handoff import load_source_table
        from src.matching.sample_candidates import sample_s1

        s1 = load_source_table(RAW_S1)
        h = select_holdout(s1)
        self.assertEqual(overlap_report(s1, h), {"p2_blocking_validation": 0, "p3_random_dev": 0, "p3_city_dev": 0})
        self.assertFalse(set(h["entity_id"]) & set(sample_s1()["entity_id"]))  # the exact P3 training sample
        self.assertTrue(20_000 <= len(h) <= 40_000)
        self.assertTrue(h["entity_id"].is_unique)


if __name__ == "__main__":
    unittest.main()
