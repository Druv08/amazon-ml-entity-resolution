"""Token-alignment ("decoy") features (src/matching/decoy_features.py) and feature group T. Invented names only.

Run from the repository root:  python -m unittest discover -s tests -t .
"""

import unittest

import numpy as np
import pandas as pd

from src.matching.decoy_features import FEATURES, decoy_features, pair_features


def feats(s1_name, cand_name, s1_addr="", cand_addr=""):
    return dict(zip(FEATURES, pair_features(s1_name, cand_name, s1_addr, cand_addr)))


class AlignmentTests(unittest.TestCase):
    def test_interior_transposition_is_a_typo(self):
        f = feats("Brenwick Partners", "Bernwick Partners")
        self.assertEqual((f["dt_typo"], f["dt_morph"], f["dt_frac_explained"]), (1.0, 0.0, 1.0))

    def test_ocr_confusions(self):
        f = feats("Harlow Foods Gallery", "Har1ow Foods Ga11ery")
        self.assertEqual(f["dt_ocr"], 2 / 3)
        self.assertEqual(f["dt_none"], 0.0)

    def test_end_change_and_suffix_are_morphs(self):
        self.assertEqual(feats("Tavell Bakery", "Tavelli Bakery")["dt_morph"], 0.5)
        self.assertEqual(feats("Quorin Bakery", "Quorinex Bakery")["dt_morph"], 0.5)
        self.assertEqual(feats("Tavella Bakery", "Tavelli Bakery")["dt_morph"], 0.5)  # last-character substitution

    def test_joined_and_domain_forms_match(self):
        for cand in ("#TavellBakery", "tavellbakery.com"):
            f = feats("Tavell Bakery", cand)
            self.assertEqual((f["dt_exact"], f["dt_cand_extra"]), (1.0, 0), cand)

    def test_replaced_token_and_generic_words(self):
        f = feats("Harlow Quorin Foods", "Harlow Brenwick Foods")
        self.assertEqual((f["dt_none"], f["dt_cand_extra"], f["dt_replaced"]), (1 / 3, 1, 1.0))
        g = feats("Tavell Bakery", "Tavell Bakery LLC Services")
        self.assertEqual((g["dt_generic_added"], g["dt_cand_extra"], g["dt_exact"]), (2, 0, 1.0))

    def test_address_markers(self):
        f = feats("Tavell Bakery", "Tavell Bakery", "12 Elm Road", "H.no 12 Elm Road")
        self.assertEqual((f["dt_cand_hno"], f["dt_num_equal"]), (1.0, 1.0))
        g = feats("Tavell Bakery", "Tavell Bakery", "5600 Elm Road", "600 Elm Road")
        self.assertEqual((g["dt_num_extend"], g["dt_num_equal"]), (1.0, 0.0))

    def test_missing_values_are_safe(self):
        f = feats(np.nan, "Tavell Bakery", np.nan, None)
        self.assertEqual(f["dt_s1_distinct"], 0)


class GroupTTests(unittest.TestCase):
    def test_group_t_equals_decoy_features(self):
        from src.matching.matcher import build_features, fit_tfidf

        records = pd.DataFrame({"entity_id": ["S1-a", "S2-x", "S3-y"],
                                "business_name": ["Tavell Bakery", "Tavelli Bakery LLC", "#TavellBakery"],
                                "business_address": ["12 Elm Road", "H.no 12 Elm Road", np.nan],
                                "country": ["US", "US", "US"]})
        pairs = pd.DataFrame({"s1_id": ["S1-a", "S1-a"], "cand_id": ["S2-x", "S3-y"], "block_score": [2.0, 1.0],
                              "rank": [1, 2], "name_score": [1.0, 0.5]})
        X = build_features(pairs, records, fit_tfidf(records), groups=("T",))
        self.assertEqual(list(X.columns[-len(FEATURES):]), FEATURES)
        np.testing.assert_array_equal(X[FEATURES].to_numpy(), decoy_features(pairs, records).to_numpy())


if __name__ == "__main__":
    unittest.main()
