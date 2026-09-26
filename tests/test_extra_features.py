"""Tests for the candidate matcher feature groups (synthetic data only)."""

import unittest

import numpy as np
import pandas as pd

from src.matching.extra_features import (NameStats, address_number_features, blocker_features, extra_features,
                                         name_rarity_features, translit_features)


def frames():
    records = pd.DataFrame({
        "entity_id": ["S1-a", "S1-b", "S2-x", "S2-y", "S3-z"],
        "business_name": ["Pioneer Bakers Pvt Ltd", "Sai Enterprises", "पायोनियर बेकर्स", "Sai Traders", "Sai Enterprises"],
        "business_address": ["1622 Oak Street, Pune", "5 Main Road", "162 OAK ST, PUNE", np.nan, "7 Main Road"],
        "country": ["India"] * 5,
    })
    pairs = pd.DataFrame({"s1_id": ["S1-a", "S1-b", "S1-b"], "cand_id": ["S2-x", "S2-y", "S3-z"],
                          "block_score": [30.0, 12.0, 20.0], "name_score": [10.0, 12.0, 15.0]})
    return pairs, records


class AddressNumberTests(unittest.TestCase):
    def test_typo_prefix_and_missing(self):
        pairs, records = frames()
        f = address_number_features(pairs, records)
        self.assertAlmostEqual(f["num_best_ratio"][0], 0.857, places=3)  # 1622 vs 162: one-digit typo
        self.assertEqual(f["num_prefix"][0], 1)
        self.assertEqual((f["num_shared_n"][0], f["num_only_s1_n"][0], f["num_only_cand_n"][0]), (0, 1, 1))
        self.assertEqual((f["num_best_ratio"][1], f["num_prefix"][1]), (-1.0, -1))  # candidate has no address
        self.assertEqual(f["num_only_s1_n"][2], 1)
        self.assertAlmostEqual(f["addr_word_jacc"][0], 0.5)  # {oak, street, pune} vs {oak, st, pune}: no expansion
        self.assertAlmostEqual(f["addr_word_jacc"][2], 1.0)  # main road == main road


class TranslitTests(unittest.TestCase):
    def test_indic_candidate_matches_english_phonetically(self):
        pairs, records = frames()
        f = translit_features(pairs, records)
        self.assertEqual(f["script_mismatch"].tolist(), [1, 0, 0])
        self.assertGreater(f["ph_jacc"][0], 0.9)  # {pnr, bkrs} on both sides
        self.assertEqual(f["ph_bigram_hit"][0], 1)
        self.assertGreater(f["latin_tsort"][0], 0.3)  # "paayoniyar bekars" vs "pioneer bakers ...": partial only,
        # which is why the phonetic keys exist (raw Devanagari vs Latin token sort would be 0)
        self.assertEqual(f["ph_jacc"][2], 1.0)


class BlockerTests(unittest.TestCase):
    def test_address_part_of_block_score(self):
        pairs, _ = frames()
        self.assertEqual(blocker_features(pairs)["block_addr_score"].tolist(), [20.0, 0.0, 5.0])


class NameRarityTests(unittest.TestCase):
    def test_chain_names_are_frequent_and_generic_words_weigh_little(self):
        pairs, records = frames()
        s1 = pd.DataFrame({"business_name": ["Pioneer Bakers Pvt Ltd", "Sai Enterprises", "Sai Enterprises Ltd",
                                             "Sai Motors"], "country": ["India"] * 4})
        s2 = pd.DataFrame({"business_name": ["पायोनियर बेकर्स", "Sai Traders", "Sai Foods"], "country": ["India"] * 3})
        s3 = pd.DataFrame({"business_name": ["Sai Enterprises"], "country": ["India"]})
        stats = NameStats.from_tables({1: s1, 2: s2, 3: s3})
        f = name_rarity_features(pairs, records, stats)
        self.assertGreater(f["s1_core_freq"][1], f["s1_core_freq"][0])  # "sai enterprises" x2 in S1 vs unique
        self.assertEqual(f["core_idf_jacc"][0], 0.0)  # Devanagari vs Latin words: no shared word
        self.assertGreater(f["core_idf_jacc"][2], f["core_idf_jacc"][1])  # full name match > only "sai" shared
        self.assertLess(stats.idf("India", "sai"), stats.idf("India", "motors"))

    def test_extra_features_dispatch(self):
        pairs, records = frames()
        with self.assertRaises(ValueError):
            extra_features(["E"], pairs, records)
        f = extra_features(["A", "D"], pairs, records)
        self.assertEqual(f.dtypes.unique().tolist(), [np.float32])
        self.assertIn("block_addr_score", f.columns)
        with self.assertRaises(ValueError):
            extra_features(["Z"], pairs, records)


if __name__ == "__main__":
    unittest.main()
