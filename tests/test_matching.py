"""Tests for the P3 matcher's decision rules and I/O (synthetic data only).

Run from the repository root:  python -m unittest discover -s tests -t .
"""

import os
import shutil
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.matching.matcher import (check_top_k, exclusive, from_store, macro_f05, to_matches,
                                  write_matching_results)


def old_exclusive(pairs, prob):
    """The pre-fix rule, used to check that behaviour without ties is unchanged."""
    best = pd.Series(prob).groupby(pairs["cand_id"].to_numpy()).transform("max").to_numpy()
    return np.where(prob >= best, prob, 0.0)


class MacroF05Tests(unittest.TestCase):
    def test_normal_case(self):
        # precision 2/3, recall 1 -> 1.25 * 2/3 / (0.25 * 2/3 + 1) = 0.714...
        self.assertAlmostEqual(macro_f05({"a": {"x", "y", "z"}}, {"a": {"x", "z"}}), 0.7142857, places=6)

    def test_singleton_correct_empty_prediction(self):
        self.assertEqual(macro_f05({"a": set()}, {"a": set()}), 1.0)
        self.assertEqual(macro_f05({}, {"a": set()}), 1.0)  # a missing prediction counts as empty

    def test_singleton_false_positive(self):
        self.assertEqual(macro_f05({"a": {"x"}}, {"a": set()}), 0.0)

    def test_averaged_over_all_s1(self):
        truth = {"a": {"x"}, "b": set(), "c": {"y", "z"}, "d": {"w"}}
        pred = {"a": {"x"}, "b": set(), "c": {"y"}, "d": {"q"}}
        # a: 1, b: 1, c: P=1 R=0.5 -> 0.8333, d: no true positive -> 0
        self.assertAlmostEqual(macro_f05(pred, truth), (1 + 1 + 1.25 * 0.5 / (0.25 + 0.5) + 0) / 4, places=6)
        self.assertEqual(macro_f05({"zz": {"x"}}, truth), 0.25)  # predictions for unknown S1 are ignored


class ExclusiveTests(unittest.TestCase):
    def test_best_s1_keeps_candidate(self):
        pairs = pd.DataFrame({"s1_id": ["a", "b", "b"], "cand_id": ["x", "x", "y"]})
        self.assertEqual(exclusive(pairs, np.array([0.9, 0.7, 0.8])).tolist(), [0.9, 0.0, 0.8])

    def test_unchanged_without_ties(self):
        rng = np.random.default_rng(0)
        n = 5000
        pairs = pd.DataFrame({"s1_id": [f"S1-{i}" for i in rng.integers(0, 800, n)],
                              "cand_id": [f"S2-{i}" for i in rng.integers(0, 1500, n)]})
        prob = rng.random(n)  # continuous -> no ties
        np.testing.assert_array_equal(exclusive(pairs, prob), old_exclusive(pairs, prob))

    def test_tied_probability_goes_to_exactly_one_s1(self):
        pairs = pd.DataFrame({"s1_id": ["S1-b", "S1-a", "S1-c", "S1-b"],
                              "cand_id": ["S2-x", "S2-x", "S2-x", "S3-y"]})
        prob = np.array([0.8, 0.8, 0.3, 0.6], dtype=np.float32)
        self.assertEqual(int((old_exclusive(pairs, prob)[:3] > 0).sum()), 2)  # the bug: both tied S1 kept it
        ex = exclusive(pairs, prob)
        self.assertEqual(ex.dtype, prob.dtype)
        self.assertEqual(ex.tolist(), [0.0, np.float32(0.8), 0.0, np.float32(0.6)])  # smallest s1_id wins
        m = to_matches(pairs, ex, 0.5, ["S1-a", "S1-b", "S1-c"])
        self.assertEqual(m, {"S1-a": {"S2-x"}, "S1-b": {"S3-y"}, "S1-c": set()})

    def test_tie_breaker_is_independent_of_row_order(self):
        rng = np.random.default_rng(1)
        n = 3000
        pairs = pd.DataFrame({"s1_id": [f"S1-{i:04d}" for i in rng.integers(0, 400, n)],
                              "cand_id": [f"S2-{i}" for i in rng.integers(0, 300, n)]})
        pairs = pairs.drop_duplicates().reset_index(drop=True)
        prob = rng.choice([0.2, 0.5, 0.9], len(pairs)).astype(np.float32)  # many ties
        base = pairs.assign(p=exclusive(pairs, prob))
        winners = base[base.p > 0].set_index("cand_id")["s1_id"]
        self.assertTrue(winners.index.is_unique)  # every candidate kept by at most one S1
        for seed in range(3):
            perm = np.random.default_rng(seed).permutation(len(pairs))
            shuf = pairs.iloc[perm].reset_index(drop=True)
            got = shuf.assign(p=exclusive(shuf, prob[perm]))
            self.assertEqual(got[got.p > 0].set_index("cand_id")["s1_id"].sort_index().to_dict(),
                             winners.sort_index().to_dict())
        top = base.groupby("cand_id")["p"].max()  # the winner always carries the candidate's best prob
        best = pd.Series(prob).groupby(pairs["cand_id"]).max()
        np.testing.assert_allclose(top.sort_index(), best.sort_index())

    def test_non_default_index(self):
        pairs = pd.DataFrame({"s1_id": ["a", "b"], "cand_id": ["x", "x"]}, index=[10, 3])
        self.assertEqual(exclusive(pairs, np.array([0.5, 0.5])).tolist(), [0.5, 0.0])


class FromStoreTests(unittest.TestCase):
    def test_frame_conversion(self):
        fr = pd.DataFrame({"s1_entity_id": ["S1-a", "S1-a", "S1-b"], "candidate_entity_id": ["S2-x", "S3-y", "S2-x"],
                           "score": [2.0, 1.0, 1.5], "rank": [1, 2, 1], "name_score": [1.0, 0.0, 0.5],
                           "s1_name": ["A", "A", "B"], "s1_address": ["NULL", "NULL", "5 Main"],
                           "s1_country": ["US"] * 3, "candidate_name": ["A", "nan", "A"],
                           "candidate_address": ["1 st", "", "1 st"], "candidate_country": ["US"] * 3,
                           "label": [1, 0, 0]})
        pairs, records = from_store(fr)
        self.assertEqual(list(pairs.columns), ["s1_id", "cand_id", "block_score", "rank", "name_score", "label"])
        self.assertEqual(pairs["block_score"].tolist(), [2.0, 1.0, 1.5])
        self.assertEqual(list(records.columns), ["entity_id", "business_name", "business_address", "country"])
        self.assertEqual(sorted(records.entity_id), ["S1-a", "S1-b", "S2-x", "S3-y"])  # one row per entity
        rec = records.set_index("entity_id")
        self.assertTrue(pd.isna(rec.loc["S1-a", "business_address"]))  # "NULL" -> NaN
        self.assertTrue(pd.isna(rec.loc["S3-y", "business_name"]))  # "nan" -> NaN
        self.assertTrue(pd.isna(rec.loc["S3-y", "business_address"]))  # "" -> NaN
        self.assertEqual(rec.loc["S2-x", "business_address"], "1 st")

    def test_from_candidate_store(self):
        from src.blocking.engine import CandidateConfig
        from src.blocking.generate_candidates import run
        from src.blocking.handoff import CandidateStore
        from tests.test_candidates import synthetic_split, write_split

        tmp = tempfile.mkdtemp()
        try:
            data = os.path.join(tmp, "raw")
            truth = write_split(data)
            out = os.path.join(tmp, "output", "candidates")
            run("train", data, os.path.join(tmp, "cache"), out, CandidateConfig(top_k=5, max_df=6, shard_size=2),
                workers=1, log=lambda *a: None)
            store = CandidateStore("train", out_dir=out, data_dir=data)
            frame = pd.concat(store.iter_frames(with_records=True, with_labels=True), ignore_index=True)
            pairs, records = from_store(frame)
            self.assertEqual(len(pairs), len(frame))
            self.assertTrue(set(pairs.s1_id) | set(pairs.cand_id) <= set(records.entity_id))
            self.assertFalse(pairs.duplicated(["s1_id", "cand_id"]).any())
            for s, c, lab in zip(pairs.s1_id, pairs.cand_id, pairs.label):
                self.assertEqual(lab, int(c in truth[s]))
            src = {r[0]: r for r in sum(synthetic_split()[:3], [])}
            for e, name in zip(records.entity_id, records.business_name):
                self.assertEqual(name, src[e][1])
            self.assertLessEqual(pairs.groupby("s1_id").size().max(), store.meta["config"]["top_k"])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class OutputTests(unittest.TestCase):
    def test_to_matches_covers_every_s1(self):
        pairs = pd.DataFrame({"s1_id": ["S1-a", "S1-a", "S1-b"], "cand_id": ["S3-z", "S2-y", "S2-x"]})
        m = to_matches(pairs, np.array([0.9, 0.6, 0.4]), 0.5, ["S1-c", "S1-a", "S1-b"])
        self.assertEqual(list(m), ["S1-c", "S1-a", "S1-b"])
        self.assertEqual(m, {"S1-c": set(), "S1-a": {"S3-z", "S2-y"}, "S1-b": set()})

    def test_write_matching_results_format(self):
        matches = {"S1-c": set(), "S1-a": {"S3-z", "S2-y"}, "S1-b": {"S2-x"}}
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "matching_results.tsv")
            write_matching_results(matches, path)
            with open(path, "rb") as f:
                raw = f.read()
        self.assertNotIn(b"\r", raw)
        self.assertEqual(raw.decode("utf-8").split("\n"), [
            "source1_entity_id\tmatched_entity_ids",
            "S1-c\t",
            "S1-a\tS2-y,S3-z",  # sorted, comma-separated, no spaces or quotes
            "S1-b\tS2-x",
            "",
        ])


class TrainWeightTests(unittest.TestCase):
    def test_default_weight_is_unchanged_and_weights_are_used(self):
        from src.matching.matcher import train
        rng = np.random.default_rng(0)
        X = pd.DataFrame({"a": rng.random(400), "b": rng.random(400)})
        y = (X["a"] + 0.3 * rng.random(400) > 0.6).astype(int).to_numpy()
        base = train(X, y).predict_proba(X)[:, 1]
        np.testing.assert_array_equal(base, train(X, y, sample_weight=None).predict_proba(X)[:, 1])
        np.testing.assert_array_equal(base, train(X, y, sample_weight=np.ones(400)).predict_proba(X)[:, 1])
        heavy = np.where(y == 0, 5.0, 1.0)
        self.assertLess(train(X, y, sample_weight=heavy).predict_proba(X)[:, 1].mean(), base.mean())


class EntityF05Tests(unittest.TestCase):
    def test_matches_macro_f05(self):
        from src.matching.hard_negatives import entity_f05
        truth = {"a": {"x"}, "b": set(), "c": {"y", "z"}, "d": {"w"}, "e": set()}
        pred = {"a": {"x"}, "b": set(), "c": {"y"}, "d": {"q"}, "e": {"v"}}
        self.assertAlmostEqual(np.mean([entity_f05(pred[s], truth[s]) for s in truth]), macro_f05(pred, truth))


class S1GateTests(unittest.TestCase):
    def frames(self):
        pairs = pd.DataFrame({"s1_id": ["S1-a", "S1-a", "S1-b", "S1-c"], "cand_id": ["S2-x", "S3-y", "S2-z", "S3-w"],
                              "rank": [1, 2, 1, 1], "block_score": [9.0, 3.0, 5.0, 1.0],
                              "name_score": [4.0, 1.0, 2.0, 0.5]})
        X = pd.DataFrame({"name_tsort": [0.9, 0.2, 0.8, 0.1], "addr_tsort": [0.8, 0.9, 0.3, 0.2],
                          "core_tsort": [0.9, 0.2, 0.8, 0.1], "num_conflict": [0, 1, 0, 0],
                          "first_num_eq": [1, 0, -1, -1], "addr_missing": [0, 0, 1, 1]})
        return pairs, X

    def test_s1_features(self):
        from src.matching.s1_gate import s1_features
        pairs, X = self.frames()
        F = s1_features(pairs, np.array([0.95, 0.4, 0.7, 0.1]), X)
        self.assertEqual(sorted(F.index), ["S1-a", "S1-b", "S1-c"])
        a = F.loc["S1-a"]
        self.assertAlmostEqual(float(a["p1"]), 0.95, places=6)
        self.assertAlmostEqual(float(a["p2"]), 0.4, places=6)
        self.assertEqual((a["n_above_0.3"], a["n_above_0.9"], a["n_cands"]), (2, 1, 2))
        self.assertEqual(float(a["top_block_score"]), 9.0)  # evidence of the most probable candidate
        self.assertEqual(float(F.loc["S1-c", "p2"]), 0.0)  # single candidate

    def test_apply_gate_and_metrics(self):
        from src.matching.s1_gate import apply_gate, gate_metrics
        truth = {"S1-a": {"S2-x"}, "S1-b": set(), "S1-c": set()}
        pred = {"S1-a": {"S2-x"}, "S1-b": {"S2-z"}, "S1-c": set()}
        s1 = pd.DataFrame({"entity_id": ["S1-a", "S1-b", "S1-c"], "country": ["US", "India", "US"], "city": ""})
        gated = apply_gate(pred, {"S1-b"})
        self.assertEqual(gated["S1-b"], set())
        m, per = gate_metrics(gated, truth, s1, {"S1-b"})
        self.assertEqual(m["f05"], 1.0)
        self.assertEqual((m["suppressed"], m["suppressed_true_singletons"], m["suppressed_matched"]), (1, 1, 0))
        m0, _ = gate_metrics(pred, truth, s1, set())
        self.assertAlmostEqual(m0["f05"], 2 / 3)
        self.assertEqual(m0["singleton_empty"], 0.5)


class TopKGuardTests(unittest.TestCase):
    def test_k_mismatch_is_refused(self):
        check_top_k(20, 20)
        with self.assertRaises(SystemExit) as ctx:
            check_top_k(20, 100, "test")
        self.assertIn("K=20", str(ctx.exception))
        self.assertIn("K=100", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
