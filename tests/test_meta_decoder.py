"""Production meta decoder (src/pipeline/meta_decoder.py, predict_hybrid.predict_hybrid_meta). Synthetic data only.

Run from the repository root:  python -m unittest discover -s tests -t .
"""

import os
import shutil
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.matching.matcher import build_features, fit_tfidf, from_store
from src.matching.stream import Winners, s1_ranks
from src.pipeline.meta_decoder import CONTEXT, META_VERSION, context_features, decide, meta_scores
from src.pipeline.predict_hybrid import hybrid_assignments, predict_hybrid_meta
from tests.test_inference import QuantizedModel, competing_split


def naive_context(s1, prob, origin):
    """Loop reference of context_features()."""
    out = []
    for i in range(len(prob)):
        m = np.flatnonzero(s1 == s1[i])
        order = m[np.argsort(-prob[m], kind="stable")]
        ps = prob[order]
        top = [ps[j] if j < len(ps) else 0.0 for j in range(3)]
        base, deep = prob[m][origin[m] == 0], prob[m][origin[m] == 1]
        row = top + [float((prob[m] >= t).sum()) for t in (0.3, 0.5, 0.7, 0.9)]
        row += [prob[m].sum(), base.max() if len(base) else 0.0, deep.max() if len(deep) else 0.0, float(len(m)),
                float(np.flatnonzero(order == i)[0]), top[0] - prob[i], top[0] - top[1]]
        out.append(row)
    return np.array(out)


class ContextTests(unittest.TestCase):
    def test_equals_loop_reference_with_ties(self):
        rng = np.random.default_rng(5)
        n = 400
        s1 = rng.integers(0, 60, n) * 3 + 7  # arbitrary, unsorted S1 codes; some S1 with a single row
        prob = rng.choice([0.1, 0.3, 0.5, 0.7, 0.9, 0.95], n)  # ties, including exactly on the levels
        origin = (rng.random(n) < 0.4).astype(int)
        got = context_features(s1, prob, origin)
        self.assertEqual(list(got.columns), CONTEXT)
        np.testing.assert_allclose(got.to_numpy(), naive_context(s1, prob, origin))

    def test_meta_scores_floor_and_row_check(self):
        class Echo:
            def predict_proba(self, M):
                return np.column_stack([1 - M["prob"], M["prob"]])

        prob = np.array([0.5, 0.01, 0.9])
        pair = np.zeros((2, 2), dtype=np.float32)
        s = meta_scores(Echo(), prob, [0, 0, 1], [1, 2, 3], [4, 4, 4], pair, ["rank", "x"], floor=0.02)
        self.assertEqual(s[1], -np.inf)
        np.testing.assert_allclose(s[[0, 2]], [0.5, 0.9], rtol=1e-6)
        with self.assertRaises(ValueError):
            meta_scores(Echo(), prob, [0, 0, 1], [1, 2, 3], [4, 4, 4], np.zeros((3, 2)), ["rank", "x"], floor=0.02)


class DecisionTests(unittest.TestCase):
    def test_vectorised_decide_equals_streaming_winners(self):
        rng = np.random.default_rng(9)
        n_s1, n_c, n = 120, 90, 900
        s1_ids = np.array([f"S1-{i}" for i in rng.permutation(n_s1) * 13 + 2], dtype=object)
        s1 = rng.integers(0, n_s1, n)
        cand = rng.integers(0, n_c, n)
        origin = (rng.random(n) < 0.5).astype(int)
        _, first = np.unique(np.c_[s1, cand], axis=0, return_index=True)  # one row per (S1, candidate)
        keep = np.sort(first)
        s1, cand, origin = s1[keep], cand[keep], origin[keep]
        score = rng.choice([-np.inf, 0.4, 0.6, 0.65, 0.8], len(keep))  # -inf = below the meta floor; ties
        rank = s1_ranks(s1_ids)[s1]
        ref = decide(cand, origin, score, rank, 0.6, 0.65)
        wins = {0: Winners(n_c, np.float64), 1: Winners(n_c, np.float64)}
        for a in range(0, len(keep), 77):  # shards
            sl = slice(a, a + 77)
            ok = score[sl] > -np.inf
            for o in (0, 1):
                m = ok & (origin[sl] == o)
                wins[o].update(cand[sl][m], score[sl][m], rank[sl][m], np.arange(len(keep))[sl][m])
        k, owner, _ = hybrid_assignments(wins[0], 0.6, wins[1], 0.65)
        got = set(zip(k, owner))
        self.assertEqual(got, set(zip(cand[ref], rank[ref])))
        self.assertGreater(len(got), 10)


class StubMeta:
    """Deterministic stand-in for the meta-model: coarse scores that depend on the S1 context and pair features."""

    def predict_proba(self, M):
        p = (0.55 * M["prob"] + 0.25 * (M["prob_pos_in_s1"] == 0) + 0.15 * (M["addr_tfidf"] > 0.5)
             - 0.1 * M["origin"] + 0.05 * (M["gap12"] > 0.2)).to_numpy(dtype=np.float64)
        p = np.round(np.clip(p, 0, 1), 1)
        return np.column_stack([1 - p, p])


def read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


class EndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from src.blocking.engine import CandidateConfig
        from src.blocking.generate_candidates import run
        from src.blocking.handoff import CandidateStore

        cls.tmp = tempfile.mkdtemp()
        data = os.path.join(cls.tmp, "raw")
        competing_split(data)
        cls.stores = {}
        for k in (3, 6):
            out = os.path.join(cls.tmp, f"k{k}")
            run("test", data, os.path.join(cls.tmp, "cache"), out, CandidateConfig(top_k=k, max_df=100, shard_size=4),
                workers=1, log=lambda *a: None)
            cls.stores[k] = CandidateStore("test", out_dir=out, data_dir=data)
        frame = pd.concat(cls.stores[6].iter_frames(with_records=True), ignore_index=True)
        pairs, records = from_store(frame)
        tf = fit_tfidf(records)
        feats = list(build_features(pairs, records, tf).columns)
        base = {"model": QuantizedModel(), "threshold": 0.6, "features": feats, "tfidf": tf, "top_k": 3}
        cls.bundle = {"meta_version": META_VERSION, "base_matcher": base, "deep_matcher": dict(base, top_k=6),
                      "meta_model": StubMeta(), "meta_floor": 0.02, "pair_features": feats,
                      "thresholds": {"base": 0.6, "deep": 0.7}, "hybrid": {"cap": 5}}
        cls.s1_order = cls.stores[3].source_tables()[1]["entity_id"].tolist()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def produce(self, name, workers=1):
        m, c = os.path.join(self.tmp, f"{name}_m.tsv"), os.path.join(self.tmp, f"{name}_c.tsv")
        r = predict_hybrid_meta(self.stores[3], self.stores[6], self.bundle, m, c, workers=workers,
                                log=lambda *a: None)
        return m, c, r

    def test_streaming_equals_offline_reference(self):
        from src.pipeline.hybrid_equivalence import offline_meta_reference

        m, c, r = self.produce("prod")
        rm, rc = os.path.join(self.tmp, "ref_m.tsv"), os.path.join(self.tmp, "ref_c.tsv")
        ref = offline_meta_reference(self.stores[3], self.stores[6], self.bundle, rm, rc)
        self.assertEqual(read(m), read(rm))
        self.assertEqual(read(c), read(rc))
        self.assertEqual((r["base_matches"], r["deep_matches"]), (ref["base_matches"], ref["deep_matches"]))
        self.assertGreater(r["base_matches"], 0)
        self.assertGreater(r["deep_matches"], 0)  # the deep stage is exercised

    def test_parallel_workers_identical(self):
        m1, c1, r1 = self.produce("w1", workers=1)
        m2, c2, r2 = self.produce("w2", workers=2)
        self.assertEqual((read(m1), read(c1)), (read(m2), read(c2)))
        self.assertEqual({k: v for k, v in r1.items() if k != "seconds"}, {k: v for k, v in r2.items() if k != "seconds"})

    def test_official_files_subset_exclusive_ordered(self):
        m, c, _ = self.produce("files")
        rows = lambda p: [(ln.split("\t")[0], ln.split("\t")[1].split(",") if ln.split("\t")[1] else [])
                          for ln in read(p).splitlines()[1:]]
        match_rows, cand_rows = rows(m), dict(rows(c))
        self.assertEqual([s for s, _ in match_rows], self.s1_order)
        seen = []
        for s, ids in match_rows:
            self.assertTrue(set(ids) <= set(cand_rows[s]))
            seen += ids
        self.assertEqual(len(seen), len(set(seen)))

    def test_decoy_features_path_equals_offline_reference(self):
        from src.matching.decoy_features import FEATURES as DECOY
        from src.pipeline.hybrid_equivalence import offline_meta_reference

        class StubDecoyMeta(StubMeta):
            def predict_proba(self, M):
                q = super().predict_proba(M)[:, 1] - 0.2 * M["dt_none"].to_numpy() + 0.1 * M["dt_exact"].to_numpy()
                q = np.round(np.clip(q, 0, 1), 1)
                return np.column_stack([1 - q, q])

        b = dict(self.bundle, meta_extra=("decoy",), meta_model=StubDecoyMeta(),
                 pair_features=list(self.bundle["pair_features"]) + DECOY)
        out = {}
        for w in (1, 2):
            m, c = os.path.join(self.tmp, f"d{w}_m.tsv"), os.path.join(self.tmp, f"d{w}_c.tsv")
            predict_hybrid_meta(self.stores[3], self.stores[6], b, m, c, workers=w, log=lambda *a: None)
            out[w] = (read(m), read(c))
        rm, rc = os.path.join(self.tmp, "dref_m.tsv"), os.path.join(self.tmp, "dref_c.tsv")
        offline_meta_reference(self.stores[3], self.stores[6], b, rm, rc)
        self.assertEqual(out[1], (read(rm), read(rc)))
        self.assertEqual(out[1], out[2])

    def test_bundle_version_checked(self):
        with self.assertRaises(ValueError):
            predict_hybrid_meta(self.stores[3], self.stores[6], dict(self.bundle, meta_version="old"),
                                os.path.join(self.tmp, "x_m.tsv"), os.path.join(self.tmp, "x_c.tsv"),
                                workers=1, log=lambda *a: None)


if __name__ == "__main__":
    unittest.main()
