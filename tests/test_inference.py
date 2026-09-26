"""Streaming matcher inference (src/matching/stream.py) vs the previous in-memory path.

The in-memory reference below is the pre-streaming predict.py logic, kept here only for verification.
Synthetic data only. Run from the repository root:  python -m unittest discover -s tests -t .
"""

import os
import shutil
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.matching.matcher import (build_features, exclusive, fit_tfidf, from_store, to_matches, top2,
                                  write_matching_results, xtop)
from src.matching.extra_features import NameStats
from src.matching.stream import CandidateIndex, TopTwo, Winners, s1_ranks, stream_predict, write_matches


def reference_predict(store, m, out_path, s1_limit=None):
    """The previous predict.py: every scored pair in memory, then exclusive() + to_matches()."""
    groups = tuple(m.get("feature_groups", ()))
    ns = NameStats.from_tables(store.source_tables()) if "E" in groups else None
    tops = [xtop(*from_store(df), m["tfidf"]) for df in store.iter_frames(with_records=True)]
    xt = {}
    for c in tops[0]:
        d = pd.concat(t[c] for t in tops)
        xt[c] = top2(d["s"], d["c"])
    parts = []
    for df in store.iter_frames(with_records=True):
        pairs, records = from_store(df)
        X = build_features(pairs, records, m["tfidf"], xt, groups, ns)[m["features"]]
        parts.append(pairs[["s1_id", "cand_id"]].assign(prob=m["model"].predict_proba(X)[:, 1].astype(np.float32)))
    scored = pd.concat(parts, ignore_index=True)
    s1_ids = store.source_tables()[1]["entity_id"]
    s1_ids = s1_ids.iloc[:s1_limit] if s1_limit else s1_ids
    matches = to_matches(scored, exclusive(scored, scored["prob"].to_numpy()), m["threshold"], s1_ids)
    write_matching_results(matches, out_path)
    return scored


def read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


class QuantizedModel:
    """Deterministic stand-in for the classifier: coarse probabilities (many exact ties) that depend on
    within-S1 and cross-entity features, so a wrong global top-2 or a shard-local exclusivity shows up."""

    def predict_proba(self, X):
        # accepted (threshold 0.6) when the global cross-entity gap is positive, or zero with a strong address
        gap = X["name_tfidf_xgap"]
        p = (0.35 + 0.5 * (gap > 0) + 0.45 * ((gap == 0) & (X["addr_tfidf"] > 0.6)) + 0.1 * X["name_tfidf"]
             - 0.05 * (X["block_score_xgap"] < 0)).to_numpy()
        p = np.round(np.clip(p, 0, 1), 1).astype(np.float64)
        return np.column_stack([1 - p, p])


# ---------------------------------------------------------------- helpers in isolation
class WinnersTests(unittest.TestCase):
    ids = np.array(["S2-x", "S2-y", "S3-z"], dtype=object)  # candidate keys 0, 1, 2

    def run_shards(self, shards, s1_ids, threshold=0.5):
        """shards: lists of (s1_id, cand_key, prob); returns {s1: [cands]} as written to the TSV."""
        rank = s1_ranks(s1_ids)
        pos = {s: i for i, s in enumerate(s1_ids)}
        w, row = Winners(len(self.ids)), 0
        for sh in shards:
            w.update([k for _, k, _ in sh], [p for _, _, p in sh], [rank[pos[s]] for s, _, _ in sh],
                     np.arange(row, row + len(sh)))
            row += len(sh)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.tsv")
            write_matches(path, w, threshold, self.ids, s1_ids, rank)
            lines = read(path).splitlines()
        self.assertEqual(lines[0], "source1_entity_id\tmatched_entity_ids")
        return [(ln.split("\t")[0], ln.split("\t")[1].split(",") if ln.split("\t")[1] else []) for ln in lines[1:]]

    def test_multi_shard_exclusivity(self):
        out = self.run_shards([[("S1-b", 0, 0.91)], [("S1-a", 0, 0.95)]], ["S1-b", "S1-a"])
        self.assertEqual(out, [("S1-b", []), ("S1-a", ["S2-x"])])

    def test_multi_shard_tie_any_order(self):
        for shards in ([[("S1-b", 0, 0.90)], [("S1-a", 0, 0.90)]], [[("S1-a", 0, 0.90)], [("S1-b", 0, 0.90)]]):
            self.assertEqual(dict(self.run_shards(shards, ["S1-b", "S1-a"])), {"S1-a": ["S2-x"], "S1-b": []})

    def test_lexicographic_not_numeric_s1_order(self):
        # "S1-10" < "S1-9" as strings, as in exclusive()'s sort
        out = dict(self.run_shards([[("S1-9", 0, 0.8)], [("S1-10", 0, 0.8)]], ["S1-9", "S1-10"]))
        self.assertEqual(out, {"S1-10": ["S2-x"], "S1-9": []})

    def test_winner_below_threshold_goes_to_nobody(self):
        out = dict(self.run_shards([[("S1-a", 0, 0.40), ("S1-b", 0, 0.30)]], ["S1-a", "S1-b"]))
        self.assertEqual(out, {"S1-a": [], "S1-b": []})

    def test_one_s1_keeps_several_candidates(self):
        out = dict(self.run_shards([[("S1-a", 2, 0.9), ("S1-a", 0, 0.8)], [("S1-a", 1, 0.7), ("S1-b", 1, 0.6)]],
                                   ["S1-a", "S1-b"]))
        self.assertEqual(out, {"S1-a": ["S2-x", "S2-y", "S3-z"], "S1-b": []})

    def test_singletons_and_file_order(self):
        out = self.run_shards([[("S1-c", 1, 0.9)]], ["S1-z", "S1-c", "S1-a"])
        self.assertEqual(out, [("S1-z", []), ("S1-c", ["S2-y"]), ("S1-a", [])])

    def test_threshold_must_be_positive(self):
        with self.assertRaises(ValueError):
            Winners(1).accepted(0.0)


class TopTwoTests(unittest.TestCase):
    def test_matches_global_top2_over_chunks(self):
        rng = np.random.default_rng(0)
        n, n_keys = 4000, 700
        keys = rng.integers(0, n_keys, n)
        scores = np.round(rng.random(n), 2)  # with ties
        cand = np.array([f"S2-{k}" for k in keys], dtype=object)
        t = TopTwo(n_keys)
        for a in range(0, n, 333):  # chunked like shards
            t.update(keys[a:a + 333], scores[a:a + 333])
        ref = top2(scores, cand)
        got = t.frame(keys, cand)
        self.assertEqual(sorted(zip(got["c"], got["s"])), sorted(zip(ref["c"], ref["s"])))

    def test_float32_storage_is_exact_and_dtype_checked(self):
        s = np.array([0.1, 0.7, 0.3], dtype=np.float32)
        t = TopTwo(2, np.float32)
        t.update([0, 0, 1], s)
        ref = top2(s, np.array(["a", "a", "b"], dtype=object))  # float64 of the same float32 values
        got = t.frame([0, 1], np.array(["a", "b"], dtype=object))
        self.assertEqual(sorted(zip(got["c"], got["s"].astype(float))), sorted(zip(ref["c"], ref["s"])))
        with self.assertRaises(TypeError):
            t.update([0], np.array([0.5]))  # float64 into a float32 store would round silently

    def test_candidate_index(self):
        idx = CandidateIndex(10, 5)
        np.testing.assert_array_equal(idx.keys([2, 3, 3], [4, 0, 4]), [4, 10, 14])
        with self.assertRaises(ValueError):
            idx.keys([1], [0])


class WinnersEquivalenceTests(unittest.TestCase):
    def test_same_as_exclusive_on_random_multi_shard_pairs(self):
        rng = np.random.default_rng(3)
        n_s1, n_c = 300, 200
        s1_ids = np.array([f"S1-{i}" for i in rng.permutation(n_s1) * 7 + 1], dtype=object)  # unsorted, varying length
        cand_ids = np.array([f"S2-{i}" for i in range(n_c)], dtype=object)
        rows = [(s, c) for s in range(n_s1) for c in rng.choice(n_c, 8, replace=False)]
        pairs = pd.DataFrame({"s1_id": [s1_ids[s] for s, _ in rows], "cand_id": [cand_ids[c] for _, c in rows]})
        prob = rng.choice([0.3, 0.6, 0.7, 0.9], len(pairs)).astype(np.float32)  # heavy ties
        with tempfile.TemporaryDirectory() as d:
            ref = os.path.join(d, "ref.tsv")
            write_matching_results(to_matches(pairs, exclusive(pairs, prob), 0.65, s1_ids), ref)
            rank, w = s1_ranks(s1_ids), Winners(n_c)
            s1_pos, ck = np.array([s for s, _ in rows]), np.array([c for _, c in rows])
            for a in range(0, len(rows), 97):  # shards
                sl = slice(a, a + 97)
                w.update(ck[sl], prob[sl], rank[s1_pos[sl]], np.arange(len(rows))[sl])
            got = os.path.join(d, "got.tsv")
            write_matches(got, w, 0.65, cand_ids, s1_ids, rank)
            self.assertEqual(read(got), read(ref))


# ---------------------------------------------------------------- end to end on a multi-shard CandidateStore
def competing_split(d):
    """S1 duplicates spread over many shards compete for the same S2/S3 records."""
    base = [("Zephyrine Bakery", "12 Quillon Street"), ("Acme Tools", "5 Harbor Road"),
            ("Maple Trust", "44 Birch Lane"), ("Orchid Dental Studio", "9 Pine Court"),
            ("Harbor Kite Cafe", "71 Elm Avenue")]
    s1, s2, s3 = [], [], []
    # businesses 0-2: distinct duplicate names (non-zero cross-entity gaps, needs the GLOBAL top-2);
    # businesses 3-4: identical duplicates (tied probabilities, needs the global tie-breaker)
    suffix = ["", " LLC", " Bakers", " Group", " Studio", " Partners"]
    for i in range(30):
        name, addr = base[i % 5]
        s1.append((f"S1-{i * 13 + 7}", name + (suffix[i // 5] if i % 5 < 3 else ""), f"{addr}, Dayton, OH", "US"))
    for j, (name, addr) in enumerate(base):
        s2.append((f"S2-{j}", name.upper(), f"{addr}, DAYTON, OH", "US"))
        s2.append((f"S2-{j + 50}", name, "", "US"))
        s3.append((f"S3-{j}", name.lower() + " inc", f"{addr}, Dayton, Ohio", "US"))
    s3 += [(f"S3-f{k}", f"Filler {k} Shop", f"{k} Main Street, Dayton, OH", "US") for k in range(20)]
    header = "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
    os.makedirs(d, exist_ok=True)
    for i, rows in ((1, s1), (2, s2), (3, s3)):
        with open(os.path.join(d, f"test_source{i}.tsv"), "w", encoding="utf-8", newline="\n") as f:
            f.write(header + "".join("\t".join(r) + "\n" for r in rows))


class EndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from src.blocking.engine import CandidateConfig
        from src.blocking.generate_candidates import run
        from src.blocking.handoff import CandidateStore

        cls.tmp = tempfile.mkdtemp()
        data = os.path.join(cls.tmp, "raw")
        competing_split(data)
        out = os.path.join(cls.tmp, "output", "candidates")
        run("test", data, os.path.join(cls.tmp, "cache"), out, CandidateConfig(top_k=6, max_df=100, shard_size=4),
            workers=1, log=lambda *a: None)
        cls.store = CandidateStore("test", out_dir=out, data_dir=data)
        frame = pd.concat(cls.store.iter_frames(with_records=True), ignore_index=True)
        pairs, records = from_store(frame)
        tf = fit_tfidf(records)
        feats = list(build_features(pairs, records, tf).columns)
        cls.model = {"model": QuantizedModel(), "threshold": 0.6, "features": feats, "tfidf": tf, "top_k": 6}
        cls.pairs = pairs

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_setup_has_cross_shard_competition(self):
        self.assertGreater(len(self.store.shard_paths), 3)
        per_cand = self.pairs.groupby("cand_id")["s1_id"].nunique()
        self.assertGreater(per_cand.max(), 4)  # candidates claimed by S1 in several shards

    def test_streaming_equals_in_memory_reference(self):
        ref, got = os.path.join(self.tmp, "ref.tsv"), os.path.join(self.tmp, "got.tsv")
        scored = reference_predict(self.store, self.model, ref)
        self.assertGreater((scored.groupby("cand_id")["prob"].transform("max") == scored["prob"]).sum(),
                           scored["cand_id"].nunique())  # the data really contains tied winners
        r = stream_predict(self.store, self.model, got, log=lambda *a: None)
        self.assertEqual(read(got), read(ref))
        self.assertEqual(r["pairs"], len(scored))
        self.assertEqual(r["s1"], 30)

    def test_streaming_equals_reference_with_feature_groups(self):
        ns = NameStats.from_tables(self.store.source_tables())
        frame = pd.concat(self.store.iter_frames(with_records=True), ignore_index=True)
        pairs, records = from_store(frame)
        feats = list(build_features(pairs, records, self.model["tfidf"], None, ("A", "C", "E"), ns).columns)
        self.assertIn("num_best_ratio", feats)
        self.assertIn("ph_jacc", feats)
        self.assertIn("core_idf_jacc", feats)
        m = dict(self.model, features=feats, feature_groups=("A", "C", "E"))
        ref, got = os.path.join(self.tmp, "ref_g.tsv"), os.path.join(self.tmp, "got_g.tsv")
        reference_predict(self.store, m, ref)
        stream_predict(self.store, m, got, log=lambda *a: None)
        self.assertEqual(read(got), read(ref))

    def test_parallel_workers_give_identical_output(self):
        ns = NameStats.from_tables(self.store.source_tables())
        frame = pd.concat(self.store.iter_frames(with_records=True), ignore_index=True)
        pairs, records = from_store(frame)
        feats = list(build_features(pairs, records, self.model["tfidf"], None, ("A", "C", "E"), ns).columns)
        for m in (self.model, dict(self.model, features=feats, feature_groups=("A", "C", "E"))):
            one, two = os.path.join(self.tmp, "w1.tsv"), os.path.join(self.tmp, "w2.tsv")
            r1 = stream_predict(self.store, m, one, log=lambda *a: None, workers=1)
            r2 = stream_predict(self.store, m, two, log=lambda *a: None, workers=2)
            self.assertEqual(read(one), read(two))
            self.assertEqual(r1, r2)

    def test_output_is_exclusive_subset_in_file_order(self):
        got = os.path.join(self.tmp, "got2.tsv")
        stream_predict(self.store, self.model, got, log=lambda *a: None)
        lines = read(got).splitlines()[1:]
        s1_order = self.store.source_tables()[1]["entity_id"].tolist()
        self.assertEqual([ln.split("\t")[0] for ln in lines], s1_order)
        allowed = self.pairs.groupby("s1_id")["cand_id"].apply(set).to_dict()
        seen = []
        for ln in lines:
            s, ids = ln.split("\t")
            ids = ids.split(",") if ids else []
            self.assertTrue(set(ids) <= allowed.get(s, set()))
            seen += ids
        self.assertEqual(len(seen), len(set(seen)))  # every candidate given to at most one S1
        self.assertGreater(len(seen), 0)


if __name__ == "__main__":
    unittest.main()
