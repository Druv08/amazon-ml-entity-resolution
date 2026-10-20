"""Production Hybrid50 inference (src/pipeline/predict_hybrid.py). Synthetic data only.

Run from the repository root:  python -m unittest discover -s tests -t .
"""

import os
import shutil
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.matching.matcher import build_features, fit_tfidf, from_store
from src.matching.stream import Assignments, Winners, s1_ranks
from src.pipeline.deep_recovery import base_predictions, merge_deep
from src.pipeline.predict_hybrid import (DeepRows, hybrid_assignments, iter_hybrid_lists, pair_runs, predict_hybrid,
                                         shard_hybrid)
from tests.test_inference import QuantizedModel, competing_split


def winners(shards, s1_ids, n_keys):
    """Winners over shards of (s1_id, candidate key, prob), rows numbered in shard order."""
    rank, pos = s1_ranks(s1_ids), {s: i for i, s in enumerate(s1_ids)}
    w, row = Winners(n_keys), 0
    for sh in shards:
        w.update([k for _, k, _ in sh], np.array([p for _, _, p in sh], dtype=np.float32),
                 [rank[pos[s]] for s, _, _ in sh], np.arange(row, row + len(sh)))
        row += len(sh)
    return w


def decide(base_shards, deep_shards, s1_ids, n_keys=4, t_base=0.7, t_deep=0.85):
    """{s1: sorted candidate keys} of hybrid_assignments(); '+' marks deep matches."""
    keys, owner, is_deep = hybrid_assignments(winners(base_shards, s1_ids, n_keys), t_base,
                                              winners(deep_shards, s1_ids, n_keys), t_deep)
    rank = s1_ranks(s1_ids)
    out = {s: [] for s in s1_ids}
    for k, o, d in zip(keys, owner, is_deep):
        out[s1_ids[int(np.flatnonzero(rank == o)[0])]].append(f"{k}{'+' if d else ''}")
    return {s: sorted(v) for s, v in out.items()}


class DecisionTests(unittest.TestCase):
    s1 = ["S1-a", "S1-b", "S1-c"]

    def test_base_assignment_cannot_be_stolen(self):
        out = decide([[("S1-a", 0, 0.75)]], [[("S1-b", 0, 0.99)]], self.s1)
        self.assertEqual(out, {"S1-a": ["0"], "S1-b": [], "S1-c": []})

    def test_unaccepted_base_candidate_is_free_for_deep(self):
        out = decide([[("S1-a", 0, 0.69)]], [[("S1-b", 0, 0.90)]], self.s1)
        self.assertEqual(out, {"S1-a": [], "S1-b": ["0+"], "S1-c": []})

    def test_deep_threshold_is_inclusive_in_float32(self):
        below = float(np.nextafter(np.float32(0.85), np.float32(0)))
        out = decide([], [[("S1-a", 1, 0.85), ("S1-b", 2, below)]], self.s1)
        self.assertEqual(out, {"S1-a": ["1+"], "S1-b": [], "S1-c": []})

    def test_deep_exclusivity_highest_probability_across_shards(self):
        out = decide([], [[("S1-a", 1, 0.90)], [("S1-c", 1, 0.95)], [("S1-b", 1, 0.93)]], self.s1)
        self.assertEqual(out, {"S1-a": [], "S1-b": [], "S1-c": ["1+"]})

    def test_deep_tie_across_shards_goes_to_smallest_s1_in_any_order(self):
        for shards in ([[("S1-c", 1, 0.9)], [("S1-b", 1, 0.9)]], [[("S1-b", 1, 0.9)], [("S1-c", 1, 0.9)]]):
            self.assertEqual(decide([], shards, self.s1), {"S1-a": [], "S1-b": ["1+"], "S1-c": []})

    def test_base_tie_across_shards_and_deep_blocked_by_it(self):
        out = decide([[("S1-c", 0, 0.8)], [("S1-b", 0, 0.8)]], [[("S1-a", 0, 1.0)]], self.s1)
        self.assertEqual(out, {"S1-a": [], "S1-b": ["0"], "S1-c": []})

    def test_equals_offline_merge_deep_on_random_sharded_pairs(self):
        rng = np.random.default_rng(7)
        n_s1, n_c = 250, 400  # ~2.5 base and ~2.5 deep claimants per candidate: thresholds and ties both matter
        s1_ids = [f"S1-{i}" for i in rng.permutation(n_s1) * 11 + 3]
        cand_ids = np.array([f"S2-{i}" for i in range(n_c)], dtype=object)
        base, deep = [], []
        for s in s1_ids:
            c = rng.choice(n_c, 8, replace=False)  # 4 base + 4 deep-only candidates per S1
            base += [(s, int(k)) for k in c[:4]]
            deep += [(s, int(k)) for k in c[4:]]
        levels = np.array([0.3, 0.6, 0.7, 0.85, 0.9, 0.95], dtype=np.float32)  # heavy ties at both thresholds
        pb, pd_ = rng.choice(levels, len(base)), rng.choice(levels, len(deep))
        bf = pd.DataFrame({"s1_id": [s for s, _ in base], "cand_id": cand_ids[[k for _, k in base]], "prob": pb})
        df = pd.DataFrame({"s1_id": [s for s, _ in deep], "cand_id": cand_ids[[k for _, k in deep]], "prob": pd_})
        keys = {s: set() for s in s1_ids}
        ref_pred, owner = base_predictions(bf, 0.7, keys)
        ref, _ = merge_deep(ref_pred, owner, df, df["prob"].to_numpy() >= 0.85)

        def shards(pairs, prob):
            rows = [(s, k, p) for (s, k), p in zip(pairs, prob)]
            return [rows[a:a + 37] for a in range(0, len(rows), 37)]

        k, o, _ = hybrid_assignments(winners(shards(base, pb), s1_ids, n_c), 0.7,
                                     winners(shards(deep, pd_), s1_ids, n_c), 0.85)
        acc = Assignments(o, cand_ids[k])
        got = {s: set(acc.of(r)) for s, r in zip(s1_ids, s1_ranks(s1_ids))}
        self.assertEqual(got, ref)
        self.assertGreater(sum(map(len, ref.values())), sum(map(len, ref_pred.values())))  # deep really added some


def shard(rows):
    """Shard frame from (s1_row, rank, candidate id) rows."""
    return pd.DataFrame(rows, columns=["s1_row", "rank", "candidate_entity_id"])


class ShardHybridTests(unittest.TestCase):
    def test_k20_only_kept_and_k50_only_added_in_rank_order_up_to_cap(self):
        base = shard([(0, 1, "S2-a"), (0, 2, "S3-k20only"), (1, 1, "S2-z")])
        deep = shard([(0, 1, "S2-d1"), (0, 2, "S2-a"), (0, 3, "S2-d2"), (0, 4, "S2-d3"), (1, 1, "S2-z"),
                      (1, 2, "S2-y")])
        h = shard_hybrid(base, deep, cap=4)
        lists = h.groupby("s1_row")["candidate_entity_id"].agg(list).to_dict()
        self.assertEqual(lists, {0: ["S2-a", "S3-k20only", "S2-d1", "S2-d2"], 1: ["S2-z", "S2-y"]})
        self.assertEqual(h.loc[h.origin == "deep", "candidate_entity_id"].tolist(), ["S2-d1", "S2-d2", "S2-y"])

    def test_misaligned_shards_rejected(self):
        with self.assertRaises(ValueError):
            shard_hybrid(shard([(0, 1, "S2-a")]), shard([(1, 1, "S2-a")]), cap=5)

    def test_deep_rows_marks_only_the_hybrid_deep_rows(self):
        with tempfile.TemporaryDirectory() as d:
            bp = os.path.join(d, "b.parquet")
            shard([(0, 1, "S2-a"), (1, 1, "S2-z")]).to_parquet(bp)
            deep = shard([(0, 1, "S2-a"), (0, 2, "S2-b"), (0, 3, "S2-c"), (1, 1, "S2-y"), (1, 2, "S2-z")])
            mask = DeepRows({"deep.parquet": bp}, cap=2)("deep.parquet", deep)
        self.assertEqual(list(mask), [False, True, False, True, False])  # S2-c is beyond the cap

    def test_iter_hybrid_lists_merges_countries_in_file_order(self):
        with tempfile.TemporaryDirectory() as d:
            pairing = {}
            for country, rows in (("India", [0, 3, 4]), ("US", [1, 2])):  # S1 row 5 has no candidate
                for part, rr in enumerate((rows[:1], rows[1:])):
                    paths = []
                    for run in ("base", "deep"):
                        p = os.path.join(d, run, country, f"{part:05d}.parquet")
                        os.makedirs(os.path.dirname(p), exist_ok=True)
                        extra = [(r, 2, f"S3-{r}") for r in rr] if run == "deep" else []
                        shard(sorted([(r, 1, f"S2-{r}") for r in rr] + extra)).to_parquet(p)
                        paths.append(p)
                    pairing[paths[1]] = paths[0]
            got = list(iter_hybrid_lists(pairing, cap=5, n_s1=6))
        self.assertEqual(got, [(r, [f"S2-{r}", f"S3-{r}"]) for r in range(5)] + [(5, [])])

    def test_pair_runs_refuses_different_plans(self):
        def run(limit=100, size=1000, names=("US/00000.parquet",)):
            d = os.path.join("x", str(limit))
            return SimpleNamespace(meta={"split": "test", "encoding": "e", "s1_limit": limit,
                                         "config": {"shard_size": size}},
                                   run_dir=d, shard_paths=[os.path.join(d, n) for n in names])
        self.assertEqual(len(pair_runs(run(), run())), 1)
        for other in (run(limit=50), run(size=500), run(names=("US/00000.parquet", "US/00001.parquet"))):
            with self.assertRaises(ValueError):
                pair_runs(run(), other)


def read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def rows(path):
    lines = read(path).splitlines()
    return lines[0], [(ln.split("\t")[0], ln.split("\t")[1].split(",") if ln.split("\t")[1] else []) for ln in lines[1:]]


class EndToEndTests(unittest.TestCase):
    """Two real P2 runs (K=3 base, K=6 deep, cap 5) over a split whose S1 compete across shards."""

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
        cls.base_model = {"model": QuantizedModel(), "threshold": 0.6, "features": feats, "tfidf": tf, "top_k": 3}
        cls.deep_model = dict(cls.base_model, threshold=0.5, top_k=6)
        cls.cap, cls.t_deep = 5, 0.8
        cls.s1_order = cls.stores[3].source_tables()[1]["entity_id"].tolist()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def produce(self, name, workers=1):
        m, c = os.path.join(self.tmp, f"{name}_m.tsv"), os.path.join(self.tmp, f"{name}_c.tsv")
        r = predict_hybrid(self.stores[3], self.stores[6], self.base_model, self.deep_model, m, c, cap=self.cap,
                           deep_threshold=self.t_deep, workers=workers, log=lambda *a: None)
        return m, c, r

    def test_equals_offline_hybrid_reference(self):
        from src.pipeline.hybrid_equivalence import offline_reference, score_in_memory

        m, c, r = self.produce("prod")
        rm, rc = os.path.join(self.tmp, "ref_m.tsv"), os.path.join(self.tmp, "ref_c.tsv")
        ref = offline_reference(score_in_memory(self.stores[3], self.base_model),
                                score_in_memory(self.stores[6], self.deep_model), self.base_model["threshold"],
                                self.s1_order, rm, rc, cap=self.cap, deep_threshold=self.t_deep)
        self.assertEqual(read(m), read(rm))
        self.assertEqual(read(c), read(rc))
        self.assertEqual((r["base_matches"], r["deep_matches"]), (ref["base_matches"], ref["deep_matches"]))
        self.assertGreater(r["deep_matches"], 0)  # the deep stage is really exercised
        self.assertGreater(r["base_matches"], 0)

    def test_parallel_workers_identical(self):
        m1, c1, r1 = self.produce("w1", workers=1)
        m2, c2, r2 = self.produce("w2", workers=2)
        self.assertEqual((read(m1), read(c1)), (read(m2), read(c2)))
        self.assertEqual({k: v for k, v in r1.items() if k != "seconds"}, {k: v for k, v in r2.items() if k != "seconds"})

    def test_official_files(self):
        m, c, _ = self.produce("files")
        hm, match_rows = rows(m)
        hc, cand_rows = rows(c)
        self.assertEqual((hm, hc), ("source1_entity_id\tmatched_entity_ids", "source1_entity_id\tcandidate_entity_ids"))
        self.assertEqual([s for s, _ in match_rows], self.s1_order)  # every S1, official order, singletons included
        self.assertEqual([s for s, _ in cand_rows], self.s1_order)
        self.assertIn([], [ids for _, ids in match_rows])  # an S1 without a match keeps its (empty) row
        cands = dict(cand_rows)
        runs = {k: pd.concat(self.stores[k].iter_shards()).groupby("s1_entity_id")["candidate_entity_id"].agg(list)
                for k in (3, 6)}
        matched = []
        for s, ids in match_rows:
            self.assertTrue(set(ids) <= set(cands[s]))  # every prediction is among the S1's candidates
            matched += ids
        self.assertEqual(len(matched), len(set(matched)))  # global exclusivity
        for s, lst in cands.items():
            self.assertEqual(len(lst), len(set(lst)))  # no duplicate candidates
            base, deep = runs[3].get(s, []), runs[6].get(s, [])
            self.assertEqual(lst[:len(base)], base)  # every exact base candidate, in base order, first
            extra = [x for x in deep if x not in base][:self.cap - len(base)]
            self.assertEqual(lst[len(base):], extra)  # then deep-only candidates in deep rank order, up to the cap
        self.assertTrue(any(len(v) == self.cap for v in cands.values()))


if __name__ == "__main__":
    unittest.main()
