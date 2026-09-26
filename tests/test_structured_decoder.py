"""Vectorised Hybrid50 decision engine of src/evaluation/structured_decoder.py vs the offline decision functions.
Synthetic data only. Run from the repository root:  python -m unittest discover -s tests -t .
"""

import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.evaluation.structured_decoder import Table, count_decode, coordinate_descent
from src.matching.hard_negatives import entity_f05
from src.pipeline.deep_recovery import base_predictions, merge_deep


def synthetic_ctx(seed=0, n_s1=120, n_c=160):
    rng = np.random.default_rng(seed)
    s1_ids = [f"S1-{i}" for i in rng.permutation(n_s1) * 7 + 5]
    cand = np.array([f"S2-{i}" for i in range(n_c)], dtype=object)
    k20, k50 = [], []
    for s in s1_ids:
        c = rng.choice(n_c, 7, replace=False)
        k20 += [(s, cand[x], r + 1) for r, x in enumerate(c[:4])]
        k50 += [(s, cand[x], r + 1) for r, x in enumerate(c)]  # K=50 contains the K=20 list plus 3 deep-only
    truth = {s: set() for s in s1_ids}
    for s, c, _ in k50:
        if rng.random() < 0.25:
            truth[s].add(c)
    lab = lambda rows: [int(c in truth[s]) for s, c, _ in rows]
    k20 = pd.DataFrame(k20, columns=["s1_id", "cand_id", "rank"]).assign(label=lambda d: lab(d.to_numpy()))
    k50 = pd.DataFrame(k50, columns=["s1_id", "cand_id", "rank"]).assign(label=lambda d: lab(d.to_numpy()))
    key = set(zip(k20.s1_id, k20.cand_id))
    deep_rows = [i for i, (s, c) in enumerate(zip(k50.s1_id, k50.cand_id)) if (s, c) not in key]
    deep = k50.iloc[deep_rows][["s1_id", "cand_id", "rank"]].rename(columns={"rank": "deep_rank"})
    deep = deep.reset_index(drop=True).assign(label=k50["label"].to_numpy()[deep_rows])
    s1 = pd.DataFrame({"entity_id": s1_ids, "city": ["" if i % 4 else "X" for i in range(n_s1)],
                       "country": ["India" if i % 2 else "US" for i in range(n_s1)]})
    levels = np.array([0.2, 0.6, 0.7, 0.85, 0.9])
    return SimpleNamespace(k20=k20, k50=k50, deep=deep, deep_idx=np.array(deep_rows), s1=s1, truth=truth), \
        rng.choice(levels, len(k20)), rng.choice(levels, len(k50))


class TableTests(unittest.TestCase):
    def test_decisions_and_f05_equal_offline_functions(self):
        for seed in range(3):
            ctx, p20, p50 = synthetic_ctx(seed)
            T = Table(ctx, p20, p50)
            got = T.f05(T.decide(np.where(T.origin == 0, 0.7, 0.85)))
            base, owner = base_predictions(ctx.k20.assign(prob=p20), 0.7, ctx.truth)
            deep = ctx.deep.assign(prob=p50[ctx.deep_idx])
            pred, _ = merge_deep(base, owner, deep, deep["prob"].to_numpy() >= 0.85)
            ref = np.array([entity_f05(pred[s], ctx.truth[s]) for s in ctx.s1.entity_id])
            np.testing.assert_allclose(got, ref)
            self.assertGreater(len(np.unique(ref)), 2)

    def test_count_decoders_keep_exclusivity_and_zero_empties(self):
        ctx, p20, p50 = synthetic_ctx(1)
        T = Table(ctx, p20, p50)
        ref = T.decide(np.where(T.origin == 0, 0.7, 0.85))
        n = np.random.default_rng(2).integers(0, 5, T.n_s1)
        for mode in ("top_n", "cap", "cap_fill"):
            acc = count_decode(T, n, mode, ref)
            self.assertEqual(len(np.unique(T.cand[acc])), acc.sum(), mode)
            per = np.bincount(T.s1[acc], minlength=T.n_s1)
            self.assertTrue((per[n == 0] == 0).all(), mode)
            self.assertTrue((per[n < 4] <= n[n < 4]).all(), mode)

    def test_coordinate_descent_finds_grid_optimum(self):
        p, best = coordinate_descent(lambda q: -(q[0] - 0.42) ** 2 - (q[1] - 0.77) ** 2, [0.5, 0.5])
        self.assertEqual(p, [0.42, 0.77])


if __name__ == "__main__":
    unittest.main()
