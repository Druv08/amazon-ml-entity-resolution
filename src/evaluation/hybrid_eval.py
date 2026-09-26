"""Development-sample evaluation of Hybrid50 decisions from any K=20 / K=50 pair probabilities (OOF only).

    ctx = HybridDev()                                    # P3 development sample, K=20 + K=50 OOF layouts
    r = ctx.run(p20, p50)                                # K=20 exclusivity + tuned threshold, deep >= 0.85 merge
    print(ctx.metrics(r["pred"], ref=ctx.per_s1(baseline_pred)))

Decision rule = src/pipeline/deep_recovery.py (base priority, deep exclusivity). Metrics on the hash-random S1 only
(the headline set), with paired per-S1 deltas: SE and a percentile bootstrap CI over S1. Never the final holdout.
"""

import numpy as np
import pandas as pd

from src.matching.hard_negatives import entity_f05
from src.matching.matcher import best_threshold, exclusive, to_matches
from src.pipeline.deep_recovery import DEEP_THRESHOLD, merge_deep
from src.pipeline.hybrid import hybrid_candidates

K20, K50 = "output/candidates_p3/k20", "output/candidates_p3/k50"


def nonlatin_ids(ids, data_dir="data/raw/train"):
    """The subset of ``ids`` (S1/S2/S3 of the training split) whose business name contains non-Latin letters."""
    from src.blocking.data_io import source_path
    from src.blocking.handoff import load_source_table
    from src.blocking.normalize import has_nonlatin

    ids, out = set(ids), set()
    for s in (1, 2, 3):
        t = load_source_table(source_path(data_dir, "train", s))[["entity_id", "business_name"]]
        t = t[t["entity_id"].isin(ids)]
        out |= {e for e, n in zip(t["entity_id"], t["business_name"]) if has_nonlatin(n)}
    return out


def bootstrap_ci(d, n=2000, seed=0):
    """95% percentile bootstrap CI of the mean of per-S1 deltas ``d``."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    m = d[idx].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


class HybridDev:
    def __init__(self, k20=K20, k50=K50, cap=50, native=True):
        from src.evaluation.k_compare import dev_truth
        from src.matching.sample_candidates import OUT

        self.s1 = pd.read_parquet(f"{OUT}/sample_s1.parquet")
        self.truth = dev_truth(self.s1)
        cols = ["s1_id", "cand_id", "rank", "label"]
        self.k20 = pd.read_parquet(f"{k20}/oof.parquet", columns=cols)
        self.k50 = pd.read_parquet(f"{k50}/oof.parquet", columns=cols)
        self.hybrid = hybrid_candidates(self.k20, self.k50, cap=cap)
        deep = self.hybrid[self.hybrid["origin"] == "deep"][["s1_id", "cand_id", "deep_rank"]]
        pos = pd.Series(np.arange(len(self.k50)),
                        index=pd.MultiIndex.from_frame(self.k50[["s1_id", "cand_id"]]))
        self.deep_idx = pos.reindex(pd.MultiIndex.from_frame(deep[["s1_id", "cand_id"]])).to_numpy()
        if np.isnan(self.deep_idx.astype(float)).any():
            raise ValueError("deep candidate missing from the K=50 OOF")
        self.deep = deep.reset_index(drop=True)
        self.deep["label"] = self.k50["label"].to_numpy()[self.deep_idx]
        rnd = self.s1[self.s1.city == ""]
        self.rnd, self.country = rnd.entity_id.tolist(), rnd.country.to_numpy()
        rank = pd.concat([self.k20[["s1_id", "cand_id", "rank"]],
                          self.deep.rename(columns={"deep_rank": "rank"})[["s1_id", "cand_id", "rank"]]])
        self.rank = dict(zip(zip(rank["s1_id"], rank["cand_id"]), rank["rank"]))
        self.in_cands = self.hybrid.groupby("s1_id")["cand_id"].agg(set).to_dict()
        self.native = None
        if native:
            ids = set(self.rnd) | {c for s in self.rnd for c in self.truth[s]}
            self.native = nonlatin_ids(ids)

    def base(self, p20, t20=None):
        """K=20 decisions: exclusivity + threshold (tuned for macro-F0.5 on all sampled S1 when t20 is None, exactly
        as train.py does) -> (pred, owner, t20)."""
        ex = exclusive(self.k20, np.asarray(p20))
        if t20 is None:
            t20, _ = best_threshold(self.k20, ex, self.truth)
        pred = to_matches(self.k20, ex, t20, self.truth)
        return pred, {c: s for s, cs in pred.items() for c in cs}, t20

    def run(self, p20, p50, t20=None, t_deep=DEEP_THRESHOLD, accept=None):
        """Hybrid50 decisions. accept: optional boolean mask over self.deep replacing "p50 >= t_deep"."""
        base_pred, owner, t20 = self.base(p20, t20)
        deep = self.deep.assign(prob=np.asarray(p50)[self.deep_idx])
        mask = deep["prob"].to_numpy() >= t_deep if accept is None else accept
        pred, added = merge_deep(base_pred, owner, deep, mask)
        return {"pred": pred, "base_pred": base_pred, "t20": t20, "added": added}

    def per_s1(self, pred):
        return np.array([entity_f05(pred[s], self.truth[s]) for s in self.rnd])

    def metrics(self, pred, ref=None, boot=True):
        """Headline metrics of ``pred`` on the random S1; ref = per_s1() of the comparison system (paired delta)."""
        per = self.per_s1(pred)
        fp = fn = fn_cand = fp_r13 = fn_native = 0
        for s in self.rnd:
            p, t = pred[s], self.truth[s]
            for c in p - t:
                fp += 1
                fp_r13 += self.rank.get((s, c), 99) <= 3
            for c in t - p:
                fn += 1
                fn_cand += c in self.in_cands.get(s, ())
                if self.native is not None:
                    fn_native += s in self.native or c in self.native
        single = [not self.truth[s] for s in self.rnd]
        r = {"f05": round(float(per.mean()), 5),
             "India": round(float(per[self.country == "India"].mean()), 4),
             "US": round(float(per[self.country == "US"].mean()), 4),
             "singletons_empty": round(float(np.mean([not pred[s] for s, x in zip(self.rnd, single) if x])), 4),
             "fp": fp, "fn": fn, "fn_in_candidates": fn_cand, "fp_rank1_3": fp_r13}
        if self.native is not None:
            r["fn_native_script"] = fn_native
        if ref is not None:
            d = per - ref
            r["delta"] = round(float(d.mean()), 5)
            r["delta_se"] = round(float(d.std(ddof=1) / np.sqrt(len(d))), 5)
            if boot:
                lo, hi = bootstrap_ci(d)
                r["delta_ci95"] = [round(lo, 5), round(hi, 5)]
            for c in ("India", "US"):
                m = self.country == c
                r[f"delta_{c}"] = round(float(d[m].mean()), 5)
            r["s1_improved"], r["s1_harmed"] = int((d > 1e-12).sum()), int((d < -1e-12).sum())
        return r
