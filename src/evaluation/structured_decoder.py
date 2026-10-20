"""Checkpoint 15: structured / adaptive decisions on top of the Hybrid50 OOF probabilities (development only).

    python -m src.evaluation.structured_decoder analyze     # 15A per-S1 probability structure by true-match count
    python -m src.evaluation.structured_decoder adaptive    # 15B cross-fitted adaptive thresholds
    python -m src.evaluation.structured_decoder meta        # 15C cross-fitted second-stage meta-model
    python -m src.evaluation.structured_decoder count       # 15D match-count decoder

One row per Hybrid50 candidate: the K=20 rows (origin 0, K=20 OOF probability) and the deep rows (origin 1, K=50
OOF probability). A decoder gives every row a score and a threshold; decisions keep the production structure:
  base   K=20 rows: the candidate's best-scoring K=20 claimant (ties: smallest s1_id) is accepted if score >= thr
  deep   deep rows: the best deep claimant is accepted if score >= thr and the base did not take the candidate
Cross-fitting: S1 are split into the five GroupKFold folds of the K=20 matcher; every tuned parameter / meta-model
used for a fold's S1 is fitted on the other four folds only. Reported numbers are on the hash-random S1, pooled over
the five held-out folds, paired against the adopted Hybrid50 (0.9525). Never the final holdout.
"""

import argparse
import json
import os
import time

import pandas as pd  # before sklearn (Windows: pyarrow after sklearn can crash)
import numpy as np

from src.evaluation.hybrid_eval import HybridDev, bootstrap_ci
from src.matching.stream import s1_ranks

OUT_DIR = "output/error_analysis"


class Table:
    """The Hybrid50 candidate rows of the development sample with everything a decoder needs, as NumPy arrays."""

    def __init__(self, ctx, p20, p50):
        b = ctx.k20[["s1_id", "cand_id", "rank", "label"]].assign(origin=0, prob=np.asarray(p20, dtype=float),
                                                                  src_row=np.arange(len(ctx.k20)))
        d = ctx.deep[["s1_id", "cand_id", "deep_rank", "label"]].rename(columns={"deep_rank": "rank"})
        d = d.assign(origin=1, prob=np.asarray(p50, dtype=float)[ctx.deep_idx], src_row=ctx.deep_idx)
        self.frame = pd.concat([b, d], ignore_index=True)
        f = self.frame
        s1_ids = ctx.s1.entity_id.to_numpy(dtype=object)
        self.s1_ids = s1_ids
        self.s1 = pd.Index(s1_ids).get_indexer(f["s1_id"])
        if (self.s1 < 0).any():
            raise ValueError("candidate row of an S1 outside the sample")
        self.n_s1 = len(s1_ids)
        self.s1_rank = s1_ranks(s1_ids)[self.s1]
        self.cand, cand_ids = pd.factorize(f["cand_id"])
        self.n_cand = len(cand_ids)
        self.origin = f["origin"].to_numpy()
        self.rank = f["rank"].to_numpy()
        self.label = f["label"].to_numpy().astype(bool)
        self.prob = f["prob"].to_numpy()
        self.src_row = f["src_row"].to_numpy()
        self.n_true = np.array([len(ctx.truth[s]) for s in s1_ids])
        self.random = (ctx.s1.city == "").to_numpy()
        self.country = ctx.s1.country.to_numpy()
        from sklearn.model_selection import GroupKFold

        fold = np.full(self.n_s1, -1)
        k20_s1 = pd.Index(s1_ids).get_indexer(ctx.k20["s1_id"])
        for k, (_, va) in enumerate(GroupKFold(5).split(ctx.k20, groups=ctx.k20["s1_id"])):
            fold[k20_s1[va]] = k
        if (fold < 0).any():
            raise ValueError("S1 without K=20 candidates")
        self.fold = fold  # per S1
        self.row_fold = fold[self.s1]
        self.win = self.winners(self.prob)

    def winners(self, score):
        """Rows that are their candidate's best claimant within their stage (score desc, then smallest s1_id)."""
        win = np.zeros(len(score), dtype=bool)
        for o in (0, 1):
            idx = np.flatnonzero(self.origin == o)
            order = idx[np.lexsort((self.s1_rank[idx], -score[idx], self.cand[idx]))]
            c = self.cand[order]
            win[order[np.r_[True, c[1:] != c[:-1]]]] = True
        return win

    def decide(self, thr, score=None, win=None):
        """Accepted rows under per-row thresholds (score defaults to the first-stage probability)."""
        score = self.prob if score is None else score
        win = (self.win if score is self.prob else self.winners(score)) if win is None else win
        ok = win & (score >= thr)
        acc_b = ok & (self.origin == 0)
        owned = np.zeros(self.n_cand, dtype=bool)
        owned[self.cand[acc_b]] = True
        return acc_b | (ok & (self.origin == 1) & ~owned[self.cand])

    def f05(self, acc):
        """Per-S1 F0.5 (all sampled S1) of the accepted rows, as matcher.macro_f05 / hard_negatives.entity_f05."""
        tp = np.bincount(self.s1[acc], weights=self.label[acc], minlength=self.n_s1)
        npred = np.bincount(self.s1[acc], minlength=self.n_s1)
        nt = self.n_true
        with np.errstate(divide="ignore", invalid="ignore"):
            p, r = tp / npred, tp / nt
            f = 1.25 * p * r / (0.25 * p + r)
        return np.where(nt == 0, (npred == 0).astype(float), np.where(tp == 0, 0.0, f))

    def summary(self, per, ref=None, acc=None):
        """Headline metrics on the random S1 from per-S1 F0.5 values (optionally paired with ref)."""
        m = self.random
        r = {"f05": round(float(per[m].mean()), 5)}
        for c in ("India", "US"):
            r[c] = round(float(per[m & (self.country == c)].mean()), 4)
        single = m & (self.n_true == 0)
        if acc is not None:
            npred = np.bincount(self.s1[acc], minlength=self.n_s1)
            r["singletons_empty"] = round(float((npred[single] == 0).mean()), 4)
            rows = acc & m[self.s1]
            r["fp"] = int((rows & ~self.label).sum())
            r["tp"] = int((rows & self.label).sum())
            r["fn"] = int(self.n_true[m].sum() - r["tp"])
            r["fp_rank1_3"] = int((rows & ~self.label & (self.rank <= 3)).sum())
        if ref is not None:
            d = (per - ref)[m]
            r["delta"] = round(float(d.mean()), 5)
            r["delta_se"] = round(float(d.std(ddof=1) / np.sqrt(len(d))), 5)
            lo, hi = bootstrap_ci(d)
            r["delta_ci95"] = [round(lo, 5), round(hi, 5)]
            for c in ("India", "US"):
                r[f"delta_{c}"] = round(float((per - ref)[m & (self.country == c)].mean()), 5)
            r["s1_improved"], r["s1_harmed"] = int((d > 1e-12).sum()), int((d < -1e-12).sum())
        return r


def load_table(native=False):
    ctx = HybridDev(native=native)
    p20 = pd.read_parquet("output/candidates_p3/k20/oof.parquet", columns=["prob"])["prob"].to_numpy()
    p50 = pd.read_parquet("output/candidates_p3/k50/oof.parquet", columns=["prob"])["prob"].to_numpy()
    return ctx, Table(ctx, p20, p50)


def adopted_thresholds(T, t20=0.7, t_deep=0.85):
    return np.where(T.origin == 0, t20, t_deep)


# ---------------------------------------------------------------- 15A analysis
def s1_aggregates(T, acc):
    """Per-S1 structure of the decision problem (all features are available at test time except n_true)."""
    f = pd.DataFrame({"s1": T.s1, "prob": T.prob, "origin": T.origin, "rank": T.rank, "acc": acc,
                      "s3": T.frame["cand_id"].str.startswith("S3-").to_numpy()})
    f = f.sort_values(["s1", "prob"], ascending=[True, False], kind="mergesort")
    f["pos"] = f.groupby("s1").cumcount()
    top = f[f["pos"] < 3].pivot(index="s1", columns="pos", values="prob").reindex(range(T.n_s1)).fillna(0.0)
    g = f.groupby("s1")
    a = pd.DataFrame(index=range(T.n_s1))
    a["top1"], a["top2"], a["top3"] = top[0], top[1], top[2]
    a["gap12"], a["gap23"] = a.top1 - a.top2, a.top2 - a.top3
    for t in (0.3, 0.5, 0.7, 0.9):
        a[f"n_above_{t}"] = g["prob"].apply(lambda x, t=t: int((x >= t).sum())).reindex(a.index).fillna(0)
    a["n_cands"] = g.size().reindex(a.index).fillna(0)
    a["n_accepted"] = g["acc"].sum().reindex(a.index).fillna(0)
    a["acc_s3"] = f[f.acc].groupby("s1")["s3"].sum().reindex(a.index).fillna(0)
    a["acc_deep"] = f[f.acc].groupby("s1")["origin"].sum().reindex(a.index).fillna(0)
    a["n_true"] = T.n_true
    return a


def analyze(out=f"{OUT_DIR}/structured_15a.json"):
    ctx, T = load_table(native=True)
    acc = T.decide(adopted_thresholds(T))
    per = T.f05(acc)
    print(f"check: adopted Hybrid50 via the vectorised decoder = {per[T.random].mean():.5f}")
    a = s1_aggregates(T, acc)
    # pair-level similarity maxima from the K=20 feature cache (base rows)
    X20 = pd.read_parquet("output/candidates_p3/k20/X_ACE.parquet",
                          columns=["name_tsort", "addr_tsort", "block_score", "script_mismatch", "ph_jacc"])
    base = T.origin == 0
    for col in ("name_tsort", "addr_tsort", "block_score"):
        v = np.full(len(T.prob), -np.inf)
        v[base] = X20[col].to_numpy()[T.src_row[base]]
        a[f"max_{col}"] = pd.Series(v).groupby(T.s1).max().reindex(a.index).to_numpy()
    a["s1_native"] = [s in ctx.native for s in T.s1_ids]
    a["random"], a["country"] = T.random, T.country
    a["bucket"] = np.minimum(a["n_true"], 4)
    r = a[a.random]
    cols = ["top1", "top2", "top3", "gap12", "gap23", "n_above_0.3", "n_above_0.5", "n_above_0.7", "n_above_0.9",
            "max_name_tsort", "max_addr_tsort", "max_block_score", "n_cands", "n_accepted", "acc_s3", "acc_deep",
            "s1_native"]
    table = r.groupby("bucket")[cols].mean().round(3)
    table.insert(0, "n_s1", r.groupby("bucket").size())
    table["f05"] = pd.Series(per[T.random]).groupby(r["bucket"].to_numpy()).mean().round(4)
    conf = pd.crosstab(np.minimum(r["n_true"], 4), np.minimum(r["n_accepted"], 4).astype(int),
                       rownames=["true"], colnames=["accepted"])
    print(table.T.to_string())
    print(conf.to_string())
    # oracle bounds: how much a perfect count / a perfect per-S1 threshold could give (analysis only)
    f = pd.DataFrame({"s1": T.s1, "prob": T.prob, "label": T.label, "win": T.win})
    f = f[f.win].sort_values(["s1", "prob"], ascending=[True, False], kind="mergesort")
    f["pos"] = f.groupby("s1").cumcount()
    oracle = f[f["pos"] < T.n_true[f["s1"].to_numpy()]]
    tp = np.bincount(oracle["s1"], weights=oracle["label"], minlength=T.n_s1)
    npred = np.bincount(oracle["s1"], minlength=T.n_s1)
    with np.errstate(divide="ignore", invalid="ignore"):
        p, rc = tp / npred, tp / T.n_true
        fo = np.where(T.n_true == 0, 1.0, np.where(tp == 0, 0.0, 1.25 * p * rc / (0.25 * p + rc)))
    print(f"oracle 'true count known, take top-N winners by prob': {fo[T.random].mean():.4f}")
    res = {"adopted": T.summary(per, acc=acc), "by_true_count": json.loads(table.to_json(orient="index")),
           "true_vs_accepted": json.loads(conf.to_json(orient="index")),
           "oracle_true_count_topN": round(float(fo[T.random].mean()), 5)}
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2)
    return res


# ---------------------------------------------------------------- 15B adaptive thresholds (cross-fitted)
GRID = np.round(np.arange(0.30, 0.99, 0.01), 2)


def first_match_rows(T):
    """Each S1's highest-probability winning K=20 row (the S1's 'first' match candidate)."""
    idx = np.flatnonzero(T.win & (T.origin == 0))
    order = idx[np.lexsort((-T.prob[idx], T.s1[idx]))]
    first = np.zeros(len(T.prob), dtype=bool)
    s = T.s1[order]
    first[order[np.r_[True, s[1:] != s[:-1]]]] = True
    return first


def threshold_variants(T):
    """name -> (initial params, params -> per-row thresholds). A deliberately small, interpretable set."""
    base, top3 = T.origin == 0, T.rank <= 3
    first = first_match_rows(T)
    return {
        "uniform (t20; deep 0.85 fixed)": ([0.70], lambda p: np.where(base, p[0], 0.85)),
        "uniform + deep tuned (t20, t_deep)": ([0.70, 0.85], lambda p: np.where(base, p[0], p[1])),
        "rank split (rank 1-3, rank 4-20, deep)": ([0.70, 0.70, 0.85],
                                                   lambda p: np.where(base, np.where(top3, p[0], p[1]), p[2])),
        "first vs additional match (first, extra, deep)": ([0.70, 0.70, 0.85],
                                                           lambda p: np.where(base, np.where(first, p[0], p[1]), p[2])),
    }


def coordinate_descent(objective, init, grid=GRID, rounds=3):
    p = list(init)
    best = objective(p)
    for _ in range(rounds):
        changed = False
        for i in range(len(p)):
            for v in grid:
                q = p[:i] + [float(v)] + p[i + 1:]
                f = objective(q)
                if f > best + 1e-12:
                    best, p, changed = f, q, True
        if not changed:
            break
    return p, best


def cross_fit_thresholds(T, make_thr, init):
    """Per fold: tune params on the other four folds' S1 (all sampled S1 there, as train.py tunes), then decide
    every S1 with its own fold's params in ONE global decision. -> (accepted rows, params per fold)."""
    params = []
    for k in range(5):
        train_s1 = T.fold != k

        def objective(p):
            return T.f05(T.decide(make_thr(p)))[train_s1].mean()

        params.append(coordinate_descent(objective, init)[0])
    thr = np.zeros(len(T.prob))
    for k, p in enumerate(params):
        m = T.row_fold == k
        thr[m] = make_thr(p)[m]
    return T.decide(thr), params


def adaptive(out=f"{OUT_DIR}/structured_15b.json"):
    ctx, T = load_table()
    ref_acc = T.decide(adopted_thresholds(T))
    ref = T.f05(ref_acc)
    print(f"adopted Hybrid50 {ref[T.random].mean():.5f}", flush=True)
    res = {"adopted": T.summary(ref, acc=ref_acc)}
    for name, (init, make_thr) in threshold_variants(T).items():
        t0 = time.time()
        acc, params = cross_fit_thresholds(T, make_thr, init)
        r = T.summary(T.f05(acc), ref=ref, acc=acc)
        r["params_per_fold"] = params
        r["seconds"] = round(time.time() - t0, 1)
        res[name] = r
        print(f"{name}: {json.dumps(r)}", flush=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2)
    return res


# ---------------------------------------------------------------- 15C second-stage meta-model (nested cross-fit)
META_FLOOR = 0.02  # rows below this first-stage probability are never accepted (no threshold gets near it)
GRID_META = np.round(np.arange(0.20, 0.99, 0.01), 2)


def s1_context(T):
    """Per-row S1 context from the first-stage OOF probabilities of all the S1's Hybrid50 rows (test-available)."""
    f = pd.DataFrame({"s1": T.s1, "prob": T.prob, "origin": T.origin})
    f = f.sort_values(["s1", "prob"], ascending=[True, False], kind="mergesort")
    f["pos"] = f.groupby("s1").cumcount()
    top = f[f["pos"] < 3].pivot(index="s1", columns="pos", values="prob").reindex(range(T.n_s1)).fillna(0.0)
    per = pd.DataFrame({"s1_top1": top[0], "s1_top2": top[1], "s1_top3": top[2]})
    g = f.groupby("s1")["prob"]
    for t in (0.3, 0.5, 0.7, 0.9):
        per[f"s1_n_above_{t}"] = f.assign(x=f["prob"] >= t).groupby("s1")["x"].sum().reindex(per.index).fillna(0)
    per["s1_prob_sum"] = g.sum().reindex(per.index).fillna(0)
    per["s1_base_top1"] = f[f.origin == 0].groupby("s1")["prob"].max().reindex(per.index).fillna(0)
    per["s1_deep_top1"] = f[f.origin == 1].groupby("s1")["prob"].max().reindex(per.index).fillna(0)
    per["s1_n_cands"] = g.size().reindex(per.index).fillna(0)
    rows = per.iloc[T.s1].reset_index(drop=True)
    rows["prob_pos_in_s1"] = f["pos"].reindex(range(len(T.prob))).to_numpy()  # 0 = the S1's best row
    rows["top1_minus_prob"] = rows["s1_top1"] - T.prob
    rows["gap12"] = rows["s1_top1"] - rows["s1_top2"]
    return rows


def meta_matrix(T, rows, pair_features=True):
    ctx = s1_context(T).iloc[rows].reset_index(drop=True)
    cols = {"prob": T.prob[rows], "origin": T.origin[rows], "blocker_rank": T.rank[rows]}
    M = pd.concat([pd.DataFrame(cols), ctx], axis=1)
    if pair_features:
        X20 = pd.read_parquet("output/candidates_p3/k20/X_ACE.parquet")
        X50 = pd.read_parquet("output/candidates_p3/k50/X_ACE.parquet")
        feats = [c for c in X20.columns if not c.startswith("__") and c != "rank"]
        if feats != [c for c in X50.columns if not c.startswith("__") and c != "rank"]:
            raise ValueError("K=20 and K=50 feature caches have different columns")
        A = np.empty((len(rows), len(feats)), dtype=np.float32)
        b = T.origin[rows] == 0
        A[b] = X20[feats].to_numpy()[T.src_row[rows][b]]
        A[~b] = X50[feats].to_numpy()[T.src_row[rows][~b]]
        M = pd.concat([M, pd.DataFrame(A, columns=feats)], axis=1)
    return M.astype(np.float32)


def cross_fit_meta(T, M, rows, log=print):
    """Nested cross-fit: for every outer fold k, four inner models (fit on three of the other folds) give clean
    scores for tuning (t_base, t_deep) on the four training folds; the outer model (fit on all four) scores fold k."""
    from src.matching.matcher import train

    y, fr = T.label[rows], T.row_fold[rows]
    score, params = np.full(len(T.prob), -np.inf), []
    for k in range(5):
        inner = np.full(len(T.prob), -np.inf)
        for j in range(5):
            if j != k:
                tr, va = (fr != k) & (fr != j), fr == j
                inner[rows[va]] = train(M[tr], y[tr]).predict_proba(M[va])[:, 1]
        win, train_s1 = T.winners(inner), T.fold != k

        def objective(p):
            return T.f05(T.decide(np.where(T.origin == 0, p[0], p[1]), score=inner, win=win))[train_s1].mean()

        p, _ = coordinate_descent(objective, [0.5, 0.5], grid=GRID_META)
        params.append(p)
        tr, va = fr != k, fr == k
        score[rows[va]] = train(M[tr], y[tr]).predict_proba(M[va])[:, 1]
        log(f"  fold {k}: thresholds {p}")
    thr = np.zeros(len(T.prob))
    for k, p in enumerate(params):
        m = T.row_fold == k
        thr[m] = np.where(T.origin[m] == 0, p[0], p[1])
    return T.decide(thr, score=score), params, score


def meta(out=f"{OUT_DIR}/structured_15c.json"):
    ctx, T = load_table()
    ref_acc = T.decide(adopted_thresholds(T))
    ref = T.f05(ref_acc)
    rows = np.flatnonzero(T.prob >= META_FLOOR)
    print(f"adopted Hybrid50 {ref[T.random].mean():.5f}; meta rows (prob >= {META_FLOOR}): {len(rows)} "
          f"({T.label[rows].sum()} true of {T.label.sum()})", flush=True)
    res = {"adopted": T.summary(ref, acc=ref_acc), "meta_rows": int(len(rows))}
    for name, pf in (("meta: prob + S1 context", False), ("meta: prob + S1 context + pair features", True)):
        t0 = time.time()
        M = meta_matrix(T, rows, pair_features=pf)
        acc, params, _ = cross_fit_meta(T, M, rows)
        r = T.summary(T.f05(acc), ref=ref, acc=acc)
        r.update(params_per_fold=params, n_features=M.shape[1], seconds=round(time.time() - t0, 1))
        res[name] = r
        print(f"{name}: {json.dumps(r)}", flush=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2)
    return res


def strict_meta(first="M0", decoy=False, first_decoy=False, log=print):
    """Strictly nested version of the pair-feature meta-model: for every outer fold k the FIRST-STAGE K=20 and K=50
    matchers are retrained without fold k (inner 4-fold OOF for the training S1, a model on all four folds for fold
    k), so no model, threshold or meta-model that touches fold k ever saw a fold-k label. The same framework gives
    the matching first-stage baseline (t20 tuned on the training folds, deep >= 0.85) for a like-for-like delta."""
    from src.matching.ensemble_experiment import MODELS, feature_cache
    from src.matching.matcher import train

    tag = first + ("_decoy" if decoy else "") + ("_firstdecoy" if first_decoy else "")
    out = f"{OUT_DIR}/structured_15c_strict_{tag}.json"
    if first == "XGB":  # checkpoint D: the one predefined XGBoost configuration (src/evaluation/gbdt_experiment.py)
        from xgboost import XGBClassifier

        from src.evaluation.gbdt_experiment import XGB

        def fit_first(X, y):
            return XGBClassifier(**XGB).fit(np.asarray(X, dtype=np.float32), y)
    else:
        params = MODELS[first]  # first-stage matcher configuration (checkpoint 14)

        def fit_first(X, y):
            return train(X, y, **params)

    def prob_first(model, X):
        return model.predict_proba(np.asarray(X, dtype=np.float32) if first == "XGB" else X)[:, 1]
    ctx = HybridDev(native=False)
    p20_ref = pd.read_parquet("output/candidates_p3/k20/oof.parquet", columns=["prob"])["prob"].to_numpy()
    p50_ref = pd.read_parquet("output/candidates_p3/k50/oof.parquet", columns=["prob"])["prob"].to_numpy()
    T0 = Table(ctx, p20_ref, p50_ref)
    ref = T0.f05(T0.decide(adopted_thresholds(T0)))
    m20, X20 = feature_cache("output/candidates_p3/k20")
    m50, X50 = feature_cache("output/candidates_p3/k50")
    for m, k in ((m20, ctx.k20), (m50, ctx.k50)):
        if not (np.array_equal(m["s1_id"].to_numpy(), k["s1_id"].to_numpy())
                and np.array_equal(m["cand_id"].to_numpy(), k["cand_id"].to_numpy())):
            raise ValueError("feature cache rows are not aligned with the OOF rows")
    if decoy or first_decoy:  # token-alignment features (src/matching/decoy_features.py), all rows, cached
        from src.evaluation.decoy_experiment import all_row_decoy_features

        D20, D50 = all_row_decoy_features(ctx)
    if first_decoy:
        X20, X50 = pd.concat([X20, D20], axis=1), pd.concat([X50, D50], axis=1)
    s1_fold = dict(zip(T0.s1_ids, T0.fold))
    f20 = ctx.k20["s1_id"].map(s1_fold).to_numpy()
    f50 = ctx.k50["s1_id"].map(s1_fold).to_numpy()
    y20, y50 = ctx.k20["label"].to_numpy(), ctx.k50["label"].to_numpy()
    p20_fin, p50_fin = np.zeros(len(y20)), np.zeros(len(y50))
    score_fin = np.full(len(T0.prob), -np.inf)
    t_base, t_meta = {}, {}
    t_start = time.time()
    for k in range(5):
        p20, p50 = np.zeros(len(y20)), np.zeros(len(y50))
        for j in range(5):  # first stage, inner OOF for the training folds
            if j == k:
                continue
            for X, y, f, p in ((X20, y20, f20, p20), (X50, y50, f50, p50)):
                tr, va = (f != k) & (f != j), f == j
                p[va] = prob_first(fit_first(X[tr], y[tr]), X[va])
        for X, y, f, p, fin in ((X20, y20, f20, p20, p20_fin), (X50, y50, f50, p50, p50_fin)):
            tr, va = f != k, f == k  # first stage for fold k: trained on the four other folds only
            p[va] = fin[va] = prob_first(fit_first(X[tr], y[tr]), X[va])
        log(f"  fold {k}: first stage done, {time.time() - t_start:.0f}s")
        T = Table(ctx, p20, p50)
        train_s1 = T.fold != k
        # baseline: t20 tuned on the training folds (deep fixed at 0.85)
        t_base[k] = coordinate_descent(lambda q: T.f05(T.decide(np.where(T.origin == 0, q[0], 0.85)))[train_s1]
                                       .mean(), [0.70])[0][0]
        rows = np.flatnonzero(T.prob >= META_FLOOR)
        M = meta_matrix(T, rows, pair_features=True)
        if decoy:
            b = T.origin[rows] == 0
            Dr = np.empty((len(rows), D20.shape[1]), dtype=np.float32)
            Dr[b] = D20.to_numpy()[T.src_row[rows][b]]
            Dr[~b] = D50.to_numpy()[T.src_row[rows][~b]]
            M = pd.concat([M, pd.DataFrame(Dr, columns=D20.columns)], axis=1)
        y, fr = T.label[rows], T.row_fold[rows]
        inner = np.full(len(T.prob), -np.inf)
        for j in range(5):
            if j != k:
                tr, va = (fr != k) & (fr != j), fr == j
                inner[rows[va]] = train(M[tr], y[tr]).predict_proba(M[va])[:, 1]
        win = T.winners(inner)
        t_meta[k] = coordinate_descent(
            lambda q: T.f05(T.decide(np.where(T.origin == 0, q[0], q[1]), score=inner, win=win))[train_s1].mean(),
            [0.5, 0.5], grid=GRID_META)[0]
        tr, va = fr != k, fr == k
        score_fin[rows[va]] = train(M[tr], y[tr]).predict_proba(M[va])[:, 1]
        log(f"  fold {k}: t20 {t_base[k]}, meta thresholds {t_meta[k]}, {time.time() - t_start:.0f}s")
    TF = Table(ctx, p20_fin, p50_fin)  # every row scored by first-stage models that never saw its fold
    thr_b = np.where(TF.origin == 0, np.array([t_base[f] for f in range(5)])[TF.row_fold], 0.85)
    acc_b = TF.decide(thr_b)
    per_b = TF.f05(acc_b)
    tm = np.array([t_meta[f] for f in range(5)])
    thr_m = np.where(TF.origin == 0, tm[TF.row_fold, 0], tm[TF.row_fold, 1])
    acc_m = TF.decide(thr_m, score=score_fin)
    per_m = TF.f05(acc_m)
    res = {"first_stage": first, "decoy_meta": decoy, "decoy_first_stage": first_decoy,
           "adopted_reference": TF.summary(ref, acc=None),
           "strict baseline (first stage, t20 per fold, deep 0.85)": TF.summary(per_b, ref=ref, acc=acc_b),
           "strict meta (prob + S1 context + pair features)": TF.summary(per_m, ref=ref, acc=acc_m),
           "strict meta vs strict baseline": TF.summary(per_m, ref=per_b),
           "t20_per_fold": t_base, "meta_thresholds_per_fold": t_meta, "seconds": round(time.time() - t_start, 1)}
    per_fold = {}
    for f in range(5):
        m = TF.random & (TF.fold == f)
        per_fold[f] = round(float((per_m - per_b)[m].mean()), 5)
    res["meta_minus_baseline_per_fold"] = per_fold
    # per-row held-out decisions (git-ignored) for the oracle-gap audit (src/evaluation/oracle_gap.py)
    TF.frame.assign(first_prob=TF.prob, meta_score=score_fin, meta_threshold=thr_m, accepted=acc_m,
                    base_threshold=thr_b, base_accepted=acc_b, fold=TF.row_fold).to_parquet(
        f"{OUT_DIR}/strict_rows_{tag}.parquet")
    for k_, v in res.items():
        print(f"{k_}: {json.dumps(v)}", flush=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2)
    return res


# ---------------------------------------------------------------- checkpoint 17: competition / ownership
def claimant_stats(T, score):
    """Test-available competition signals per row, over ALL Hybrid50 rows of every sampled S1 that list the same
    candidate: number of claiming S1, the best competing claimant's score and this row's margin over it."""
    f = pd.DataFrame({"cand": T.cand, "s": score, "s1": T.s1})
    n = f.groupby("cand")["s1"].transform("nunique").to_numpy()
    o = np.lexsort((-score, T.cand))
    c = T.cand[o]
    first = np.r_[True, c[1:] != c[:-1]]
    best = np.maximum.reduceat(score[o], np.flatnonzero(first))
    cnt = np.diff(np.r_[np.flatnonzero(first), len(o)])
    second = np.full(len(best), -np.inf)
    has2 = cnt > 1
    second[has2] = score[o][np.flatnonzero(first)[has2] + 1]
    gi = np.cumsum(first) - 1  # candidate group of each sorted row
    b, s2 = np.empty(len(score)), np.empty(len(score))
    b[o], s2[o] = best[gi], second[gi]
    other = np.where(score >= b, s2, b)  # best competitor of this row (ties: the tied row)
    return n, other, score - other


def competition(out=f"{OUT_DIR}/competition_17.json"):
    """How much global ownership matters and whether a claimant-margin rule helps, measured where competition
    exists (the city S1: every S1 of two cities is in the sample) and on the random S1. Ground-truth ownership is
    used only to DESCRIBE the false positives, never in a decision rule."""
    ctx, T = load_table()
    gt = pd.read_csv("data/raw/train/train_ground_truth.tsv", sep="\t", dtype=str, keep_default_na=False)
    owner = {c: s for s, m in zip(gt.source1_entity_id, gt.matched_entity_ids) for c in filter(None, m.split(","))}
    del gt
    thr = adopted_thresholds(T)
    n_cl, other, margin = claimant_stats(T, T.prob)
    all_win = np.ones(len(T.prob), dtype=bool)
    variants = {"adopted (exclusivity)": T.decide(thr),
                "no exclusivity (every row competes alone)": T.decide(thr, win=all_win)}
    for m in (0.05, 0.1, 0.2):
        acc = T.decide(thr)
        variants[f"exclusivity + winner margin >= {m} over the best competing claimant"] = acc & (margin >= m)
    city = ~T.random
    res = {}
    ref_per = T.f05(variants["adopted (exclusivity)"])
    cand_ids = T.frame["cand_id"].to_numpy()
    s1_ids = T.frame["s1_id"].to_numpy()
    for name, acc in variants.items():
        per = T.f05(acc)
        fp = acc & ~T.label
        claimed_other = np.array([owner.get(c, s) != s for c, s in zip(cand_ids[fp], s1_ids[fp])])
        r = {}
        for scope, m in (("random S1", T.random), ("city S1 (competition-rich)", city)):
            rows = m[T.s1]
            d = (per - ref_per)[m]
            fpm = fp & rows
            r[scope] = {"f05": round(float(per[m].mean()), 5), "delta_vs_adopted": round(float(d.mean()), 5),
                        "delta_se": round(float(d.std(ddof=1) / np.sqrt(len(d))), 5),
                        "fp": int(fpm.sum()), "tp": int((acc & T.label & rows).sum()),
                        "fp_candidate_is_another_S1s_true_match":
                            int(claimed_other[rows[fp]].sum()),
                        "fp_with_competing_claimant_in_sample": int((fpm & (n_cl > 1)).sum())}
        res[name] = r
        print(f"{name}: {json.dumps(r)}", flush=True)
    rows = T.random[T.s1]
    res["claimants_per_accepted_candidate"] = {
        scope: {"mean": round(float(n_cl[variants['adopted (exclusivity)'] & m[T.s1]].mean()), 3),
                "share_with_competitor": round(float((n_cl[variants['adopted (exclusivity)'] & m[T.s1]] > 1).mean()), 4)}
        for scope, m in (("random S1", T.random), ("city S1", city))}
    print(json.dumps(res["claimants_per_accepted_candidate"]))
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2)
    return res


# ---------------------------------------------------------------- 15D match-count decoder
def count_features(T, acc_ref):
    """S1-level aggregates of the first-stage probabilities (test-available)."""
    f = pd.DataFrame({"s1": T.s1, "prob": T.prob, "origin": T.origin})
    f = f.sort_values(["s1", "prob"], ascending=[True, False], kind="mergesort")
    f["pos"] = f.groupby("s1").cumcount()
    top = f[f["pos"] < 6].pivot(index="s1", columns="pos", values="prob").reindex(range(T.n_s1)).fillna(0.0)
    a = pd.DataFrame({f"top{i + 1}": top[i] for i in range(6)})
    for i in range(5):
        a[f"gap{i + 1}{i + 2}"] = a[f"top{i + 1}"] - a[f"top{i + 2}"]
    for t in (0.1, 0.3, 0.5, 0.7, 0.9):
        a[f"n_above_{t}"] = f.assign(x=f["prob"] >= t).groupby("s1")["x"].sum().reindex(a.index).fillna(0)
    a["prob_sum"] = f.groupby("s1")["prob"].sum().reindex(a.index).fillna(0)
    a["base_top1"] = f[f.origin == 0].groupby("s1")["prob"].max().reindex(a.index).fillna(0)
    a["deep_top1"] = f[f.origin == 1].groupby("s1")["prob"].max().reindex(a.index).fillna(0)
    a["n_accepted_ref"] = np.bincount(T.s1[acc_ref], minlength=T.n_s1)
    a["india"] = (T.country == "India").astype(float)
    return a.astype(np.float32)


def _first_n(T, rows, n):
    """Of the candidate rows (indices), the first n[s1] per S1 by probability."""
    rows = rows[np.lexsort((-T.prob[rows], T.s1[rows]))]
    pos = pd.Series(T.s1[rows]).groupby(T.s1[rows]).cumcount().to_numpy()
    return rows[pos < n[T.s1[rows]]]


def count_decode(T, n_pred, mode, acc_ref, fill_floor=0.5):
    """Accepted rows given a predicted match count per S1 (class 4 = '4 or more': no cap beyond the adopted rule).
    top_n:    the S1's N best winning K=20 rows by probability, then deep rows (base priority) up to N;
    cap:      the adopted threshold decisions, keeping at most the N most probable per S1 (N = 0 empties the S1);
    cap_fill: cap, then fill up to N with the next best winning rows whose probability >= fill_floor."""
    cap = np.where(n_pred == 4, 10 ** 6, n_pred)
    if mode == "top_n":
        four_plus = n_pred[T.s1] == 4  # '4 or more': the 4 best, plus whatever else the adopted rule accepts
        acc = np.zeros(len(T.prob), dtype=bool)
        acc[_first_n(T, np.flatnonzero(T.win & (T.origin == 0)), n_pred)] = True
        acc |= acc_ref & (T.origin == 0) & four_plus
        owned = np.zeros(T.n_cand, dtype=bool)
        owned[T.cand[acc]] = True
        left = np.maximum(n_pred - np.bincount(T.s1[acc], minlength=T.n_s1), 0)
        deep = np.flatnonzero(T.win & (T.origin == 1) & ~owned[T.cand])
        acc[_first_n(T, deep, left)] = True
        acc |= acc_ref & (T.origin == 1) & four_plus & ~owned[T.cand]
        return acc
    keep = np.zeros(len(T.prob), dtype=bool)
    keep[_first_n(T, np.flatnonzero(acc_ref), cap)] = True
    if mode == "cap":
        return keep
    taken = np.zeros(T.n_cand, dtype=bool)
    taken[T.cand[keep]] = True
    owned = np.zeros(T.n_cand, dtype=bool)
    owned[T.cand[acc_ref & (T.origin == 0)]] = True  # base priority as in the adopted decisions
    ok = T.win & ~keep & (T.prob >= fill_floor) & ~taken[T.cand] & ~((T.origin == 1) & owned[T.cand])
    left = np.maximum(cap - np.bincount(T.s1[keep], minlength=T.n_s1), 0)
    add = _first_n(T, np.flatnonzero(ok), left)
    add = add[~pd.Series(T.cand[add]).duplicated().to_numpy()]  # a base and a deep winner of one candidate
    keep[add] = True
    return keep


def count(out=f"{OUT_DIR}/structured_15d.json"):
    from sklearn.ensemble import HistGradientBoostingClassifier

    ctx, T = load_table()
    ref_acc = T.decide(adopted_thresholds(T))
    ref = T.f05(ref_acc)
    A = count_features(T, ref_acc)
    y = np.minimum(T.n_true, 4)
    n_pred = np.zeros(T.n_s1, dtype=int)
    for k in range(5):  # cross-fitted by S1: the count of a fold's S1 comes from a model fit on the other folds
        tr, va = T.fold != k, T.fold == k
        m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, early_stopping=True, random_state=0)
        n_pred[va] = m.fit(A[tr], y[tr]).predict(A[va])
    rnd = T.random
    ref_count = np.minimum(A["n_accepted_ref"].to_numpy().astype(int), 4)
    res = {"adopted": T.summary(ref, acc=ref_acc),
           "count_accuracy_random": round(float((n_pred[rnd] == y[rnd]).mean()), 4),
           "adopted_rule_count_accuracy_random": round(float((ref_count[rnd] == y[rnd]).mean()), 4),
           "confusion_true_vs_pred": json.loads(pd.crosstab(y[rnd], n_pred[rnd], rownames=["true"],
                                                            colnames=["pred"]).to_json(orient="index"))}
    print(json.dumps({k: res[k] for k in ("count_accuracy_random", "adopted_rule_count_accuracy_random")}), flush=True)
    for mode in ("top_n", "cap", "cap_fill"):
        acc = count_decode(T, n_pred, mode, ref_acc)
        if len(np.unique(T.cand[acc])) != acc.sum():
            raise AssertionError(f"{mode}: a candidate accepted twice")
        r = T.summary(T.f05(acc), ref=ref, acc=acc)
        res[f"count decoder: {mode}"] = r
        print(f"count decoder {mode}: {json.dumps(r)}", flush=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["analyze", "adaptive", "meta", "strict_meta", "count", "competition"])
    ap.add_argument("--first", default="M0", help="strict_meta: first-stage model (M0-M3 of checkpoint 14, or XGB)")
    ap.add_argument("--decoy", action="store_true", help="strict_meta: token-alignment features in the meta-model")
    ap.add_argument("--first-decoy", action="store_true", help="strict_meta: ... and in the first-stage matchers")
    a = ap.parse_args(argv)
    t0 = time.time()
    if a.stage == "strict_meta":
        strict_meta(a.first, decoy=a.decoy, first_decoy=a.first_decoy)
    else:
        {"analyze": analyze, "adaptive": adaptive, "meta": meta, "count": count, "competition": competition}[a.stage]()
    print(f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
