"""Checkpoint C: does the token-alignment ("decoy") feature group help the meta decoder? (development only)

    python -m src.evaluation.decoy_experiment

Controlled comparison on cached artifacts: the same M3 first-stage OOF, the same Hybrid50 rows >= META_FLOOR, the same
K=20 matcher S1 folds and the same nested per-fold threshold tuning (src/pipeline/train_meta.py protocol); the only
difference is whether the meta-model also gets src/matching/decoy_features.py. Paired per-S1 deltas with a bootstrap
CI on the random development S1. A winning variant is then confirmed with the strictly nested protocol.
"""

import argparse
import json
import os
import time

import pandas as pd  # before sklearn (Windows: pyarrow after sklearn can crash)
import numpy as np

from src.evaluation.hybrid_eval import HybridDev
from src.evaluation.structured_decoder import GRID_META, Table, coordinate_descent
from src.matching.matcher import train

K20, K50 = "output/candidates_p3/k20", "output/candidates_p3/k50"
OUT_DIR = "output/error_analysis"


def dev_decoy_features(T, rows, cache=f"{OUT_DIR}/decoy_features_dev.parquet"):
    """Decoy features of the development rows >= floor (cached, keyed by the rows' (s1_id, cand_id))."""
    from src.evaluation.blocker_rescue import load_records
    from src.matching.decoy_features import decoy_features

    pairs = T.frame.iloc[rows][["s1_id", "cand_id"]].reset_index(drop=True)
    if os.path.exists(cache):
        c = pd.read_parquet(cache)
        if c[["s1_id", "cand_id"]].equals(pairs):
            return c.drop(columns=["s1_id", "cand_id"]), None
    t0 = time.time()
    recs = load_records(set(pairs["s1_id"]) | set(pairs["cand_id"]))
    D = decoy_features(pairs, recs).reset_index(drop=True)
    pd.concat([pairs, D], axis=1).to_parquet(cache)  # git-ignored
    return D, round(time.time() - t0, 1)


def all_row_decoy_features(ctx, cache=f"{OUT_DIR}/decoy_features_all.parquet", log=print):
    """Decoy features of every K=20 and K=50 development pair -> (D20 aligned to ctx.k20, D50 aligned to ctx.k50).
    Built once (records only, no model), cached and checked against the runs' (s1_id, cand_id) rows."""
    from src.evaluation.blocker_rescue import load_records
    from src.matching.decoy_features import decoy_features

    pairs = pd.concat([ctx.k50[["s1_id", "cand_id"]], ctx.k20[["s1_id", "cand_id"]]]).drop_duplicates()
    pairs = pairs.reset_index(drop=True)
    if os.path.exists(cache):
        c = pd.read_parquet(cache)
        if not c[["s1_id", "cand_id"]].equals(pairs):
            raise SystemExit(f"{cache} does not match the development runs; delete it to rebuild")
    else:
        t0 = time.time()
        recs = load_records(set(pairs["s1_id"]) | set(pairs["cand_id"]))
        c = pd.concat([pairs, decoy_features(pairs, recs).reset_index(drop=True)], axis=1)
        c.to_parquet(cache)  # git-ignored
        log(f"decoy features for {len(pairs)} pairs built in {time.time() - t0:.0f}s")
    key = pd.MultiIndex.from_frame(c[["s1_id", "cand_id"]])
    feats = c.drop(columns=["s1_id", "cand_id"])
    out = []
    for run in (ctx.k20, ctx.k50):
        pos = pd.Index(key).get_indexer(pd.MultiIndex.from_frame(run[["s1_id", "cand_id"]]))
        out.append(feats.iloc[pos].reset_index(drop=True))
    return out


def cross_fit(T, M, rows):
    """5-fold meta OOF + thresholds tuned per fold on the other folds -> accepted rows."""
    y, fr = T.label[rows], T.row_fold[rows]
    oof = np.full(len(T.prob), -np.inf)
    for k in range(5):
        tr, va = fr != k, fr == k
        oof[rows[va]] = train(M[tr], y[tr]).predict_proba(M[va])[:, 1]
    win = T.winners(oof)
    tm = []
    for k in range(5):
        s1m = T.fold != k
        tm.append(coordinate_descent(
            lambda q: T.f05(T.decide(np.where(T.origin == 0, q[0], q[1]), score=oof, win=win))[s1m].mean(),
            [0.5, 0.5], grid=GRID_META)[0])
    tm = np.array(tm)
    return T.decide(np.where(T.origin == 0, tm[T.row_fold, 0], tm[T.row_fold, 1]), score=oof, win=win), tm, oof


def first_stage_oof(ctx, D20, D50, log=print):
    """M3 OOF with the decoy features added to the first-stage matchers (same GroupKFold as train.py), cached."""
    from sklearn.model_selection import GroupKFold

    from src.matching.feature_cache import feature_cache
    from src.matching.matcher import PRODUCTION_PARAMS

    out = {}
    for name, d, D in (("k20", K20, D20), ("k50", K50, D50)):
        path = os.path.join(d, "oof_M3_decoy.parquet")
        if os.path.exists(path):
            out[name] = pd.read_parquet(path)["prob"].to_numpy()
            continue
        t0 = time.time()
        meta, X = feature_cache(d)
        X = pd.concat([X, D], axis=1)
        y, groups = meta["label"].to_numpy(), meta["s1_id"].to_numpy()
        p = np.zeros(len(X))
        for tr, va in GroupKFold(5).split(X, y, groups):
            p[va] = train(X.iloc[tr], y[tr], **PRODUCTION_PARAMS).predict_proba(X.iloc[va])[:, 1]
        meta.assign(prob=p).to_parquet(path)  # git-ignored
        out[name] = p
        log(f"  first-stage M3 + decoy OOF {name}: {time.time() - t0:.0f}s")
    return out["k20"], out["k50"]


def main(argv=None):
    import pickle

    from src.pipeline.train_meta import dev_meta_rows

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=None)
    ap.add_argument("--first-stage", action="store_true", help="also add the decoy features to the M3 matchers")
    a = ap.parse_args(argv)
    if a.first_stage:
        return main_first_stage(a.out or f"{OUT_DIR}/decoy_experiment_first_stage.json")
    a.out = a.out or f"{OUT_DIR}/decoy_experiment.json"
    t0 = time.time()
    ctx = HybridDev(native=False)
    p20 = pd.read_parquet(f"{K20}/oof.parquet", columns=["prob"])["prob"].to_numpy()
    p50 = pd.read_parquet(f"{K50}/oof.parquet", columns=["prob"])["prob"].to_numpy()
    T = Table(ctx, p20, p50)
    models = {}
    for role, d in (("base", K20), ("deep", K50)):
        with open(os.path.join(d, "matcher.pkl"), "rb") as fh:
            models[role] = pickle.load(fh)
    rows, M, _ = dev_meta_rows(T, models["base"], models["deep"])
    t_matrix = time.time() - t0
    D, t_feat = dev_decoy_features(T, rows)
    ref_acc = T.decide(np.where(T.origin == 0, models["base"]["threshold"], 0.85))
    ref = T.f05(ref_acc)
    res = {"first stage only (M3 Hybrid50)": T.summary(ref, acc=ref_acc)}
    per = {}
    for name, X in (("meta (production features)", M), ("meta + decoy features", pd.concat([M, D], axis=1))):
        t1 = time.time()
        acc, tm, _ = cross_fit(T, X, rows)
        per[name] = T.f05(acc)
        r = T.summary(per[name], ref=ref, acc=acc)
        r.update(thresholds_per_fold=tm.tolist(), n_features=X.shape[1], seconds=round(time.time() - t1, 1))
        res[name] = r
        print(f"{name}: {json.dumps(r)}", flush=True)
    res["decoy vs production meta (paired)"] = T.summary(per["meta + decoy features"],
                                                         ref=per["meta (production features)"])
    print("paired:", json.dumps(res["decoy vs production meta (paired)"]))
    res["seconds"] = {"cached matrix": round(t_matrix, 1), "decoy features (first build)": t_feat,
                      "total": round(time.time() - t0, 1)}
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2)


def decoy_rows(T, rows, D20, D50):
    b = T.origin[rows] == 0
    Dr = np.empty((len(rows), D20.shape[1]), dtype=np.float32)
    Dr[b] = D20.to_numpy()[T.src_row[rows][b]]
    Dr[~b] = D50.to_numpy()[T.src_row[rows][~b]]
    return pd.DataFrame(Dr, columns=D20.columns)


def main_first_stage(out):
    """Decoy features in the M3 first stage too: first-stage-only Hybrid50 (nested t20) and meta + decoy on top."""
    import pickle

    from src.pipeline.train_meta import dev_meta_rows

    t0 = time.time()
    ctx = HybridDev(native=False)
    D20, D50 = all_row_decoy_features(ctx)
    models = {}
    for role, d in (("base", K20), ("deep", K50)):
        with open(os.path.join(d, "matcher.pkl"), "rb") as fh:
            models[role] = pickle.load(fh)
    p20 = pd.read_parquet(f"{K20}/oof.parquet", columns=["prob"])["prob"].to_numpy()
    p50 = pd.read_parquet(f"{K50}/oof.parquet", columns=["prob"])["prob"].to_numpy()
    T0 = Table(ctx, p20, p50)
    ref_acc = T0.decide(np.where(T0.origin == 0, models["base"]["threshold"], 0.85))
    ref = T0.f05(ref_acc)
    rows0, M0, _ = dev_meta_rows(T0, models["base"], models["deep"])
    acc_m0, _, _ = cross_fit(T0, pd.concat([M0, decoy_rows(T0, rows0, D20, D50)], axis=1), rows0)
    per_m0 = T0.f05(acc_m0)
    t1 = time.time()
    q20, q50 = first_stage_oof(ctx, D20, D50)
    t_first = time.time() - t1
    T = Table(ctx, q20, q50)
    tb = [coordinate_descent(lambda q: T.f05(T.decide(np.where(T.origin == 0, q[0], 0.85)))[T.fold != k].mean(),
                             [0.70])[0][0] for k in range(5)]
    acc_b = T.decide(np.where(T.origin == 0, np.array(tb)[T.row_fold], 0.85))
    rows, M, _ = dev_meta_rows(T, models["base"], models["deep"])
    acc_m, tm, _ = cross_fit(T, pd.concat([M, decoy_rows(T, rows, D20, D50)], axis=1), rows)
    res = {"reference: M3 first stage only (0.9558)": T0.summary(ref, acc=ref_acc),
           "M3 first stage + meta + decoy (cross-fitted)": T0.summary(per_m0, ref=ref, acc=acc_m0),
           "M3+decoy first stage only (t20 per fold, deep 0.85)": T.summary(T.f05(acc_b), ref=ref, acc=acc_b),
           "M3+decoy first stage + meta + decoy (cross-fitted)": T.summary(T.f05(acc_m), ref=ref, acc=acc_m),
           "decoy first stage vs M3 first stage (both + meta + decoy, paired)": T.summary(T.f05(acc_m), ref=per_m0),
           "t20_per_fold": tb, "meta_thresholds_per_fold": tm.tolist(),
           "seconds": {"first-stage OOF (K=20 + K=50, cached features)": round(t_first, 1),
                       "total": round(time.time() - t0, 1)}}
    for k, v in res.items():
        print(f"{k}: {json.dumps(v)}", flush=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2)


if __name__ == "__main__":
    main()
