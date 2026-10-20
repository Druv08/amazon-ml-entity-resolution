"""Checkpoint D: one controlled test of a stronger GBDT family for the first stage (development only).

    python -m src.evaluation.gbdt_experiment

XGBoost (already installed; no new dependency) against the production HGB (M3), on exactly the same cached matrices
(61 matcher features + the 13 token-alignment features), the same GroupKFold(5) as train.py, the same Hybrid50
decision (K=20 threshold tuned per fold on the other folds, deep >= 0.85) and then the meta + decoy decoder on top
(cross-fitted, src/evaluation/decoy_experiment.py). One predefined configuration, no search. Paired per-S1 deltas.
"""

import argparse
import json
import os
import time

import pandas as pd  # before sklearn / xgboost (Windows: pyarrow after sklearn can crash)
import numpy as np

from src.evaluation.decoy_experiment import (K20, K50, OUT_DIR, all_row_decoy_features, cross_fit, decoy_rows,
                                             first_stage_oof)
from src.evaluation.hybrid_eval import HybridDev
from src.evaluation.structured_decoder import Table, coordinate_descent

XGB = dict(n_estimators=700, learning_rate=0.05, max_depth=8, subsample=0.8, colsample_bytree=0.8,
           min_child_weight=2, reg_lambda=1.0, tree_method="hist", n_jobs=-1, random_state=0)


def xgb_oof(ctx, D20, D50, log=print):
    from sklearn.model_selection import GroupKFold
    from xgboost import XGBClassifier

    from src.matching.feature_cache import feature_cache

    out, secs = {}, {}
    for name, d, D in (("k20", K20, D20), ("k50", K50, D50)):
        path = os.path.join(d, "oof_XGB_decoy.parquet")
        if os.path.exists(path):
            out[name] = pd.read_parquet(path)["prob"].to_numpy()
            continue
        t0 = time.time()
        meta, X = feature_cache(d)
        X = pd.concat([X, D], axis=1).to_numpy(dtype=np.float32)
        y, groups = meta["label"].to_numpy(), meta["s1_id"].to_numpy()
        p = np.zeros(len(X))
        for tr, va in GroupKFold(5).split(X, y, groups):
            p[va] = XGBClassifier(**XGB).fit(X[tr], y[tr]).predict_proba(X[va])[:, 1]
        meta.assign(prob=p).to_parquet(path)  # git-ignored
        out[name], secs[name] = p, round(time.time() - t0, 1)
        log(f"  XGBoost + decoy OOF {name}: {secs[name]}s")
    return out["k20"], out["k50"], secs


def hybrid_first_stage(T):
    tb = [coordinate_descent(lambda q: T.f05(T.decide(np.where(T.origin == 0, q[0], 0.85)))[T.fold != k].mean(),
                             [0.70])[0][0] for k in range(5)]
    return T.decide(np.where(T.origin == 0, np.array(tb)[T.row_fold], 0.85)), tb


def main(argv=None):
    import pickle

    from src.pipeline.train_meta import dev_meta_rows

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=f"{OUT_DIR}/gbdt_experiment.json")
    a = ap.parse_args(argv)
    t0 = time.time()
    ctx = HybridDev(native=False)
    D20, D50 = all_row_decoy_features(ctx)
    models = {}
    for role, d in (("base", K20), ("deep", K50)):
        with open(os.path.join(d, "matcher.pkl"), "rb") as fh:
            models[role] = pickle.load(fh)
    h20, h50 = first_stage_oof(ctx, D20, D50)  # M3 + decoy (cached)
    x20, x50, secs = xgb_oof(ctx, D20, D50)
    res, per = {}, {}
    for name, (q20, q50) in (("HGB M3 + decoy", (h20, h50)), ("XGBoost + decoy", (x20, x50))):
        T = Table(ctx, q20, q50)
        acc_b, tb = hybrid_first_stage(T)
        rows, M, _ = dev_meta_rows(T, models["base"], models["deep"])
        acc_m, tm, _ = cross_fit(T, pd.concat([M, decoy_rows(T, rows, D20, D50)], axis=1), rows)
        per[name] = (T.f05(acc_b), T.f05(acc_m))
        res[name] = {"first stage (Hybrid50, t20 per fold)": T.summary(per[name][0], acc=acc_b),
                     "+ meta + decoy (cross-fitted)": T.summary(per[name][1], acc=acc_m), "t20_per_fold": tb}
        print(name, json.dumps(res[name]), flush=True)
    T = Table(ctx, h20, h50)
    res["XGBoost vs HGB, first stage (paired)"] = T.summary(per["XGBoost + decoy"][0], ref=per["HGB M3 + decoy"][0])
    res["XGBoost vs HGB, + meta + decoy (paired)"] = T.summary(per["XGBoost + decoy"][1], ref=per["HGB M3 + decoy"][1])
    res["xgboost_params"] = XGB
    res["seconds"] = {"xgboost OOF": secs, "total": round(time.time() - t0, 1)}
    for k in ("XGBoost vs HGB, first stage (paired)", "XGBoost vs HGB, + meta + decoy (paired)"):
        print(k, json.dumps(res[k]))
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2, default=str)


if __name__ == "__main__":
    main()
