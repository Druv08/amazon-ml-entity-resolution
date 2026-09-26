"""Train the production meta-decoder bundle from cached development artifacts (no blocking, no featurization).

    python -m src.pipeline.train_meta        # -> output/candidates_p3/hybrid_meta.pkl (+ hybrid_meta.json)

Inputs (git-ignored): the K=20 / K=50 production matchers (M3) with their out-of-fold probabilities
(src/matching/train.py), the fingerprinted feature caches (src/matching/feature_cache.py) and the development sample.
  1. Hybrid50 rows with the first-stage OOF probabilities (base rows: K=20, deep rows: K=50).
  2. The meta matrix (src/pipeline/meta_decoder.py) of the rows >= META_FLOOR, checked identical to the matrix of the
     development experiment (structured_decoder.meta_matrix).
  3. 5-fold meta OOF over the K=20 matcher's S1 folds. Production thresholds (t_base, t_deep) are tuned on all
     development S1 (as train.py tunes the first-stage threshold); per-fold thresholds tuned on the other folds give
     a cross-fitted estimate. The production decision function is checked equal to the experiment's.
  4. The meta-model refit on all rows. Bundle = both first-stage matchers + meta-model + thresholds + versions +
     hybrid configuration + cache fingerprints, i.e. everything predict_hybrid needs.
"""

import argparse
import hashlib
import json
import os
import pickle
import time

import pandas as pd  # before sklearn (Windows: pyarrow after sklearn can crash)
import numpy as np

from src.evaluation.hybrid_eval import HybridDev
from src.evaluation.structured_decoder import GRID_META, Table, coordinate_descent
from src.evaluation.structured_decoder import meta_matrix as experiment_meta_matrix
from src.matching.feature_cache import FEATURE_VERSION, feature_cache, fingerprint
from src.matching.matcher import PRODUCTION_PARAMS, train
from src.pipeline.meta_decoder import (META_FLOOR, META_VERSION, context_features, decide, meta_matrix,
                                       pair_feature_names)

K20, K50 = "output/candidates_p3/k20", "output/candidates_p3/k50"


def sha256(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def dev_meta_rows(T, m20, m50, extra=(), ctx=None):
    """(rows >= floor, meta matrix) of the development Hybrid50 rows, from the fingerprinted feature caches.
    extra=("decoy",) appends the token-alignment features (src/matching/decoy_features.py, cached for all rows)."""
    _, X20 = feature_cache(K20)
    _, X50 = feature_cache(K50)
    if list(X20.columns) != list(m20["features"]) or list(X50.columns) != list(m50["features"]):
        raise ValueError("feature cache columns differ from the matchers' features")
    if list(m20["features"]) != list(m50["features"]):
        raise ValueError("K=20 and K=50 matchers use different features")
    rows = np.flatnonzero(T.prob >= META_FLOOR)
    base = T.origin[rows] == 0
    A = np.empty((len(rows), X20.shape[1]), dtype=np.float32)
    A[base] = X20.to_numpy()[T.src_row[rows][base]]
    A[~base] = X50.to_numpy()[T.src_row[rows][~base]]
    names = list(m20["features"])
    if "decoy" in extra:
        from src.evaluation.decoy_experiment import all_row_decoy_features, decoy_rows

        D20, D50 = all_row_decoy_features(ctx)
        D = decoy_rows(T, rows, D20, D50)
        A, names = np.hstack([A, D.to_numpy(dtype=np.float32)]), names + list(D.columns)
    context = context_features(T.s1, T.prob, T.origin)
    M = meta_matrix(T.prob[rows], T.origin[rows], T.rank[rows], context.iloc[rows], A, names)
    return rows, M, names


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="output/candidates_p3/hybrid_meta.pkl")
    ap.add_argument("--cap", type=int, default=50)
    ap.add_argument("--decoy", action="store_true", help="add the token-alignment features to the meta-model")
    a = ap.parse_args(argv)
    extra = ("decoy",) if a.decoy else ()
    t0 = time.time()
    models = {}
    for role, d in (("base", K20), ("deep", K50)):
        with open(os.path.join(d, "matcher.pkl"), "rb") as fh:
            models[role] = pickle.load(fh)
        if models[role].get("params") != PRODUCTION_PARAMS:
            raise SystemExit(f"{d}/matcher.pkl was not trained with PRODUCTION_PARAMS; re-run src.matching.train")
    ctx = HybridDev(native=False, cap=a.cap)
    p20 = pd.read_parquet(f"{K20}/oof.parquet", columns=["prob"])["prob"].to_numpy()
    p50 = pd.read_parquet(f"{K50}/oof.parquet", columns=["prob"])["prob"].to_numpy()
    T = Table(ctx, p20, p50)
    rows, M, pair_names = dev_meta_rows(T, models["base"], models["deep"], extra, ctx)
    M_exp = experiment_meta_matrix(T, rows, pair_features=True)  # extra (decoy) columns come from the shared cache
    same_matrix = bool(M.iloc[:, :M_exp.shape[1]].equals(M_exp))
    if not same_matrix:
        raise SystemExit("production meta matrix differs from the development experiment's matrix")
    t_cache = time.time() - t0
    y, fr = T.label[rows], T.row_fold[rows]
    oof = np.full(len(T.prob), -np.inf)
    t1 = time.time()
    for k in range(5):
        tr, va = fr != k, fr == k
        oof[rows[va]] = train(M[tr], y[tr]).predict_proba(M[va])[:, 1]
    win = T.winners(oof)

    def f05_on(s1_mask):
        return lambda q: T.f05(T.decide(np.where(T.origin == 0, q[0], q[1]), score=oof, win=win))[s1_mask].mean()

    t_base, t_deep = coordinate_descent(f05_on(np.ones(T.n_s1, dtype=bool)), [0.5, 0.5], grid=GRID_META)[0]
    acc_exp = T.decide(np.where(T.origin == 0, t_base, t_deep), score=oof, win=win)
    acc_prod = decide(T.cand, T.origin, oof, T.s1_rank, t_base, t_deep)
    same_decision = bool(np.array_equal(acc_exp, acc_prod))
    if not same_decision:
        raise SystemExit("production decision differs from the development experiment's decision")
    per_fold = {k: coordinate_descent(f05_on(T.fold != k), [0.5, 0.5], grid=GRID_META)[0] for k in range(5)}
    tm = np.array([per_fold[k] for k in range(5)])
    acc_cf = T.decide(np.where(T.origin == 0, tm[T.row_fold, 0], tm[T.row_fold, 1]), score=oof, win=win)
    ref_acc = T.decide(np.where(T.origin == 0, models["base"]["threshold"], 0.85))
    ref = T.f05(ref_acc)
    estimate = T.summary(T.f05(acc_cf), ref=ref, acc=acc_cf)
    final = train(M, y)
    t_fit = time.time() - t1
    bundle = {
        "meta_version": META_VERSION, "feature_version": FEATURE_VERSION,
        "base_matcher": models["base"], "deep_matcher": models["deep"],
        "meta_model": final, "meta_features": list(M.columns), "meta_floor": META_FLOOR,
        "pair_features": pair_names, "meta_extra": extra,
        "thresholds": {"base": float(t_base), "deep": float(t_deep)},
        "hybrid": {"base_top_k": models["base"]["top_k"], "deep_top_k": models["deep"]["top_k"], "cap": a.cap},
        "first_stage_sha256": {"base": sha256(f"{K20}/matcher.pkl"), "deep": sha256(f"{K50}/matcher.pkl")},
        "fingerprints": {"base": fingerprint(K20), "deep": fingerprint(K50)},
    }
    report = {"out": a.out, "meta_rows": int(len(rows)), "meta_features": int(M.shape[1]),
              "thresholds": bundle["thresholds"], "thresholds_per_fold": per_fold,
              "checks": {"meta_matrix_identical_to_experiment": same_matrix,
                         "decision_identical_to_experiment": same_decision},
              "first_stage_only_hybrid50": T.summary(ref, acc=ref_acc),
              "cross_fitted_estimate_vs_first_stage": estimate,
              "strictly_nested_estimate": "docs/p3_matching.md: 0.9574 (first stage retrained per fold)",
              "seconds": {"load_and_matrix_from_cache": round(t_cache, 1), "meta_cv_and_fit": round(t_fit, 1),
                          "total": round(time.time() - t0, 1)}}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "wb") as fh:
        pickle.dump(bundle, fh, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.splitext(a.out)[0] + ".json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
