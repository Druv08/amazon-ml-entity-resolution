"""Controlled feature-group experiments on the P3 development sample (never the final holdout).

python -m src.matching.feature_experiments --cands output/candidates_p3/k20 --variants base A C D E
python -m src.matching.feature_experiments --variants base A+E --seeds 0 1      # robustness of a promising variant

A variant is "base" (the production features) or "+"-joined groups from src.matching.extra_features added to them.
Every variant uses the same development S1, the same GroupKFold folds, an unweighted HistGradientBoostingClassifier
(so "base" reproduces src.matching.train exactly) and its own threshold tuned on its OOF probabilities.
Deltas are paired per random development S1 against "base" with the same seed.
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd  # noqa: F401  before sklearn: loading pyarrow (via pandas) after sklearn crashes on Windows
from sklearn.model_selection import GroupKFold

from src.evaluation.matcher_errors import OUT_DIR, load_dev
from src.matching.extra_features import extra_features, load_name_stats
from src.matching.hard_negatives import evaluate, paired_delta
from src.matching.matcher import train


def group_features(d, group, tag, cache_dir=OUT_DIR):
    path = os.path.join(cache_dir, f"feat_{group}_{tag}.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)
    ns = load_name_stats("train", cache_dir=cache_dir) if group == "E" else None
    f = extra_features([group], d["pairs"], d["records"], ns)
    f.to_parquet(path)
    return f


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cands", default="output/candidates_p3/k20")
    ap.add_argument("--variants", nargs="+", default=["base", "A", "C", "D", "E"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    t0 = time.time()
    tag = os.path.basename(os.path.normpath(a.cands))
    d = load_dev(a.cands)
    pairs, X = d["pairs"], d["X"]
    y = pairs["label"].to_numpy()
    folds = list(GroupKFold(5).split(X, y, pairs["s1_id"]))
    claimed = d["claimed"]
    hard = {"blocker rank 1-3": (pairs["rank"] <= 3).to_numpy(),
            "lookalike name (tsort>=0.9)": (X["name_tsort"] >= 0.9).to_numpy(),
            "claimed by another S1": pairs["cand_id"].isin(claimed).to_numpy()}
    groups = sorted({g for v in a.variants if v != "base" for g in v.split("+")})
    feats = {}
    for g in groups:
        t = time.time()
        feats[g] = group_features(d, g, tag)
        print(f"group {g}: {feats[g].shape[1]} features in {time.time() - t:.0f}s", flush=True)

    results, base = [], {}
    for seed in a.seeds:
        for v in a.variants:
            t = time.time()
            Xv = X if v == "base" else pd.concat([X] + [feats[g] for g in v.split("+")], axis=1)
            oof = np.zeros(len(pairs))
            for tr, va in folds:
                oof[va] = train(Xv.iloc[tr], y[tr], random_state=seed).predict_proba(Xv.iloc[va])[:, 1]
            r, per, countries = evaluate(pairs, oof, d["truth"], d["s1"], hard)
            if v == "base":
                base[seed] = per
            r.update({"variant": v, "seed": seed, "features": Xv.shape[1], "seconds": round(time.time() - t, 1)})
            r["delta_vs_base"] = paired_delta(per, base[seed], countries) if seed in base else None
            results.append(r)
            dl = r["delta_vs_base"]["all"] if r["delta_vs_base"] else (float("nan"), float("nan"))
            print(f"{v:10s} seed={seed} n={Xv.shape[1]:3d} F0.5 {r['f05']:.4f} (d {dl[0]:+.4f} +- {dl[1]:.4f}) "
                  f"India {r['India']:.4f} US {r['US']:.4f} single {r['singletons_empty']:.3f} AUC {r['auc']:.4f} "
                  f"FP {r['fp_total']} r1-3 {r['fp_blocker rank 1-3']} look {r['fp_lookalike name (tsort>=0.9)']} "
                  f"claimed {r['fp_claimed by another S1']} t {r['threshold']} [{r['seconds']}s]", flush=True)
    out = a.out or os.path.join(OUT_DIR, f"features_{tag}_{'_'.join(a.variants)}_s{'-'.join(map(str, a.seeds))}.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"wrote {out} ({time.time() - t0:.0f}s)")
