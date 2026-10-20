"""Controlled experiment: up-weight hard negatives (label 0, blocker rank <= 3) when training the matcher.

python -m src.matching.hard_negatives --cands output/candidates_p3/k20 --weights 1 1.5 2 3

Same data, features and GroupKFold folds as src.matching.train for every configuration; only the training
sample_weight changes. Each configuration gets its own threshold, tuned on its out-of-fold probabilities exactly as
train.report() does. Deltas vs the baseline (weight 1, seed 0) are paired over the same random development S1,
with their standard error, and a second-seed baseline shows how much training randomness alone moves the numbers.
Development sample only; never the final holdout (docs/final_holdout.md). Writes a JSON summary under output/.
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd  # noqa: F401  before sklearn: loading pyarrow (via pandas) after sklearn crashes on Windows
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from src.matching.matcher import best_threshold, exclusive, to_matches, train
from src.matching.sample_candidates import OUT
from src.matching.train import featurize, load


def entity_f05(p, t):
    """F0.5 of one S1, as in matcher.macro_f05."""
    if not t:
        return 1.0 if not p else 0.0
    tp = len(p & t)
    if tp == 0:
        return 0.0
    pr, rc = tp / len(p), tp / len(t)
    return 1.25 * pr * rc / (0.25 * pr + rc)


def evaluate(pairs, prob, truth, s1, hard):
    """Same decision rule and metric definitions as train.report(), plus counts and per-S1 scores."""
    y = pairs["label"].to_numpy()
    ex = exclusive(pairs, prob)
    t, _ = best_threshold(pairs, ex, truth)
    pred = to_matches(pairs, ex, t, truth)
    rnd = s1[s1.city == ""]
    per = np.array([entity_f05(pred[s], truth[s]) for s in rnd.entity_id])
    acc, neg = ex >= t, y == 0
    fp = acc & neg
    single = np.array([not truth[s] for s in rnd.entity_id])
    out = {"threshold": t, "f05": float(per.mean()), "auc": float(roc_auc_score(y, prob)),
           "singletons_empty": float(np.mean([not pred[s] for s in rnd.entity_id[single]])),
           "fp_total": int(fp.sum())}
    for c in ("India", "US"):
        out[c] = float(per[(rnd.country == c).to_numpy()].mean())
    for name, m in hard.items():
        out[f"fp_{name}"] = int((fp & m).sum())
    return out, per, rnd.country.to_numpy()


def paired_delta(per, base, countries):
    d = per - base
    res = {"all": (float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d))))}
    for c in ("India", "US"):
        dc = d[countries == c]
        res[c] = (float(dc.mean()), float(dc.std(ddof=1) / np.sqrt(len(dc))))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cands", default=f"{OUT}/k20")
    ap.add_argument("--weights", nargs="+", type=float, default=[1.0, 1.5, 2.0, 3.0])
    ap.add_argument("--hard-rank", type=int, default=3, help="negatives with blocker rank <= this are up-weighted")
    ap.add_argument("--seed-check", nargs="*", type=float, default=[1.0],
                    help="weights re-run with random_state=1 (training-noise reference)")
    ap.add_argument("--out", default="output/experiments/hard_negatives_k20.json")
    a = ap.parse_args()
    t0 = time.time()

    pairs, records, truth, s1, claimed, top_k = load(a.cands)
    X, y, _, hard = featurize(pairs, records, claimed)
    del claimed
    folds = list(GroupKFold(5).split(X, y, pairs["s1_id"]))  # identical folds for every configuration
    is_hard = (y == 0) & (pairs["rank"].to_numpy() <= a.hard_rank)
    print(f"features {time.time() - t0:.0f}s: {X.shape}; hard negatives (rank<={a.hard_rank}): {is_hard.sum()}",
          flush=True)

    runs = [(w, 0) for w in a.weights] + [(w, 1) for w in a.seed_check]
    results, base = [], None
    for w, seed in runs:
        t1 = time.time()
        sw = np.where(is_hard, w, 1.0)
        oof = np.zeros(len(pairs))
        for tr, va in folds:
            oof[va] = train(X.iloc[tr], y[tr], sample_weight=sw[tr], random_state=seed).predict_proba(X.iloc[va])[:, 1]
        r, per, countries = evaluate(pairs, oof, truth, s1, hard)
        r.update({"weight": w, "seed": seed, "seconds": round(time.time() - t1, 1)})
        if base is None:
            base = per
        r["delta_vs_baseline"] = paired_delta(per, base, countries)
        results.append(r)
        dl = r["delta_vs_baseline"]["all"]
        print(f"w={w:<4} seed={seed} F0.5 {r['f05']:.4f} (d {dl[0]:+.4f} +- {dl[1]:.4f}) India {r['India']:.4f} "
              f"US {r['US']:.4f} single {r['singletons_empty']:.3f} AUC {r['auc']:.4f} FP {r['fp_total']} "
              f"rank1-3 FP {r['fp_blocker rank 1-3']} t {r['threshold']} [{r['seconds']}s]", flush=True)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump({"cands": a.cands, "top_k": top_k, "hard_rank": a.hard_rank, "hard_negatives": int(is_hard.sum()),
                   "runs": results}, fh, indent=2)
    print(f"wrote {a.out} ({time.time() - t0:.0f}s)")
