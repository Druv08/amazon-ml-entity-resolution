"""Checkpoint 14: does averaging a few different HGB matchers beat the single production model?

    python -m src.matching.ensemble_experiment oof          # OOF probabilities of M0-M3 at K=20 and K=50
    python -m src.matching.ensemble_experiment eval         # K=20 base and Hybrid50 for every model / ensemble

A small PREDEFINED set (no search), so the development sample is not overfitted:
    M0 production    lr 0.05, 31 leaves, max_iter 500
    M1 conservative  lr 0.03, 31 leaves, max_iter 800 (same number of effective steps)
    M2 simpler       lr 0.05, 15 leaves
    M3 richer        lr 0.05, 63 leaves
Same A+C+E features, same GroupKFold(5) by S1, same development sample, same Hybrid50 architecture (deep >= 0.85).
Ensembles: arithmetic mean, median and mean log-odds of the four probabilities. The K=20 threshold is re-tuned for
every model / ensemble on development (as train.py). Feature matrices and OOF files go under output/ (git-ignored).
"""

import argparse
import json
import os
import time

import pandas as pd  # before sklearn (Windows: pyarrow after sklearn can crash)
import numpy as np
from sklearn.model_selection import GroupKFold

from src.matching.feature_cache import feature_cache  # noqa: F401  (fingerprinted cache)
from src.matching.matcher import train
from src.matching.sample_candidates import OUT

MODELS = {"M0": {}, "M1": dict(learning_rate=0.03, max_iter=800), "M2": dict(max_leaf_nodes=15),
          "M3": dict(max_leaf_nodes=63)}
GROUPS = ("A", "C", "E")


def make_oof(cands, names, log=print):
    meta, X = feature_cache(cands)
    y, groups = meta["label"].to_numpy(), meta["s1_id"].to_numpy()
    folds = list(GroupKFold(5).split(X, y, groups))
    timing = {}
    for name in names:
        out = os.path.join(cands, f"oof_{name}.parquet")
        if os.path.exists(out):
            continue
        oof, fit_s, pred_s, iters = np.zeros(len(X)), 0.0, 0.0, []
        for tr, va in folds:
            t0 = time.time()
            m = train(X.iloc[tr], y[tr], **MODELS[name])
            t1 = time.time()
            oof[va] = m.predict_proba(X.iloc[va])[:, 1]
            fit_s, pred_s = fit_s + t1 - t0, pred_s + time.time() - t1
            iters.append(int(m.n_iter_))
        meta[["s1_id", "cand_id", "rank", "label"]].assign(prob=oof).to_parquet(out)
        timing[name] = {"fit_seconds": round(fit_s, 1), "predict_seconds_per_1m_rows": round(pred_s / len(X) * 1e6, 2),
                        "n_iter": iters}
        log(f"  {cands} {name}: {timing[name]}")
        with open(os.path.join(cands, f"oof_{name}.json"), "w", encoding="utf-8") as fh:
            json.dump(timing[name], fh)
    return timing


def load_probs(cands, names):
    return {n: pd.read_parquet(os.path.join(cands, f"oof_{n}.parquet"), columns=["prob"])["prob"].to_numpy()
            for n in names}


def ensembles(p):
    """{name: probabilities} of the single models plus mean / median / mean log-odds of all of them."""
    names = list(MODELS)
    P = np.column_stack([p[n] for n in names])
    q = np.clip(P, 1e-7, 1 - 1e-7)
    z = np.log(q / (1 - q)).mean(axis=1)
    return {**{n: p[n] for n in names}, "mean": P.mean(axis=1), "median": np.median(P, axis=1),
            "logit_mean": 1 / (1 + np.exp(-z))}


def evaluate(out="output/error_analysis/ensemble.json", log=print):
    from src.evaluation.hybrid_eval import HybridDev

    ctx = HybridDev()
    p20, p50 = ensembles(load_probs(f"{OUT}/k20", MODELS)), ensembles(load_probs(f"{OUT}/k50", MODELS))
    stored = {k: pd.read_parquet(f"{OUT}/{k}/oof.parquet", columns=["prob"])["prob"].to_numpy() for k in ("k20", "k50")}
    same = {k: bool(np.array_equal(stored[k], (p20 if k == "k20" else p50)["M0"])) for k in stored}
    log(f"regenerated M0 OOF identical to the stored production OOF: {same}")
    ref = ctx.run(stored["k20"], stored["k50"], t20=0.7)  # the adopted Hybrid50 (0.9525)
    ref_per = ctx.per_s1(ref["pred"])
    timing = {}
    for k in ("k20", "k50"):
        for n in MODELS:
            with open(f"{OUT}/{k}/oof_{n}.json", encoding="utf-8") as fh:
                timing[f"{k}_{n}"] = json.load(fh)
    rows = []
    for name in p20:
        r = ctx.run(p20[name], p50[name])
        base = ctx.metrics(r["base_pred"], ref=ref_per)
        hyb = ctx.metrics(r["pred"], ref=ref_per)
        cost = sum(timing[f"k20_{n}"]["predict_seconds_per_1m_rows"] for n in (MODELS if name not in MODELS else [name]))
        rows.append({"model": name, "t20": r["t20"], "k20": base, "hybrid50": hyb,
                     "added_deep": len(r["added"]), "predict_s_per_1m": round(cost, 2)})
        log(f"{name:10s} t20={r['t20']:.2f} K20 {base['f05']:.4f} | hybrid {hyb['f05']:.4f} "
            f"(d {hyb['delta']:+.4f} +- {hyb['delta_se']:.4f}, CI {hyb['delta_ci95']}) India {hyb['India']:.4f} "
            f"US {hyb['US']:.4f} single {hyb['singletons_empty']:.3f} FP {hyb['fp']} FN {hyb['fn']} "
            f"r1-3 FP {hyb['fp_rank1_3']} native FN {hyb['fn_native_script']} predict {cost:.1f}s/1M", flush=True)
    report = {"m0_regenerated_identical": same, "reference_hybrid50": ctx.metrics(ref["pred"]), "timing": timing,
              "results": rows}
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["oof", "eval"])
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    a = ap.parse_args(argv)
    if a.stage == "oof":
        for k in ("k20", "k50"):
            make_oof(f"{OUT}/{k}", a.models, log=lambda m: print(m, flush=True))
    else:
        evaluate()


if __name__ == "__main__":
    main()
