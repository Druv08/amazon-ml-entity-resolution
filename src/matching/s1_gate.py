"""S1-level no-match gate: should an S1 output zero matches although some candidate clears the threshold?

python -m src.matching.s1_gate --cands output/candidates_p3/k20 [--oof path/to/oof.parquet]

A singleton scores 1.0 only with an empty prediction and 0.0 with any match. The gate uses S1-level signals built
from the S1's own candidates only: pair probabilities and pair features. Those are shard-local at inference, so
they need no global state. The gate is trained out of fold, with the same S1 fold assignment as the pair model.
It is applied AFTER exclusivity and the threshold: a suppressed S1 outputs nothing, and its candidates are not
reassigned. Its decision threshold is tuned on the out-of-fold gate probabilities, like the pair threshold.
Development sample only; never the final holdout.
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd  # before sklearn: loading pyarrow (via pandas) after sklearn crashes on Windows
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

from src.matching.hard_negatives import entity_f05
from src.matching.matcher import best_threshold, exclusive, macro_f05, to_matches


def s1_features(pairs, prob, X):
    """One row per S1: the shape of its candidate probability distribution and its best evidence."""
    d = pd.DataFrame({"s1_id": pairs["s1_id"].to_numpy(), "p": np.asarray(prob, dtype=float),
                      "rank": pairs["rank"].to_numpy(), "block_score": pairs["block_score"].to_numpy(),
                      "name_score": pairs["name_score"].to_numpy(), "name_tsort": X["name_tsort"].to_numpy(),
                      "addr_tsort": X["addr_tsort"].to_numpy(), "core_tsort": X["core_tsort"].to_numpy(),
                      "num_conflict": X["num_conflict"].to_numpy(), "first_num_eq": X["first_num_eq"].to_numpy(),
                      "addr_missing": X["addr_missing"].to_numpy()})
    d = d.sort_values(["s1_id", "p"], ascending=[True, False], kind="mergesort")
    g = d.groupby("s1_id", sort=False)
    top = g.head(1).set_index("s1_id")
    second = g.nth(1).set_index("s1_id")["p"]
    f = pd.DataFrame(index=top.index)
    f["p1"] = top["p"]
    f["p2"] = second.reindex(f.index).fillna(0.0)
    f["gap12"] = f["p1"] - f["p2"]
    for t in (0.3, 0.5, 0.7, 0.9):
        f[f"n_above_{t}"] = g["p"].apply(lambda s, t=t: int((s >= t).sum())).reindex(f.index)
    f["p_sum"] = g["p"].sum().reindex(f.index)
    f["n_cands"] = g.size().reindex(f.index)
    for c in ("rank", "block_score", "name_score", "name_tsort", "addr_tsort", "core_tsort", "num_conflict",
              "first_num_eq", "addr_missing"):
        f[f"top_{c}"] = top[c]  # evidence of the most probable candidate
    f["best_block_score"] = g["block_score"].max().reindex(f.index)
    f["best_name_tsort"] = g["name_tsort"].max().reindex(f.index)
    f["best_addr_tsort"] = g["addr_tsort"].max().reindex(f.index)
    return f.astype(np.float32)


def gate_oof(F, singleton, folds_by_s1, seed=0):
    """Out-of-fold P(singleton) per S1 row of F. folds_by_s1: fold id per row (the pair model's S1 folds)."""
    p = np.zeros(len(F))
    for k in np.unique(folds_by_s1):
        tr, va = folds_by_s1 != k, folds_by_s1 == k
        m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, early_stopping=True,
                                           random_state=seed).fit(F[tr], singleton[tr])
        p[va] = m.predict_proba(F[va])[:, 1]
    return p


def apply_gate(pred, suppress):
    """Empty the predictions of the suppressed S1."""
    return {s: (set() if s in suppress else v) for s, v in pred.items()}


def gate_metrics(pred, truth, s1, suppressed):
    rnd = s1[s1.city == ""]
    ids = rnd.entity_id.tolist()
    per = np.array([entity_f05(pred[s], truth[s]) for s in ids])
    single = np.array([not truth[s] for s in ids])
    sup = np.array([s in suppressed for s in ids])
    country = rnd.country.to_numpy()
    return {"f05": float(per.mean()), "India": float(per[country == "India"].mean()),
            "US": float(per[country == "US"].mean()),
            "singleton_empty": float(np.mean([not pred[s] for s, x in zip(ids, single) if x])),
            "matched_f05": float(per[~single].mean()), "suppressed": int(sup.sum()),
            "suppressed_true_singletons": int((sup & single).sum()),
            "suppressed_matched": int((sup & ~single).sum())}, per


if __name__ == "__main__":
    from src.evaluation.matcher_errors import load_dev
    from src.matching.hard_negatives import paired_delta

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cands", default="output/candidates_p3/k20")
    ap.add_argument("--oof", default=None, help="pair OOF parquet (s1_id, cand_id, prob); default <cands>/oof.parquet")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--out", default="output/error_analysis/s1_gate.json")
    a = ap.parse_args()
    t0 = time.time()
    d = load_dev(a.cands)
    pairs, X, truth, s1 = d["pairs"], d["X"], d["truth"], d["s1"]
    oof = pd.read_parquet(a.oof or os.path.join(a.cands, "oof.parquet"))
    prob = pairs[["s1_id", "cand_id"]].merge(oof[["s1_id", "cand_id", "prob"]], on=["s1_id", "cand_id"],
                                             how="left")["prob"].to_numpy()
    assert not np.isnan(prob).any()
    ex = exclusive(pairs, prob)
    t, _ = best_threshold(pairs, ex, truth)
    base_pred = to_matches(pairs, ex, t, truth)
    base, base_per = gate_metrics(base_pred, truth, s1, set())
    print(f"pair model: threshold {t}, F0.5 {base['f05']:.4f}, singleton empty {base['singleton_empty']:.3f}")

    F = s1_features(pairs, prob, X)
    singleton = np.array([not truth[s] for s in F.index], dtype=int)
    fold_of_pair = np.zeros(len(pairs), dtype=int)
    for k, (_, va) in enumerate(GroupKFold(5).split(pairs, groups=pairs["s1_id"])):
        fold_of_pair[va] = k
    fold = pd.Series(fold_of_pair, index=pairs["s1_id"].to_numpy()).groupby(level=0).first().reindex(F.index).to_numpy()
    has_match = np.array([bool(base_pred[s]) for s in F.index])
    results = [{"config": "pair model only", "threshold": t, **base}]
    for seed in a.seeds:
        p = gate_oof(F.to_numpy(), singleton, fold, seed)
        # gate threshold tuned on the OOF gate probabilities over all sampled S1, like the pair threshold
        taus = np.round(np.arange(0.30, 0.99, 0.02), 2)
        scores = [macro_f05(apply_gate(base_pred, set(F.index[(p >= tau) & has_match])), truth) for tau in taus]
        tau = float(taus[int(np.argmax(scores))])
        sup = set(F.index[(p >= tau) & has_match])
        m, per = gate_metrics(apply_gate(base_pred, sup), truth, s1, sup)
        m.update({"config": f"pair model + S1 gate (seed {seed})", "gate_threshold": tau,
                  "delta": paired_delta(per, base_per, s1[s1.city == ""].country.to_numpy())})
        results.append(m)
        print(f"gate seed {seed}: tau {tau} F0.5 {m['f05']:.4f} (d {m['delta']['all'][0]:+.4f} +- "
              f"{m['delta']['all'][1]:.4f}) singleton empty {m['singleton_empty']:.3f} matched {m['matched_f05']:.4f} "
              f"India {m['India']:.4f} US {m['US']:.4f} suppressed {m['suppressed']} "
              f"(true singletons {m['suppressed_true_singletons']}, matched {m['suppressed_matched']})", flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"wrote {a.out} ({time.time() - t0:.0f}s)")
