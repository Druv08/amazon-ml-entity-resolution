"""Deep recovery: keep the trusted K=20 decisions and ADD only confident matches from the hybrid deep pool.

    python -m src.pipeline.deep_recovery                 # threshold sweep on the P3 development sample

Base    = improved K=20 matcher: exclusivity + its threshold on the K=20 candidates (never changed here).
Deep    = hybrid candidates that are not in the exact K=20 list (src/pipeline/hybrid.py), scored with the K=50
          matcher's out-of-fold probability.
Merge   = base matches  +  deep candidates accepted by a rule, subject to global exclusivity:
          * a deep candidate already assigned by the base to another S1 is never added (base has priority);
          * several S1 claiming the same deep candidate: highest probability wins, then the smallest s1_id;
          * base matches are never removed.
Metrics on the hash-random development S1 (the headline set). Never the final holdout.
"""

import argparse
import json
import os
import pickle

import numpy as np
import pandas as pd

from src.matching.hard_negatives import entity_f05

DEEP_THRESHOLD = 0.85  # adopted rule "deep prob >= 0.85" (docs/p3_matching.md "Deep recovery"); plateau 0.66-0.86
from src.matching.matcher import exclusive, to_matches


def base_predictions(k20_oof, threshold, truth):
    """-> (pred {s1: set}, owner {cand: s1}) of the K=20 base."""
    ex = exclusive(k20_oof, k20_oof["prob"].to_numpy())
    pred = to_matches(k20_oof, ex, threshold, truth)
    owner = {c: s for s, cs in pred.items() for c in cs}
    return pred, owner


def deep_pool(hybrid, k50_oof, truth):
    """Deep rows of the hybrid with the K=50 OOF probability and the label."""
    deep = hybrid[hybrid["origin"] == "deep"][["s1_id", "cand_id", "deep_rank"]]
    deep = deep.merge(k50_oof[["s1_id", "cand_id", "prob"]], on=["s1_id", "cand_id"], how="left")
    if deep["prob"].isna().any():
        raise ValueError("deep candidate without a K=50 OOF probability")
    deep["label"] = [int(c in truth.get(s, ())) for s, c in zip(deep["s1_id"], deep["cand_id"])]
    return deep.reset_index(drop=True)


def merge_deep(base_pred, owner, deep, accept):
    """Base matches + accepted deep rows (boolean mask over ``deep``) under global exclusivity."""
    cand = deep[accept.astype(bool)]
    cand = cand[~cand["cand_id"].isin(owner.keys())]  # the base already gave it to some S1
    cand = cand.sort_values(["cand_id", "prob", "s1_id"], ascending=[True, False, True], kind="mergesort")
    cand = cand.drop_duplicates("cand_id")  # one S1 per deep candidate
    pred = {s: set(v) for s, v in base_pred.items()}
    for s, c in zip(cand["s1_id"], cand["cand_id"]):
        pred[s].add(c)
    return pred, cand


def evaluate(pred, base_pred, truth, s1, added):
    rnd = s1[s1.city == ""]
    ids, country = rnd.entity_id.tolist(), rnd.country.to_numpy()
    per = np.array([entity_f05(pred[s], truth[s]) for s in ids])
    base = np.array([entity_f05(base_pred[s], truth[s]) for s in ids])
    single = np.array([not truth[s] for s in ids])
    in_rnd = added["s1_id"].isin(set(ids)).to_numpy()
    lab = added["label"].to_numpy() == 1
    d = per - base
    return {"f05": round(float(per.mean()), 5), "delta": round(float(d.mean()), 5),
            "delta_se": round(float(d.std(ddof=1) / np.sqrt(len(d))), 5),
            "India": round(float(per[country == "India"].mean()), 4), "US": round(float(per[country == "US"].mean()), 4),
            "singletons_empty": round(float(np.mean([not pred[s] for s, x in zip(ids, single) if x])), 4),
            "added": int(in_rnd.sum()), "added_tp": int((in_rnd & lab).sum()), "added_fp": int((in_rnd & ~lab).sum()),
            "s1_improved": int((d > 1e-12).sum()), "s1_harmed": int((d < -1e-12).sum())}


def deep_features(deep, records, base_pred, k20_oof):
    """Pair-local evidence for the deep rows (name / address similarity, number agreement, transliteration) and the
    S1's K=20 context. Everything is computable identically on test."""
    from rapidfuzz import fuzz, process

    from src.blocking.normalize import has_nonlatin
    from src.matching.extra_features import address_number_features, translit_features
    from src.matching.matcher import _norm

    f = deep.copy()
    rec = records.set_index("entity_id")
    for col, key in (("business_name", "name"), ("business_address", "addr")):
        va = rec[col].reindex(f["s1_id"]).fillna("").tolist()
        vb = rec[col].reindex(f["cand_id"]).fillna("").tolist()
        cache = {v: " ".join(_norm(v)) for v in set(va) | set(vb)}
        f[f"{key}_tsort"] = process.cpdist([cache[v] for v in va], [cache[v] for v in vb],
                                           scorer=fuzz.token_sort_ratio, workers=-1) / 100
    f["cand_addr_missing"] = rec["business_address"].reindex(f["cand_id"]).isna().to_numpy()
    f["cand_nonlatin"] = [has_nonlatin(v) for v in rec["business_name"].reindex(f["cand_id"]).fillna("")]
    f["country"] = rec["country"].reindex(f["s1_id"]).to_numpy()
    f = pd.concat([f, address_number_features(f, records), translit_features(f, records)], axis=1)
    best20 = k20_oof.groupby("s1_id")["prob"].max()
    f["base_n_matches"] = [len(base_pred.get(s, ())) for s in f["s1_id"]]
    f["base_best_prob"] = f["s1_id"].map(best20).fillna(0.0).to_numpy()
    g = f.groupby("s1_id")["prob"]
    f["deep_second_prob"] = g.transform(lambda x: x.nlargest(2).iloc[-1] if len(x) > 1 else 0.0)
    f["deep_is_best"] = (f["prob"] == g.transform("max")).to_numpy()
    return f


def rule_masks(f, t):
    """Small evidence-based deep acceptance rules (docs/p3_matching.md)."""
    p = f["prob"].to_numpy()
    both_nums = f["num_best_ratio"].to_numpy() >= 0
    strong_conflict = both_nums & (f["num_best_ratio"].to_numpy() < 0.5)
    name_ev = (f["name_tsort"].to_numpy() >= 0.8) | (f["ph_jacc"].to_numpy() >= 0.5)
    addr_ok = f["cand_addr_missing"].to_numpy() | (f["addr_word_jacc"].to_numpy() >= 0.3)
    margin = f["deep_is_best"].to_numpy() | (p - f["deep_second_prob"].to_numpy() >= 0.2)
    return {
        "A prob": p >= t,
        "B prob, no strong number conflict": (p >= t) & ~strong_conflict,
        "C prob, strong name/transliteration evidence": (p >= t) & name_ev,
        "D prob, beats the S1's other deep candidates": (p >= t) & margin,
        "E prob, name evidence and acceptable address": (p >= t) & name_ev & addr_ok,
    }


def load_inputs(k20="output/candidates_p3/k20", k50="output/candidates_p3/k50", cap=50):
    from src.evaluation.k_compare import dev_truth
    from src.matching.sample_candidates import OUT
    from src.pipeline.hybrid import hybrid_candidates

    s1 = pd.read_parquet(f"{OUT}/sample_s1.parquet")
    truth = dev_truth(s1)
    k20_oof, k50_oof = pd.read_parquet(f"{k20}/oof.parquet"), pd.read_parquet(f"{k50}/oof.parquet")
    with open(f"{k20}/matcher.pkl", "rb") as fh:
        t20 = pickle.load(fh)["threshold"]
    hybrid = hybrid_candidates(k20_oof, k50_oof, cap=cap)
    base_pred, owner = base_predictions(k20_oof, t20, truth)
    return s1, truth, k20_oof, k50_oof, t20, hybrid, base_pred, owner, deep_pool(hybrid, k50_oof, truth)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--thresholds", nargs="+", type=float,
                    default=[0.70, 0.75, 0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.98, 0.99])
    ap.add_argument("--out", default="output/error_analysis/deep_threshold_sweep.json")
    a = ap.parse_args(argv)
    s1, truth, _, _, t20, _, base_pred, owner, deep = load_inputs()
    rnd = set(s1.loc[s1.city == "", "entity_id"])
    dr = deep[deep["s1_id"].isin(rnd)]
    print(f"K20 base threshold {t20}; deep pool: {len(deep)} rows ({len(dr)} random-S1), true {int(deep.label.sum())} "
          f"({int(dr.label.sum())} random-S1), positive rate {deep.label.mean():.4%}")
    q = [0.5, 0.9, 0.99]
    for lab in (1, 0):
        pr = deep.loc[deep.label == lab, "prob"]
        print(f"  label {lab}: prob quantiles {dict(zip(q, np.round(pr.quantile(q).to_numpy(), 4)))}, "
              f">=0.9: {int((pr >= 0.9).sum())}, >=0.97: {int((pr >= 0.97).sum())}")
    results = [{"rule": "K20 base", **evaluate(base_pred, base_pred, truth, s1, deep.iloc[:0])}]
    for t in a.thresholds:
        pred, added = merge_deep(base_pred, owner, deep, deep["prob"].to_numpy() >= t)
        r = {"rule": f"deep prob >= {t}", "threshold": t, **evaluate(pred, base_pred, truth, s1, added)}
        results.append(r)
        print(f"  T={t:<5} F0.5 {r['f05']:.4f} (d {r['delta']:+.4f} +- {r['delta_se']:.4f}) India {r['India']:.4f} "
              f"US {r['US']:.4f} single {r['singletons_empty']:.3f} added {r['added']} (TP {r['added_tp']}, "
              f"FP {r['added_fp']}) S1 +{r['s1_improved']}/-{r['s1_harmed']}", flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)


if __name__ == "__main__":
    main()
