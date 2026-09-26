"""Compare matcher runs (e.g. different K) from their saved OOF predictions, on the P3 development sample.

    python -m src.evaluation.k_compare base20=output/candidates_p3/k20/oof_baseline47.parquet:0.65 \\
        k20=output/candidates_p3/k20 k50=output/candidates_p3/k50

Each argument is name=<cands dir> (uses <dir>/oof.parquet and the threshold in <dir>/matcher.pkl) or
name=<oof.parquet>:<threshold>. Applies the production rule (exclusivity + threshold) and reports pair recall,
the candidate ceiling, macro-F0.5 overall / India / US / matched S1, singleton accuracy, AUC and FP counts, on the
hash-random development S1 (the headline set of src.matching.train). Never the final holdout.
"""

import argparse
import json
import os
import pickle

import numpy as np
import pandas as pd  # noqa: F401  before sklearn: loading pyarrow (via pandas) after sklearn crashes on Windows
from sklearn.metrics import roc_auc_score

from src.matching.hard_negatives import entity_f05
from src.matching.matcher import exclusive, to_matches
from src.matching.sample_candidates import OUT


def dev_truth(s1, gt_path="data/raw/train/train_ground_truth.tsv"):
    gt = pd.read_csv(gt_path, sep="\t", dtype=str, keep_default_na=False)
    gt = gt[gt.source1_entity_id.isin(s1.entity_id)]
    return {s: set(filter(None, m.split(","))) for s, m in zip(gt.source1_entity_id, gt.matched_entity_ids)}


def run_metrics(oof, threshold, truth, s1):
    """oof: DataFrame(s1_id, cand_id, rank, label, prob) over ALL sampled S1 (exclusivity needs all of them)."""
    ex = exclusive(oof, oof["prob"].to_numpy())
    pred = to_matches(oof, ex, threshold, truth)
    perfect = to_matches(oof, oof["label"].to_numpy().astype(float), 0.5, truth)
    rnd = s1[s1.city == ""]
    ids, country = rnd.entity_id.tolist(), rnd.country.to_numpy()
    per = np.array([entity_f05(pred[s], truth[s]) for s in ids])
    ceil = np.array([entity_f05(perfect[s], truth[s]) for s in ids])
    single = np.array([not truth[s] for s in ids])
    acc, neg = ex >= threshold, oof["label"].to_numpy() == 0
    n_true = sum(len(truth[s]) for s in ids)
    in_rnd = oof["s1_id"].isin(set(ids)).to_numpy()
    return {
        "threshold": threshold,
        "pairs": int(in_rnd.sum()), "cands_per_s1": round(float(in_rnd.sum() / len(ids)), 2),
        "pair_recall": round(float(oof.loc[in_rnd, "label"].sum() / n_true), 4),
        "ceiling": round(float(ceil.mean()), 4),
        "f05": round(float(per.mean()), 4),
        "India": round(float(per[country == "India"].mean()), 4), "US": round(float(per[country == "US"].mean()), 4),
        "matched_f05": round(float(per[~single].mean()), 4),
        "singletons_empty": round(float(np.mean([not pred[s] for s, x in zip(ids, single) if x])), 4),
        "auc": round(float(roc_auc_score(oof["label"], oof["prob"])), 4),
        "fp": int((acc & neg).sum()), "fp_rank_1_3": int((acc & neg & (oof["rank"].to_numpy() <= 3)).sum()),
    }


def parse(spec):
    name, src = spec.split("=", 1)
    if src.endswith(".parquet") or ".parquet:" in src:
        path, t = src.rsplit(":", 1)
        return name, pd.read_parquet(path), float(t)
    with open(os.path.join(src, "matcher.pkl"), "rb") as fh:
        t = pickle.load(fh)["threshold"]
    return name, pd.read_parquet(os.path.join(src, "oof.parquet")), t


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default="output/error_analysis/k_compare.json")
    a = ap.parse_args(argv)
    s1 = pd.read_parquet(f"{OUT}/sample_s1.parquet")
    truth = dev_truth(s1)
    res = {}
    for spec in a.runs:
        name, oof, t = parse(spec)
        res[name] = run_metrics(oof, t, truth, s1)
        print(name, json.dumps(res[name]))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2)


if __name__ == "__main__":
    main()
