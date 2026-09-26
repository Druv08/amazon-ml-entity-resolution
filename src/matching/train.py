"""Train + validate the matcher on P2's candidates (CandidateStore), 5-fold GroupKFold by S1.

python -m src.matching.sample_candidates --top-k 20           # P2's blocker on the P3 train sample
python -m src.matching.train --cands output/candidates_p3/k20
python -m src.matching.train --cands output/candidates_p3/k20 --variants "" "^(rank|name_score)"   # feature ablation

Each --variants entry is a regex of feature columns to DROP ("" = all features); every variant is cross-validated on
the same features, only the first is saved (<cands>/matcher.pkl, <cands>/oof.parquet).
Headline numbers use the hash-random S1 only (unbiased); the city S1 are there so competing S1 are scored together.
"""
import argparse
import pickle
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from src.blocking.handoff import CandidateStore
from src.matching.matcher import (best_threshold, build_features, exclusive, fit_tfidf, from_store, macro_f05,
                                  to_matches, train, xtop)
from src.matching.sample_candidates import OUT


def f05(pred, truth, ids):
    return macro_f05(pred, {s: truth[s] for s in ids})


def report(pairs, prob, truth, s1, hard):
    """Final decision rule (exclusivity + threshold tuned for macro-F0.5 on all sampled S1) -> the numbers P2 asked for."""
    y = pairs["label"].to_numpy()
    ex = exclusive(pairs, prob)
    t, _ = best_threshold(pairs, ex, truth)
    pred, perfect = to_matches(pairs, ex, t, truth), to_matches(pairs, y.astype(float), 0.5, truth)
    rnd, city = s1[s1.city == ""], s1[s1.city != ""]
    r = {"auc": roc_auc_score(y, prob), "threshold": t, "f05": f05(pred, truth, rnd.entity_id),
         "ceiling": f05(perfect, truth, rnd.entity_id), "f05_cities": f05(pred, truth, city.entity_id),
         "f05_cities_no_excl": f05(to_matches(pairs, prob, t, truth), truth, city.entity_id)}
    print(f"  pair AUC {r['auc']:.4f} | threshold {t} | macro-F0.5 {r['f05']:.4f} (ceiling {r['ceiling']:.4f}, "
          f"{len(rnd)} random S1) | cities {r['f05_cities']:.4f} (no exclusivity {r['f05_cities_no_excl']:.4f})")
    for c, g in rnd.groupby("country"):
        print(f"  {c:6s} {len(g):6d} S1: macro-F0.5 {f05(pred, truth, g.entity_id):.4f}, "
              f"ceiling {f05(perfect, truth, g.entity_id):.4f}")
    single = [s for s in rnd.entity_id if not truth[s]]
    matched = [s for s in rnd.entity_id if truth[s]]
    r["singletons_empty"] = np.mean([not pred[s] for s in single])
    print(f"  singletons {len(single)}: {r['singletons_empty']:.3f} correctly empty | "
          f"matched S1 {len(matched)}: macro-F0.5 {f05(pred, truth, matched):.4f}")
    acc, neg = ex >= t, y == 0
    fp = (acc & neg).sum()
    for name, m in [("all negatives", neg)] + [(k, neg & v) for k, v in hard.items()]:
        print(f"  {name:28s} {m.sum():9d} pairs, {(acc & m).sum() / max(m.sum(), 1):.4%} accepted "
              f"({(acc & m).sum() / max(fp, 1):.0%} of FPs)")
    return r


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cands", default=f"{OUT}/k20", help="out_dir of src.matching.sample_candidates")
    ap.add_argument("--variants", nargs="+", default=[""], help="regexes of feature columns to drop, one per variant")
    a = ap.parse_args()
    t0 = time.time()

    store = CandidateStore("train", out_dir=a.cands)
    pairs, records = from_store(pd.concat(store.iter_frames(with_records=True, with_labels=True), ignore_index=True))
    top_k = store.meta["config"]["top_k"]
    del store
    s1 = pd.read_parquet(f"{OUT}/sample_s1.parquet")
    gt = pd.read_csv("data/raw/train/train_ground_truth.tsv", sep="\t", dtype=str, keep_default_na=False)
    claimed = gt.matched_entity_ids.str.split(",").explode()  # every S2/S3 that is SOME S1's match
    gt = gt[gt.source1_entity_id.isin(s1.entity_id)]
    truth = {s: set(filter(None, m.split(","))) for s, m in zip(gt.source1_entity_id, gt.matched_entity_ids)}
    n_true = sum(map(len, truth.values()))
    print(f"loaded {time.time() - t0:.0f}s: {len(pairs)} pairs, {len(truth)} S1, {len(pairs) / len(truth):.1f} "
          f"cands/S1, pair recall {pairs.label.sum() / n_true:.4f}", flush=True)

    tfidf = fit_tfidf(records)
    xt = xtop(pairs, records, tfidf)  # cross-entity top-2 over the whole sample, features built per 3k-S1 chunk
    grp = pairs["s1_id"].factorize()[0] // 3000
    X = pd.concat([build_features(p, records, tfidf, xt).astype(np.float32) for _, p in pairs.groupby(grp)]).sort_index()
    y = pairs["label"].to_numpy()
    print(f"features {time.time() - t0:.0f}s: {X.shape}, positive rate {y.mean():.3f}", flush=True)
    hard = {"blocker rank 1-3": (pairs["rank"] <= 3).to_numpy(),
            "lookalike name (tsort>=0.9)": (X["name_tsort"] >= 0.9).to_numpy(),
            "claimed by another S1": pairs["cand_id"].isin(claimed).to_numpy()}
    del claimed

    for i, v in enumerate(a.variants):
        Xv = X.drop(columns=X.filter(regex=v).columns) if v else X
        oof = np.zeros(len(pairs))
        for tr, va in GroupKFold(5).split(Xv, y, pairs["s1_id"]):
            oof[va] = train(Xv.iloc[tr], y[tr]).predict_proba(Xv.iloc[va])[:, 1]
        print(f"variant drop={v!r} ({Xv.shape[1]} features) CV {time.time() - t0:.0f}s", flush=True)
        r = report(pairs, oof, truth, s1, hard)
        if i == 0:
            pairs[["s1_id", "cand_id", "rank", "label"]].assign(prob=oof).to_parquet(f"{a.cands}/oof.parquet")  # P4
            with open(f"{a.cands}/matcher.pkl", "wb") as fh:  # predict.py always applies exclusive()
                pickle.dump({"model": train(Xv, y), "threshold": r["threshold"], "features": list(Xv.columns),
                             "tfidf": tfidf, "top_k": top_k}, fh)
            print(f"  saved {a.cands}/matcher.pkl + oof.parquet, {time.time() - t0:.0f}s", flush=True)
