"""Verify production Hybrid50 inference (predict_hybrid.py) against the offline decision logic on real smoke runs.

    python -m src.pipeline.hybrid_equivalence --base-cands output/p3_smoke/k20_5000 --deep-cands output/p3_smoke/k50_5000
    ... --decoder threshold                                   # the earlier threshold rule instead of the meta decoder

Reference = every pair of both runs scored IN MEMORY (global cross-entity top-2 over the whole run: the pre-streaming
predict path) and the hybrid built with hybrid_candidates() on the whole run. Meta decoder: meta_decoder.meta_scores()
over all rows at once and meta_decoder.decide() (vectorised exclusivity). Threshold rule: base_predictions()
(exclusive + threshold) and merge_deep() with deep prob >= DEEP_THRESHOLD. Both official files must be
byte-identical to the production ones.
Test split (no labels): this checks the implementation, not the score. Output goes under output/ (git-ignored).
"""

import argparse
import filecmp
import json
import os
import pickle
import time

import numpy as np
import pandas as pd

from src.blocking.generate_candidates import write_candidate_pairs
from src.blocking.handoff import CandidateStore
from src.matching.extra_features import NameStats
from src.matching.matcher import build_features, from_store, top2, write_matching_results, xtop
from src.pipeline.deep_recovery import DEEP_THRESHOLD, base_predictions, deep_pool, merge_deep
from src.pipeline.hybrid import hybrid_candidates


def score_in_memory(store, model, name_stats=None, with_features=False, extra=()):
    """DataFrame(s1_id, cand_id, rank, prob) of every pair of a run, with the cross-entity top-2 over all of it
    (plus the float32 feature matrix in model["features"] order when with_features)."""
    groups = tuple(model.get("feature_groups", ()))
    ns = name_stats if "E" in groups else None
    tops = [xtop(*from_store(df), model["tfidf"]) for df in store.iter_frames(with_records=True)]
    xt = {}
    for c in tops[0]:
        d = pd.concat(t[c] for t in tops)
        xt[c] = top2(d["s"], d["c"])
    parts, feats = [], []
    for df in store.iter_frames(with_records=True):
        pairs, records = from_store(df)
        X = build_features(pairs, records, model["tfidf"], xt, groups, ns)[model["features"]]
        parts.append(pairs[["s1_id", "cand_id"]].assign(rank=df["rank"].to_numpy(),
                                                        prob=model["model"].predict_proba(X)[:, 1].astype(np.float32)))
        if with_features:
            F = X.to_numpy(dtype=np.float32)
            if "decoy" in extra:
                from src.matching.decoy_features import decoy_features

                F = np.hstack([F, decoy_features(pairs, records).to_numpy(dtype=np.float32)])
            feats.append(F)
    scored = pd.concat(parts, ignore_index=True)
    return (scored, np.vstack(feats)) if with_features else scored


def offline_meta_reference(base_store, deep_store, bundle, matching_path, candidate_path, name_stats=None):
    """The meta decoder computed in memory over the whole run, written as the two official files."""
    from src.matching.stream import s1_ranks
    from src.pipeline.meta_decoder import decide, meta_scores

    cap, floor, extra = bundle["hybrid"]["cap"], bundle["meta_floor"], tuple(bundle.get("meta_extra", ()))
    b, Xb = score_in_memory(base_store, bundle["base_matcher"], name_stats, with_features=True, extra=extra)
    d, Xd = score_in_memory(deep_store, bundle["deep_matcher"], name_stats, with_features=True, extra=extra)
    hybrid = hybrid_candidates(b, d, cap=cap)
    deep = hybrid[hybrid["origin"] == "deep"]  # sorted by S1, then K=50 rank
    pos = pd.Series(np.arange(len(d)), index=pd.MultiIndex.from_frame(d[["s1_id", "cand_id"]]))
    dpos = pos.reindex(pd.MultiIndex.from_frame(deep[["s1_id", "cand_id"]])).to_numpy()
    s1_id = np.concatenate([b["s1_id"].to_numpy(), d["s1_id"].to_numpy()[dpos]])
    cand = np.concatenate([b["cand_id"].to_numpy(), d["cand_id"].to_numpy()[dpos]])
    prob = np.concatenate([b["prob"].to_numpy(), d["prob"].to_numpy()[dpos]]).astype(np.float64)
    rank = np.concatenate([b["rank"].to_numpy(), d["rank"].to_numpy()[dpos]])
    origin = np.r_[np.zeros(len(b), dtype=np.int64), np.ones(len(dpos), dtype=np.int64)]
    X = np.vstack([Xb, Xd[dpos]])
    s1_codes = pd.factorize(s1_id)[0]
    score = meta_scores(bundle["meta_model"], prob, origin, rank, s1_codes, X[prob >= floor],
                        bundle["pair_features"], floor)
    all_s1 = base_store.source_tables()[1]["entity_id"].to_numpy(dtype=object)
    s1_rank = pd.Series(s1_ranks(all_s1), index=all_s1).reindex(s1_id).to_numpy()
    acc = decide(cand, origin, score, s1_rank, bundle["thresholds"]["base"], bundle["thresholds"]["deep"])
    s1_ids = all_s1[:base_store.meta["s1_limit"]]
    pred = {s: set() for s in s1_ids}
    for s, c in zip(s1_id[acc], cand[acc]):
        pred[s].add(c)
    write_matching_results(pred, matching_path)
    lists = hybrid.groupby("s1_id", sort=False)["cand_id"].agg(list).to_dict()
    write_candidate_pairs(candidate_path, ((s, lists.get(s, [])) for s in s1_ids))
    return {"base_matches": int((acc & (origin == 0)).sum()), "deep_matches": int((acc & (origin == 1)).sum()),
            "meta_rows": int((score > -np.inf).sum()), "hybrid_candidates": len(hybrid)}


def offline_reference(base_scored, deep_scored, base_threshold, s1_ids, matching_path, candidate_path,
                      cap=50, deep_threshold=DEEP_THRESHOLD):
    """The offline Hybrid50 decision (src/pipeline/deep_recovery.py) written as the two official files."""
    keys = {s: set() for s in s1_ids}  # only the S1 keys are used: no labels on test
    hybrid = hybrid_candidates(base_scored, deep_scored, cap=cap)
    base_pred, owner = base_predictions(base_scored, base_threshold, keys)
    deep = deep_pool(hybrid, deep_scored, keys)
    pred, added = merge_deep(base_pred, owner, deep, deep["prob"].to_numpy() >= deep_threshold)
    write_matching_results(pred, matching_path)
    lists = hybrid.groupby("s1_id", sort=False)["cand_id"].agg(list).to_dict()
    write_candidate_pairs(candidate_path, ((s, lists.get(s, [])) for s in s1_ids))
    return {"base_matches": sum(len(v) for v in base_pred.values()), "deep_matches": len(added),
            "deep_rows": len(deep), "hybrid_candidates": len(hybrid)}


def main(argv=None):
    from src.pipeline.predict_hybrid import predict_hybrid, predict_hybrid_meta

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-cands", required=True)
    ap.add_argument("--deep-cands", required=True)
    ap.add_argument("--base-model", default="output/candidates_p3/k20/matcher.pkl")
    ap.add_argument("--deep-model", default="output/candidates_p3/k50/matcher.pkl")
    ap.add_argument("--cap", type=int, default=50)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--decoder", choices=["meta", "threshold"], default="meta")
    ap.add_argument("--bundle", default="output/candidates_p3/hybrid_meta.pkl")
    ap.add_argument("--out", default=None, help="default: <base-cands>/hybrid_check_<decoder>")
    a = ap.parse_args(argv)
    out = a.out or os.path.join(a.base_cands, f"hybrid_check_{a.decoder}")
    os.makedirs(out, exist_ok=True)
    base = CandidateStore("test", out_dir=a.base_cands)
    deep = CandidateStore("test", out_dir=a.deep_cands, data_dir=base.data_dir, tables=base.source_tables())
    t0 = time.time()
    log = lambda msg: print(f"  {msg}, {time.time() - t0:.0f}s", flush=True)  # noqa: E731
    paths = os.path.join(out, "matching_results.tsv"), os.path.join(out, "candidate_pairs.tsv")
    ref_paths = os.path.join(out, "ref_matching_results.tsv"), os.path.join(out, "ref_candidate_pairs.tsv")
    if a.decoder == "meta":
        with open(a.bundle, "rb") as fh:
            bundle = pickle.load(fh)
        prod = predict_hybrid_meta(base, deep, bundle, *paths, workers=a.workers, log=log)
        print("production:", json.dumps(prod), flush=True)
        t1 = time.time()
        ref = offline_meta_reference(base, deep, bundle, *ref_paths, NameStats.from_tables(base.source_tables()))
    else:
        models = []
        for path in (a.base_model, a.deep_model):
            with open(path, "rb") as fh:
                models.append(pickle.load(fh))
        prod = predict_hybrid(base, deep, *models, *paths, cap=a.cap, workers=a.workers, log=log)
        print("production:", json.dumps(prod), flush=True)
        t1 = time.time()
        ns = NameStats.from_tables(base.source_tables())
        base_scored, deep_scored = score_in_memory(base, models[0], ns), score_in_memory(deep, models[1], ns)
        s1_ids = base.source_tables()[1]["entity_id"].to_numpy(dtype=object)[:base.meta["s1_limit"]]
        ref = offline_reference(base_scored, deep_scored, models[0]["threshold"], s1_ids, *ref_paths, cap=a.cap)
    ref["seconds"] = round(time.time() - t1, 1)
    print("reference:", json.dumps(ref), flush=True)
    same = {f: filecmp.cmp(os.path.join(out, f), os.path.join(out, "ref_" + f), shallow=False)
            for f in ("matching_results.tsv", "candidate_pairs.tsv")}
    counts = all(prod[k] == ref[k] for k in ("base_matches", "deep_matches"))
    report = {"identical": same, "counts_equal": counts, "production": prod, "reference": ref}
    with open(os.path.join(out, "equivalence.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps({"identical": same, "counts_equal": counts}))
    if not (all(same.values()) and counts):
        raise SystemExit("production Hybrid50 output differs from the offline reference")


if __name__ == "__main__":
    main()
