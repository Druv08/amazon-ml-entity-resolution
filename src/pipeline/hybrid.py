"""Hybrid candidate set: the exact candidates of a base P2 run (K=20) plus the best extra candidates of a deeper run
(K=50), capped per S1.

P2 scales its reserved slots with K, so the K=20 list is NOT the first 20 entries of the K=50 list. The hybrid
therefore keeps every base candidate, in base rank order, and then fills the remaining slots (up to ``cap``) with
deep-run candidates that are not already present, in deep rank order:

    HYBRID = exact base candidates  +  up to (cap - |base|) deep-only candidates

The result is deterministic and has no duplicates. Its blocking recall can never be lower than the base run's.
Frames are long format, one row per (S1, candidate). The column names default to the P3 OOF layout
(s1_id, cand_id, rank); CandidateStore frames use s1_entity_id / candidate_entity_id.
If the hybrid is adopted, candidate_pairs.tsv must be written from the hybrid set, because a plain K=50 list does not
contain every K=20 candidate.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd


def hybrid_candidates(base, deep, cap=50, s1="s1_id", cand="cand_id", rank="rank"):
    """-> DataFrame(s1, cand, origin, base_rank, deep_rank, hybrid_rank), sorted by (s1, hybrid_rank).

    origin is "base" for every candidate of the base run and "deep" for the added deep-only candidates.
    base_rank / deep_rank are the ranks in each run (NaN when absent)."""
    b = base[[s1, cand, rank]].rename(columns={rank: "base_rank"})
    d = deep[[s1, cand, rank]].rename(columns={rank: "deep_rank"})
    if b.duplicated([s1, cand]).any() or d.duplicated([s1, cand]).any():
        raise ValueError("duplicate (S1, candidate) rows in an input run")
    b = b.merge(d, on=[s1, cand], how="left")
    b["origin"] = "base"
    extra = d.merge(b[[s1, cand]], on=[s1, cand], how="left", indicator=True)
    extra = extra[extra["_merge"] == "left_only"].drop(columns="_merge")
    extra = extra.sort_values([s1, "deep_rank", cand], kind="mergesort")
    n_base = b.groupby(s1).size()
    slots = (cap - extra[s1].map(n_base).fillna(0)).clip(lower=0).to_numpy()
    extra = extra[extra.groupby(s1).cumcount().to_numpy() < slots].copy()
    extra["base_rank"] = np.nan
    extra["origin"] = "deep"
    b = b.sort_values([s1, "base_rank", cand], kind="mergesort")
    out = pd.concat([b, extra[b.columns]], ignore_index=True)
    out["_o"] = (out["origin"] == "deep").astype(int)
    out = out.sort_values([s1, "_o", "base_rank", "deep_rank", cand], kind="mergesort").drop(columns="_o")
    out["hybrid_rank"] = out.groupby(s1).cumcount() + 1
    return out.reset_index(drop=True)[[s1, cand, "origin", "base_rank", "deep_rank", "hybrid_rank"]]


def label_pairs(frame, truth, s1="s1_id", cand="cand_id"):
    return np.array([c in truth.get(s, ()) for s, c in zip(frame[s1], frame[cand])], dtype=int)


def overlap_stats(base, deep, truth, country, s1="s1_id", cand="cand_id"):
    """Per-group comparison of two candidate runs. country: {s1_id: country}. Returns {group: stats}."""
    kb = set(zip(base[s1], base[cand]))
    kd = set(zip(deep[s1], deep[cand]))
    rows = []
    for key, where in (("base_only", kb - kd), ("both", kb & kd), ("deep_only", kd - kb)):
        rows += [(s, c, key) for s, c in where]
    f = pd.DataFrame(rows, columns=["s1", "cand", "where"])
    f["true"] = [c in truth.get(s, ()) for s, c in zip(f["s1"], f["cand"])]
    f["country"] = f["s1"].map(country)
    out = {}
    for g, sub in [("overall", f)] + [(c, f[f["country"] == c]) for c in sorted(f["country"].dropna().unique())]:
        n_s1 = sub["s1"].nunique()
        st = {"s1": n_s1}
        for key in ("base_only", "both", "deep_only"):
            m = sub["where"] == key
            st[f"{key}_candidates"] = int(m.sum())
            st[f"{key}_per_s1"] = round(float(m.sum() / max(n_s1, 1)), 2)
            st[f"{key}_true"] = int((m & sub["true"]).sum())
        st["base_candidates_per_s1"] = round((st["base_only_candidates"] + st["both_candidates"]) / max(n_s1, 1), 2)
        st["deep_candidates_per_s1"] = round((st["deep_only_candidates"] + st["both_candidates"]) / max(n_s1, 1), 2)
        out[g] = st
    return out


def blocking_metrics(frame, truth, country, s1_ids, s1="s1_id", cand="cand_id"):
    """Recall / ceiling of a candidate set for the given S1 (the ceiling = perfect decisions within the set)."""
    from src.matching.hard_negatives import entity_f05

    have = frame.groupby(s1)[cand].apply(set).to_dict()
    n_true = rec = full = 0
    by_c, per = {}, []
    for s in s1_ids:
        t, h = truth[s], have.get(s, set())
        found = len(t & h)
        n_true += len(t)
        rec += found
        full += bool(t) and found == len(t)
        per.append(entity_f05(t & h, t))
        c = country[s]
        a = by_c.setdefault(c, [0, 0])
        a[0] += found
        a[1] += len(t)
    matched = sum(1 for s in s1_ids if truth[s])
    n = frame[frame[s1].isin(set(s1_ids))]
    return {"pair_recall": round(rec / max(n_true, 1), 4), "ceiling": round(float(np.mean(per)), 4),
            "cands_per_s1": round(len(n) / max(len(s1_ids), 1), 2),
            "all_true_retained": round(full / max(matched, 1), 4), "true_pairs_found": rec,
            **{f"recall_{c}": round(v[0] / max(v[1], 1), 4) for c, v in sorted(by_c.items())}}


def main(argv=None):
    """Development-sample analysis: overlap of two P3 candidate runs and blocking metrics of their hybrid.

    python -m src.pipeline.hybrid --base output/candidates_p3/k20 --deep output/candidates_p3/k50 --cap 50
    """
    from src.evaluation.k_compare import dev_truth
    from src.matching.sample_candidates import OUT

    ap = argparse.ArgumentParser(description=main.__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default=f"{OUT}/k20")
    ap.add_argument("--deep", default=f"{OUT}/k50")
    ap.add_argument("--cap", type=int, default=50)
    ap.add_argument("--out", default=f"{OUT}/hybrid50")
    a = ap.parse_args(argv)
    s1 = pd.read_parquet(f"{OUT}/sample_s1.parquet")
    truth, country = dev_truth(s1), dict(zip(s1.entity_id, s1.country))
    base, deep = (pd.read_parquet(os.path.join(d, "oof.parquet")) for d in (a.base, a.deep))
    h = hybrid_candidates(base, deep, cap=a.cap)
    rnd = s1.loc[s1.city == "", "entity_id"].tolist()
    report = {"overlap": overlap_stats(base, deep, truth, country),
              "blocking_random_dev_s1": {n: blocking_metrics(f, truth, country, rnd)
                                         for n, f in (("base", base), ("deep", deep), ("hybrid", h))}}
    os.makedirs(a.out, exist_ok=True)
    h.to_parquet(os.path.join(a.out, "hybrid_candidates.parquet"))  # git-ignored (output/)
    with open(os.path.join(a.out, "hybrid_stats.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps(report["blocking_random_dev_s1"], indent=2))


if __name__ == "__main__":
    main()
