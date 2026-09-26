"""Error analysis of the matcher's out-of-fold decisions on the P3 development sample.

    python -m src.evaluation.matcher_errors --cands output/candidates_p3/k20

Uses <cands>/oof.parquet (written by src.matching.train) and the development sample only. NEVER the final holdout.
Applies the production decision rule (exclusive() + the matcher.pkl threshold) and classifies every error:

    FP    accepted pair whose candidate is not a true match of that S1
    FN1   true pair that is not in the K candidates at all (blocking loss)
    FN2   true pair in the candidates, probability below the threshold
    FN3   true pair above the threshold but lost to exclusivity (another S1 won the candidate)

Each error type is broken down by the categories in CATEGORY_DOC (count, share, mean probability / rank, name and
address similarity). The most common category combinations ("failure patterns") are counted too. Reports contain
aggregates only, never raw records, and go to output/error_analysis/ (git-ignored).
Feature matrices are cached there as well, so later experiments can reuse them.
"""

import argparse
import json
import os
import pickle
import time

import numpy as np
import pandas as pd

from src.blocking.features import record_features
from src.blocking.normalize import has_nonlatin

OUT_DIR = "output/error_analysis"
RANK_BUCKETS = [(1, 1, "1"), (2, 2, "2"), (3, 3, "3"), (4, 10, "4-10"), (11, 20, "11-20"), (21, 10**9, ">20")]
SIM_COLUMNS = ["name_tfidf", "name_tsort", "core_tsort", "addr_tfidf", "addr_tsort"]
CATEGORY_DOC = {
    "rank_1": "blocker rank 1", "rank_2": "blocker rank 2", "rank_3": "blocker rank 3",
    "rank_4-10": "blocker rank 4-10", "rank_11-20": "blocker rank 11-20", "rank_>20": "blocker rank > 20",
    "India": "S1 country India", "US": "S1 country US",
    "singleton_s1": "S1 has no true match", "matched_s1": "S1 has at least one true match",
    "s1_address_missing": "S1 address missing", "cand_address_missing": "candidate address missing",
    "both_addresses": "both addresses present",
    "num_conflict": "both addresses have numbers, none shared",
    "first_num_equal": "first house number equal", "first_num_unequal": "first house number differs",
    "first_num_missing": "a side has no number",
    "name_high_addr_weak": "name_tsort >= 0.9 and addr_tsort < 0.5",
    "core_name_high": "core (legal-form free) name token-sort >= 0.9",
    "claimed_by_other_s1": "candidate is another S1's true match in training GT",
    "multi_s1_candidate": "candidate appears in more than one S1's candidate list",
    "nonlatin_candidate": "candidate name contains Indic / non-Latin script",
    "translit_phonetic_match": "non-Latin candidate sharing a P2 phonetic key with the Latin S1 name",
    "S2": "candidate from source 2", "S3": "candidate from source 3",
}


# ---------------------------------------------------------------- data
def load_dev(cands, cache_dir=OUT_DIR):
    """-> dict(pairs, records, truth, s1, claimed, X, model) for the development sample, features cached."""
    from src.matching.train import featurize, load

    tag = os.path.basename(os.path.normpath(cands))
    cache = os.path.join(cache_dir, f"dev_{tag}.pkl")
    if os.path.exists(cache):
        with open(cache, "rb") as fh:
            d = pickle.load(fh)
    else:
        pairs, records, truth, s1, claimed, top_k = load(cands)
        X, y, _, _ = featurize(pairs, records, claimed)
        d = {"pairs": pairs, "records": records, "truth": truth, "s1": s1, "claimed": set(claimed), "X": X,
             "top_k": top_k}
        os.makedirs(cache_dir, exist_ok=True)
        with open(cache, "wb") as fh:
            pickle.dump(d, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return d


def rank_bucket(rank):
    rank = np.asarray(rank)
    out = np.empty(len(rank), dtype=object)
    for lo, hi, name in RANK_BUCKETS:
        out[(rank >= lo) & (rank <= hi)] = name
    return out


def decide(pairs, prob, threshold):
    """Production decision rule -> (accepted mask, exclusive probs)."""
    from src.matching.matcher import exclusive

    ex = exclusive(pairs, np.asarray(prob))
    return ex >= threshold, ex


def classify_errors(label, prob, ex, threshold):
    """Per candidate pair: 'FP', 'FN2', 'FN3', 'TP' or 'TN' (FN1 pairs are not in the candidate list)."""
    label, prob, ex = np.asarray(label), np.asarray(prob), np.asarray(ex)
    acc = ex >= threshold
    out = np.full(len(label), "TN", dtype=object)
    out[acc & (label == 1)] = "TP"
    out[acc & (label == 0)] = "FP"
    out[~acc & (label == 1) & (prob < threshold)] = "FN2"
    out[~acc & (label == 1) & (prob >= threshold)] = "FN3"
    return out


def missing_true_pairs(pairs, truth):
    """True (s1, cand) pairs of the sampled S1 that are not in the candidate lists (FN1)."""
    have = set(zip(pairs["s1_id"], pairs["cand_id"]))
    return [(s, c) for s, cs in truth.items() for c in cs if (s, c) not in have]


def pair_categories(pairs, X, records, truth, claimed):
    """Boolean category columns (CATEGORY_DOC) for every candidate pair."""
    rec = records.set_index("entity_id")
    s1_addr = rec["business_address"].reindex(pairs["s1_id"]).isna().to_numpy()
    c_addr = rec["business_address"].reindex(pairs["cand_id"]).isna().to_numpy()
    s1_country = rec["country"].reindex(pairs["s1_id"]).to_numpy()
    c_name = rec["business_name"].reindex(pairs["cand_id"]).fillna("")
    s1_name = rec["business_name"].reindex(pairs["s1_id"]).fillna("")
    cat = pd.DataFrame(index=pairs.index)
    buckets = rank_bucket(pairs["rank"].to_numpy())
    for _, _, b in RANK_BUCKETS:
        cat[f"rank_{b}"] = buckets == b
    cat["India"], cat["US"] = s1_country == "India", s1_country == "US"
    single = np.array([not truth.get(s) for s in pairs["s1_id"]])
    cat["singleton_s1"], cat["matched_s1"] = single, ~single
    cat["s1_address_missing"], cat["cand_address_missing"] = s1_addr, c_addr
    cat["both_addresses"] = ~s1_addr & ~c_addr
    cat["num_conflict"] = X["num_conflict"].to_numpy() == 1
    cat["first_num_equal"] = X["first_num_eq"].to_numpy() == 1
    cat["first_num_unequal"] = X["first_num_eq"].to_numpy() == 0
    cat["first_num_missing"] = X["first_num_eq"].to_numpy() == -1
    cat["name_high_addr_weak"] = (X["name_tsort"].to_numpy() >= 0.9) & (X["addr_tsort"].to_numpy() < 0.5)
    cat["core_name_high"] = X["core_tsort"].to_numpy() >= 0.9
    own = np.array([c in claimed and c not in truth.get(s, ()) for s, c in zip(pairs["s1_id"], pairs["cand_id"])])
    cat["claimed_by_other_s1"] = own
    cat["multi_s1_candidate"] = (pairs.groupby("cand_id")["s1_id"].transform("size") > 1).to_numpy()
    nonlatin_cache = {v: has_nonlatin(v) for v in set(c_name)}
    cat["nonlatin_candidate"] = c_name.map(nonlatin_cache).to_numpy()
    cat["translit_phonetic_match"] = _phonetic_match(s1_name.to_numpy(), c_name.to_numpy(), cat["nonlatin_candidate"])
    cat["S2"] = pairs["cand_id"].str.startswith("S2-").to_numpy()
    cat["S3"] = ~cat["S2"]
    return cat


def _phonetic_match(s1_names, cand_names, nonlatin):
    """For non-Latin candidates: does any P2 phonetic key (after offline transliteration) match the S1 name?"""
    out = np.zeros(len(s1_names), dtype=bool)
    cache = {}

    def keys(v):
        if v not in cache:
            cache[v] = record_features(v, "", with_fallback=False)["p"]
        return cache[v]

    for i in np.flatnonzero(np.asarray(nonlatin)):
        out[i] = bool(keys(s1_names[i]) & keys(cand_names[i]))
    return out


def breakdown(mask, cat, prob, rank, X):
    """Stats of the pairs in ``mask`` overall and per category."""
    total = int(mask.sum())
    rows = {}
    for c in ["all"] + list(cat.columns):
        m = mask if c == "all" else mask & cat[c].to_numpy()
        n = int(m.sum())
        if n == 0 and c != "all":
            continue
        row = {"count": n, "share": round(n / max(total, 1), 4),
               "mean_prob": round(float(np.mean(prob[m])), 4) if n else None,
               "mean_rank": round(float(np.mean(rank[m])), 2) if n else None}
        for s in SIM_COLUMNS:
            row[f"mean_{s}"] = round(float(np.mean(X[s].to_numpy()[m])), 3) if n else None
        rows[c] = row
    return rows


def patterns(mask, cat, top=20):
    """Most common combinations of the main categorical attributes among the pairs in ``mask``."""
    parts = [
        rank_bucket_label(cat),
        np.where(cat["India"], "India", "US"),
        np.where(cat["singleton_s1"], "singletonS1", "matchedS1"),
        np.where(cat["both_addresses"], "both_addr", np.where(cat["cand_address_missing"], "cand_addr_missing",
                                                               "s1_addr_missing")),
        np.where(cat["num_conflict"], "num_conflict", np.where(cat["first_num_equal"], "num_equal", "num_other")),
        np.where(cat["core_name_high"], "core_name_high", "core_name_low"),
        np.where(cat["claimed_by_other_s1"], "claimed_by_other", "unclaimed"),
        np.where(cat["nonlatin_candidate"], "nonlatin", "latin"),
    ]
    sig = pd.Series([" | ".join(p) for p in zip(*parts)])[np.asarray(mask)]
    vc = sig.value_counts().head(top)
    total = int(np.asarray(mask).sum())
    return [{"pattern": k, "count": int(v), "share": round(int(v) / max(total, 1), 4)} for k, v in vc.items()]


def rank_bucket_label(cat):
    out = np.empty(len(cat), dtype=object)
    for _, _, b in RANK_BUCKETS:
        out[cat[f"rank_{b}"].to_numpy()] = f"rank {b}"
    return out


def fn1_breakdown(missing, truth, records_all):
    """Blocking losses by S1 country, source and candidate script (records_all: id -> (name, country))."""
    rows = {"count": len(missing)}
    c = pd.Series([records_all.get(s, ("", ""))[1] for s, _ in missing]).value_counts().to_dict()
    rows["by_s1_country"] = c
    rows["by_source"] = pd.Series([m[:2] for _, m in missing]).value_counts().to_dict()
    rows["nonlatin_candidate"] = int(sum(has_nonlatin(records_all.get(m, ("", ""))[0]) for _, m in missing))
    return rows


def analyse(d, oof, threshold, raw_names=None):
    pairs, X, truth = d["pairs"], d["X"], d["truth"]
    key = pairs[["s1_id", "cand_id"]].merge(oof[["s1_id", "cand_id", "prob"]], how="left", on=["s1_id", "cand_id"])
    if key["prob"].isna().any() or len(key) != len(pairs):
        raise ValueError("oof.parquet does not match the candidate pairs")
    prob = key["prob"].to_numpy()
    acc, ex = decide(pairs, prob, threshold)
    kind = classify_errors(pairs["label"].to_numpy(), prob, ex, threshold)
    cat = pair_categories(pairs, X, d["records"], truth, d["claimed"])
    rank = pairs["rank"].to_numpy()
    rnd = set(d["s1"].loc[d["s1"].city == "", "entity_id"])
    in_rnd = pairs["s1_id"].isin(rnd).to_numpy()
    missing = missing_true_pairs(pairs, truth)
    single_s1 = [s for s in truth if not truth[s]]
    accepted_s1 = set(pairs.loc[acc, "s1_id"])
    report = {
        "threshold": threshold,
        "pairs": len(pairs), "s1": len(truth), "true_pairs": int(sum(map(len, truth.values()))),
        "counts": {k: int((kind == k).sum()) for k in ("TP", "FP", "FN2", "FN3")} | {"FN1": len(missing)},
        "counts_random_dev_s1": {k: int(((kind == k) & in_rnd).sum()) for k in ("TP", "FP", "FN2", "FN3")},
        "singleton_s1_with_a_match": int(sum(s in accepted_s1 for s in single_s1)),
        "singleton_s1": len(single_s1),
        "FP": breakdown(kind == "FP", cat, prob, rank, X),
        "FN2": breakdown(kind == "FN2", cat, prob, rank, X),
        "FN3": breakdown(kind == "FN3", cat, prob, rank, X),
        "FP_patterns": patterns(kind == "FP", cat),
        "FN2_patterns": patterns(kind == "FN2", cat),
        "category_doc": CATEGORY_DOC,
    }
    if raw_names is not None:
        report["FN1"] = fn1_breakdown(missing, truth, raw_names)
    fp_s1 = pairs.loc[kind == "FP", "s1_id"]
    report["fp_per_s1"] = {"s1_with_fp": int(fp_s1.nunique()),
                           "singleton_s1_fp_pairs": int((kind == "FP")[cat["singleton_s1"].to_numpy()].sum())}
    return report, kind, cat, prob


def raw_lookup(ids, data_dir="data/raw/train"):
    """id -> (name, country) for the given S1/S2/S3 ids, streaming the raw files."""
    from src.blocking.data_io import iter_records, source_path

    ids, out = set(ids), {}
    for s in (1, 2, 3):
        for r in iter_records(source_path(data_dir, "train", s)):
            if r.entity_id in ids:
                out[r.entity_id] = (r.name, r.country)
    return out


def to_markdown(rep):
    lines = [f"# Matcher error analysis (threshold {rep['threshold']})", "",
             f"pairs {rep['pairs']:,} | S1 {rep['s1']:,} | true pairs {rep['true_pairs']:,}", "",
             "| type | count |", "|---|---|"]
    lines += [f"| {k} | {v:,} |" for k, v in rep["counts"].items()]
    lines += ["", f"singleton S1 given a match: {rep['singleton_s1_with_a_match']} of {rep['singleton_s1']}", ""]
    for kind in ("FP", "FN2", "FN3"):
        lines += [f"## {kind}", "", "| category | count | share | mean prob | mean rank | name_tsort | addr_tsort |",
                  "|---|---|---|---|---|---|---|"]
        for c, r in rep[kind].items():
            lines.append(f"| {c} | {r['count']:,} | {r['share']:.1%} | {r['mean_prob']} | {r['mean_rank']} | "
                         f"{r['mean_name_tsort']} | {r['mean_addr_tsort']} |")
        lines.append("")
    for kind in ("FP_patterns", "FN2_patterns"):
        lines += [f"## top {kind.replace('_', ' ')}", "", "| pattern | count | share |", "|---|---|---|"]
        lines += [f"| {p['pattern']} | {p['count']} | {p['share']:.1%} |" for p in rep[kind]]
        lines.append("")
    if "FN1" in rep:
        lines += ["## FN1 (not in candidates)", "", "```", json.dumps(rep["FN1"], indent=2), "```"]
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cands", default="output/candidates_p3/k20")
    ap.add_argument("--out-dir", default=OUT_DIR)
    a = ap.parse_args(argv)
    t0 = time.time()
    d = load_dev(a.cands, a.out_dir)
    with open(os.path.join(a.cands, "matcher.pkl"), "rb") as fh:
        threshold = pickle.load(fh)["threshold"]
    oof = pd.read_parquet(os.path.join(a.cands, "oof.parquet"))
    missing = missing_true_pairs(d["pairs"], d["truth"])
    names = raw_lookup({s for s, _ in missing} | {c for _, c in missing})
    rep, *_ = analyse(d, oof, threshold, names)
    tag = os.path.basename(os.path.normpath(a.cands))
    os.makedirs(a.out_dir, exist_ok=True)
    with open(os.path.join(a.out_dir, f"errors_{tag}.json"), "w", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=2, default=str)
    with open(os.path.join(a.out_dir, f"errors_{tag}.md"), "w", encoding="utf-8") as fh:
        fh.write(to_markdown(rep))
    print(to_markdown(rep))
    print(f"\nwrote {a.out_dir}/errors_{tag}.json/.md in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
