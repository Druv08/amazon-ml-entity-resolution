"""Checkpoint 16: true pairs that never reach the matcher (Hybrid50 FN1) and a targeted rescue (development only).

    python -m src.evaluation.blocker_rescue analyze                     # categorise the Hybrid50 FN1 pairs
    python -m src.evaluation.blocker_rescue recall --channel trigram    # recall of a targeted rescue channel
    python -m src.evaluation.blocker_rescue score --channel trigram     # rescued pairs through a rescue classifier

FN1 = a ground-truth pair whose candidate is not in the S1's Hybrid50 candidate list. The categories are
non-exclusive flags computed from the two records only (the same helpers the matcher uses), compared with the true
pairs the blocker does find (TP-reachable). Rescue channels reuse P2's encoded token fields and vocabulary rules and
add at most +5 / +10 candidates per S1. Development sample only; never the final holdout.
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from src.evaluation.hybrid_eval import HybridDev

OUT_DIR = "output/error_analysis"


def load_records(ids, data_dir="data/raw/train"):
    """entity_id, business_name, business_address, country of the given training ids (NA strings -> NaN, as
    matcher.from_store)."""
    from src.blocking.data_io import source_path
    from src.blocking.handoff import load_source_table
    from src.matching.matcher import _COLS, _NA

    ids, parts = set(ids), []
    for s in (1, 2, 3):
        t = load_source_table(source_path(data_dir, "train", s))[_COLS]
        parts.append(t[t["entity_id"].isin(ids)])
    rec = pd.concat(parts, ignore_index=True)
    rec[_COLS[1:3]] = rec[_COLS[1:3]].replace(_NA, np.nan)
    return rec


def pair_traits(pairs, records):
    """Record-only evidence of (s1_id, cand_id) pairs -> DataFrame of similarities and category flags."""
    from rapidfuzz import fuzz, process

    from src.blocking.normalize import has_nonlatin
    from src.matching.extra_features import address_number_features, translit_features
    from src.matching.matcher import _jaccard, _norm

    f = pairs[["s1_id", "cand_id"]].reset_index(drop=True).copy()
    rec = records.set_index("entity_id")
    toks = {}
    for col, key in (("business_name", "name"), ("business_address", "addr")):
        va = rec[col].reindex(f["s1_id"]).tolist()
        vb = rec[col].reindex(f["cand_id"]).tolist()
        for v in set(va) | set(vb):
            if v not in toks:
                toks[v] = _norm(v)
        ja = [" ".join(toks[v]) for v in va]
        jb = [" ".join(toks[v]) for v in vb]
        f[f"{key}_tsort"] = process.cpdist(ja, jb, scorer=fuzz.token_sort_ratio, workers=-1) / 100
        f[f"{key}_tset"] = process.cpdist(ja, jb, scorer=fuzz.token_set_ratio, workers=-1) / 100
        f[f"{key}_jacc"] = [_jaccard(set(toks[a]), set(toks[b])) for a, b in zip(va, vb)]
        f[f"{key}_missing_any"] = [not isinstance(a, str) or not isinstance(b, str) for a, b in zip(va, vb)]
    names_a = rec["business_name"].reindex(f["s1_id"]).fillna("").tolist()
    names_b = rec["business_name"].reindex(f["cand_id"]).fillna("").tolist()
    f["cand_native_script"] = [has_nonlatin(v) for v in names_b]
    f["s1_name_tokens"] = [len(toks.get(v, _norm(v))) for v in names_a]
    f["country"] = rec["country"].reindex(f["s1_id"]).to_numpy()
    tri = lambda v: {v[i:i + 3] for i in range(max(len(v) - 2, 1))}
    na = [" ".join(_norm(v)) for v in names_a]
    nb = [" ".join(_norm(v)) for v in names_b]
    f["name_tri_jacc"] = [_jaccard(tri(a), tri(b)) if a and b else 0.0 for a, b in zip(na, nb)]
    f = pd.concat([f, address_number_features(f, records), translit_features(f, records)], axis=1)
    both_nums = (f["num_shared_n"] + f["num_only_s1_n"] > 0) & (f["num_shared_n"] + f["num_only_cand_n"] > 0)
    flags = {
        "India": f["country"] == "India",
        "US": f["country"] == "US",
        "native script candidate": f["cand_native_script"],
        "transliteration variant (Latin, phonetic match, low token overlap)":
            ~f["cand_native_script"] & (f["name_jacc"] < 0.34) & ((f["ph_jacc"] >= 0.5) | (f["latin_tsort"] >= 0.8)),
        "alias / rebrand (different name, same address)":
            (f["name_tset"] < 0.5) & (f["ph_jacc"] < 0.3) & (f["addr_tsort"] >= 0.7),
        "missing address (either side)": f["addr_missing_any"],
        "short S1 name (<= 1 token)": f["s1_name_tokens"] <= 1,
        "no shared name token": f["name_jacc"] == 0,
        "address-only evidence (no name token, address >= 0.6)": (f["name_jacc"] == 0) & (f["addr_tsort"] >= 0.6),
        "house-number conflict (both numbered, none shared)": both_nums & (f["num_shared_n"] == 0),
        "name char-trigram Jaccard >= 0.5": f["name_tri_jacc"] >= 0.5,
    }
    for k, v in flags.items():
        f[k] = np.asarray(v, dtype=bool)
    return f, list(flags)


def analyze(out=f"{OUT_DIR}/blocker_fn1.json"):
    ctx = HybridDev(native=False)
    have = ctx.in_cands
    rnd = set(ctx.rnd)
    fn1 = [(s, c) for s, t in ctx.truth.items() for c in sorted(t) if c not in have.get(s, ())]
    tp = [(s, c) for s, t in ctx.truth.items() for c in sorted(t) if c in have.get(s, ())]
    frame = pd.DataFrame(fn1 + tp, columns=["s1_id", "cand_id"])
    frame["reach"] = ["FN1"] * len(fn1) + ["reachable"] * len(tp)
    records = load_records(set(frame["s1_id"]) | set(frame["cand_id"]))
    f, flags = pair_traits(frame, records)
    f["reach"], f["random"] = frame["reach"].to_numpy(), f["s1_id"].isin(rnd).to_numpy()
    # S1 context: does the matcher already find something for this S1? (weak-evidence targeting)
    k20 = pd.read_parquet("output/candidates_p3/k20/oof.parquet", columns=["s1_id", "prob"])
    top = k20.groupby("s1_id")["prob"].max()
    f["s1_top_prob"] = f["s1_id"].map(top).fillna(0).to_numpy()
    f["S1 weak evidence (K=20 top prob < 0.5)"] = f["s1_top_prob"] < 0.5
    flags.append("S1 weak evidence (K=20 top prob < 0.5)")
    res = {}
    for scope, m in (("random S1", f["random"]), ("all dev S1", np.ones(len(f), dtype=bool))):
        g = f[m]
        a, b = g[g.reach == "FN1"], g[g.reach == "reachable"]
        res[scope] = {"FN1_pairs": int(len(a)), "reachable_true_pairs": int(len(b)),
                      "FN1_share_of_true": round(len(a) / max(len(g), 1), 4),
                      "flags": {k: {"FN1": int(a[k].sum()), "FN1_share": round(float(a[k].mean()), 4),
                                    "reachable_share": round(float(b[k].mean()), 4)} for k in flags},
                      "medians": {c: {"FN1": round(float(a[c].median()), 3), "reachable": round(float(b[c].median()), 3)}
                                  for c in ("name_tsort", "name_tset", "name_jacc", "name_tri_jacc", "addr_tsort",
                                            "addr_jacc", "ph_jacc")}}
    print(json.dumps(res["random S1"], indent=1))
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2)
    f.to_parquet(f"{OUT_DIR}/blocker_fn1_pairs.parquet")  # git-ignored
    return res


# ---------------------------------------------------------------- targeted rescue channels
# channel -> (fields with weights, max_df): P2's own token fields and vocabulary rules (engine.CountryIndex)
CHANNELS = {"trigram": ((("g", 1.0),), 2000),                  # the engine's trigram fallback vocabulary
            "address": ((("a", 1.5), ("ab", 1.0)), 20000),     # rare address tokens / bigrams, P2's weights
            "name": ((("n", 1.0),), 20000),                    # rare name tokens alone
            "phonetic": ((("p", 0.5), ("pb", 0.5)), 20000)}    # phonetic keys (transliteration-aware), P2's weights


def token_rescue(ctx, channel="trigram", depth=10, chunk=500, cache_root="data/processed/blocking", log=print):
    """Per development S1: the ``depth`` best S2/S3 candidates of its country by the channel's rare-token evidence
    (IDF-weighted overlap, as P2's scorer) that are NOT in its Hybrid50 list. Reuses P2's encoded token fields
    ("g" = trigrams of the transliterated, space-free name; "a"/"ab" = address tokens / bigrams) and the vocabulary
    rule of engine.CountryIndex (tokens with df <= max_df over S1+S2+S3 of the country).
    Deterministic: score desc, then candidate index. -> DataFrame(s1_id, cand_id, rescue_rank, tri_score)."""
    import scipy.sparse as sp

    from src.blocking.encode import EncodedSource, cache_dir
    from src.blocking.engine import _lookup

    fields, max_df = CHANNELS[channel]
    d = cache_dir(cache_root, "train")
    with open(os.path.join(d, "meta.json"), encoding="utf-8") as fh:
        countries = json.load(fh)["countries"]
    enc = {}
    keys = ["ids", "country", "flags"] + [f"{f}_{x}" for f, _ in fields for x in ("ptr", "tok")]
    for s in (1, 2, 3):  # only the channel's fields (the full encoding is ~5 GB)
        with np.load(os.path.join(d, f"source{s}.npz")) as z:
            enc[s] = EncodedSource({k: z[k] for k in keys}, countries)
    s1_ids = ctx.s1.entity_id.to_numpy(dtype=object)
    s1_row = pd.Index(enc[1].ids.astype(str)).get_indexer(s1_ids)
    if (s1_row < 0).any():
        raise ValueError("development S1 missing from the train encoding")
    parts = []
    for country in sorted(enc[1].country_names):
        t0 = time.time()
        rows = {s: enc[s].rows_of_country(country) for s in (1, 2, 3)}
        n_records = sum(len(r) for r in rows.values())
        n_cand = len(rows[2]) + len(rows[3])
        index = {}
        for f, w in fields:
            per = {s: enc[s].row_tokens(f, rows[s]) for s in (1, 2, 3)}
            uniq, df = np.unique(np.concatenate([per[s][1] for s in (1, 2, 3)]), return_counts=True)
            keep = df <= max_df
            vocab, idf = uniq[keep], (np.log((n_records + 1) / df[keep]) * w).astype(np.float32)
            lengths = np.concatenate([per[2][0], per[3][0]])
            toks = np.concatenate([per[2][1], per[3][1]])
            del per, uniq, df
            cand = np.repeat(np.arange(n_cand, dtype=np.int64), lengths)
            col = _lookup(vocab, toks)
            ok = col >= 0
            ct = sp.csr_matrix((np.ones(int(ok.sum()), dtype=np.float32), (col[ok], cand[ok])),
                               shape=(len(vocab), n_cand))
            index[f] = (vocab, idf, ct)
            del cand, col, toks, ok
        cand_ids = np.concatenate([enc[2].ids[rows[2]], enc[3].ids[rows[3]]]).astype(str)
        target = np.flatnonzero(np.isin(s1_row, rows[1]))
        log(f"  [{channel} {country}] index: {sum(len(v[0]) for v in index.values())} tokens, {n_cand} candidates, "
            f"{len(target)} dev S1, {time.time() - t0:.0f}s")
        for a in range(0, len(target), chunk):
            tgt = target[a:a + chunk]
            S = None
            for f, (vocab, idf, ct) in index.items():
                ln, tk = enc[1].row_tokens(f, s1_row[tgt])
                r = np.repeat(np.arange(len(tgt)), ln)
                c = _lookup(vocab, tk)
                m = c >= 0
                W = sp.csr_matrix((idf[c[m]], (r[m], c[m])), shape=(len(tgt), len(vocab)))
                part = W @ ct
                S = part if S is None else S + part
            S = S.tocsr()
            S.sort_indices()
            for i, t in enumerate(tgt):
                lo, hi = S.indptr[i], S.indptr[i + 1]
                cols, vals = S.indices[lo:hi], S.data[lo:hi]
                if not len(cols):
                    continue
                n = min(len(cols), depth + 60)
                sel = np.argpartition(-vals, n - 1)[:n] if len(cols) > n else np.arange(len(cols))
                sel = sel[np.lexsort((cols[sel], -vals[sel]))]
                have = ctx.in_cands.get(s1_ids[t], set())
                picked = [(cand_ids[cols[j]], float(vals[j])) for j in sel if cand_ids[cols[j]] not in have][:depth]
                parts.extend((s1_ids[t], cid, k + 1, v) for k, (cid, v) in enumerate(picked))
        log(f"  [{channel} {country}] done {time.time() - t0:.0f}s")
        del index
    return pd.DataFrame(parts, columns=["s1_id", "cand_id", "rescue_rank", "tri_score"])


def rescue_candidates(ctx, channel, depth=10):
    path = f"{OUT_DIR}/rescue_{channel}{depth}.parquet"
    if not os.path.exists(path):
        token_rescue(ctx, channel, depth=depth).to_parquet(path)  # git-ignored
    return pd.read_parquet(path)


def rescue_recall(channel="trigram"):
    """Recall / ceiling / cost of adding the top 5 or 10 rescue candidates of a channel, for a few targeting rules."""
    from src.blocking.normalize import has_nonlatin
    from src.pipeline.hybrid import blocking_metrics

    out = f"{OUT_DIR}/blocker_rescue_recall_{channel}.json"
    ctx = HybridDev(native=False)
    t0 = time.time()
    res_c = rescue_candidates(ctx, channel)
    seconds = round(time.time() - t0, 1)
    res_c["label"] = [c in ctx.truth.get(s, ()) for s, c in zip(res_c.s1_id, res_c.cand_id)]
    fn1 = pd.read_parquet(f"{OUT_DIR}/blocker_fn1_pairs.parquet")
    fn1 = fn1[fn1.reach == "FN1"]
    country = dict(zip(ctx.s1.entity_id, ctx.s1.country))
    k20 = pd.read_parquet("output/candidates_p3/k20/oof.parquet", columns=["s1_id", "prob"])
    weak = set(k20.groupby("s1_id")["prob"].max().loc[lambda x: x < 0.5].index)
    cand_trait = load_records(set(res_c.cand_id))
    addr_missing = set(cand_trait.loc[cand_trait.business_address.isna(), "entity_id"])
    nonlatin = {e for e, n in zip(cand_trait.entity_id, cand_trait.business_name.fillna("")) if has_nonlatin(n)}
    base = ctx.hybrid[["s1_id", "cand_id"]]
    rules = {
        "all S1, +5": res_c.rescue_rank <= 5,
        "all S1, +10": res_c.rescue_rank <= 10,
        "weak-evidence S1 only (K=20 top prob < 0.5), +10": res_c.s1_id.isin(weak) & (res_c.rescue_rank <= 10),
        "candidates with missing address or non-Latin name, +10":
            res_c.cand_id.isin(addr_missing | nonlatin) & (res_c.rescue_rank <= 10),
        "India S1 only, +10": (res_c.s1_id.map(country) == "India") & (res_c.rescue_rank <= 10),
    }
    rnd = ctx.rnd
    rnd_set = set(rnd)
    report = {"channel": channel, "retrieval_seconds": seconds,
              "hybrid50": blocking_metrics(ctx.hybrid, ctx.truth, country, rnd)}
    for name, m in rules.items():
        add = res_c[m]
        frame = pd.concat([base, add[["s1_id", "cand_id"]]], ignore_index=True)
        met = blocking_metrics(frame, ctx.truth, country, rnd)
        a_r = add[add.s1_id.isin(rnd_set)]
        per_s1 = a_r.groupby("s1_id").size().reindex(rnd).fillna(0)
        rec = set(zip(a_r.s1_id[a_r.label], a_r.cand_id[a_r.label]))
        f = fn1[fn1.random.to_numpy() & np.array([(s, c) in rec for s, c in zip(fn1.s1_id, fn1.cand_id)], dtype=bool)]
        met.update(added_per_s1_mean=round(float(per_s1.mean()), 2), added_per_s1_p95=float(per_s1.quantile(0.95)),
                   added_per_s1_max=int(per_s1.max()), extra_true=int(a_r.label.sum()),
                   extra_false=int((~a_r.label).sum()),
                   recovered_India=int((f.country == "India").sum()), recovered_US=int((f.country == "US").sum()),
                   recovered_native_script=int(f["native script candidate"].sum()),
                   recovered_missing_address=int(f["missing address (either side)"].sum()))
        report[name] = met
        print(f"{name}: {json.dumps(met)}", flush=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    return report


RESCUE_THRESHOLD = 0.85  # pre-declared (= the adopted deep rule), so the headline rescue number involves no tuning


def rescue_features(res_c):
    """Matcher features (A+C+E) of the rescue pairs: the channel score stands in for P2's block / name score and the
    rescue rank for P2's rank; within-S1 and cross-entity features are computed over the rescue lists."""
    import pickle

    from src.matching.extra_features import load_name_stats
    from src.matching.matcher import build_features, xtop

    with open("output/candidates_p3/k50/matcher.pkl", "rb") as fh:
        tfidf = pickle.load(fh)["tfidf"]
    records = load_records(set(res_c.s1_id) | set(res_c.cand_id))
    pairs = res_c.rename(columns={"tri_score": "block_score", "rescue_rank": "rank"})[["s1_id", "cand_id",
                                                                                        "block_score", "rank"]]
    pairs = pairs.assign(name_score=pairs["block_score"]).reset_index(drop=True)
    xt = xtop(pairs, records, tfidf)
    ns = load_name_stats("train")
    grp = pairs["s1_id"].factorize()[0] // 3000
    return pd.concat([build_features(p, records, tfidf, xt, ("A", "C", "E"), ns).astype(np.float32)
                      for _, p in pairs.groupby(grp)]).sort_index()


def rescue_score(channel="trigram", depth=10):
    """Rescue candidates -> 5-fold cross-fitted rescue classifier (folds = the K=20 matcher's S1 folds) -> merged
    after the adopted Hybrid50 decisions with the lowest priority (base > deep > rescue), rescue exclusivity by
    probability, pre-declared threshold RESCUE_THRESHOLD (other thresholds reported for information). Dev only."""
    from sklearn.metrics import roc_auc_score

    from src.evaluation.structured_decoder import Table, adopted_thresholds
    from src.matching.matcher import train
    from src.matching.stream import s1_ranks

    ctx = HybridDev(native=False)
    p20 = pd.read_parquet("output/candidates_p3/k20/oof.parquet", columns=["prob"])["prob"].to_numpy()
    p50 = pd.read_parquet("output/candidates_p3/k50/oof.parquet", columns=["prob"])["prob"].to_numpy()
    T = Table(ctx, p20, p50)
    acc = T.decide(adopted_thresholds(T))
    ref = T.f05(acc)
    res_c = rescue_candidates(ctx, channel)
    res_c = res_c[res_c.rescue_rank <= depth].reset_index(drop=True)
    res_c["label"] = [c in ctx.truth.get(s, ()) for s, c in zip(res_c.s1_id, res_c.cand_id)]
    t0 = time.time()
    X = rescue_features(res_c)
    feat_s = time.time() - t0
    s1_idx = pd.Index(T.s1_ids).get_indexer(res_c.s1_id)
    fold = T.fold[s1_idx]
    y = res_c["label"].to_numpy()
    p = np.zeros(len(res_c))
    for k in range(5):
        tr, va = fold != k, fold == k
        p[va] = train(X[tr], y[tr]).predict_proba(X[va])[:, 1]
    taken = set(T.frame["cand_id"].to_numpy()[acc])  # base and deep matches keep priority
    s1_rank = s1_ranks(T.s1_ids)[s1_idx]  # exclusivity tie-break: smallest s1_id
    rows = []
    for t in (0.5, 0.7, RESCUE_THRESHOLD, 0.9, 0.95):
        f = res_c.assign(prob=p, s1r=s1_rank)
        f = f[(f.prob >= t) & ~f.cand_id.isin(taken)]
        f = f.sort_values(["cand_id", "prob", "s1r"], ascending=[True, False, True], kind="mergesort")
        f = f.drop_duplicates("cand_id")
        tp = np.bincount(T.s1, weights=T.label & acc, minlength=T.n_s1) + \
            np.bincount(s1_idx[f.index], weights=f.label, minlength=T.n_s1)
        npred = np.bincount(T.s1[acc], minlength=T.n_s1) + np.bincount(s1_idx[f.index], minlength=T.n_s1)
        nt = T.n_true
        with np.errstate(divide="ignore", invalid="ignore"):
            pr, rc = tp / npred, tp / nt
            per = np.where(nt == 0, (npred == 0).astype(float),
                           np.where(tp == 0, 0.0, 1.25 * pr * rc / (0.25 * pr + rc)))
        r = T.summary(per, ref=ref)
        rnd_rows = T.random[s1_idx[f.index]]
        lab = f.label.to_numpy()
        r.update(threshold=t, added=int(rnd_rows.sum()), added_tp=int((rnd_rows & lab).sum()),
                 added_fp=int((rnd_rows & ~lab).sum()))
        rows.append(r)
        print(f"rescue T={t}: {json.dumps(r)}", flush=True)
    report = {"channel": channel, "depth": depth, "rescue_rows": int(len(res_c)), "rescue_true": int(y.sum()),
              "rescue_auc": round(float(roc_auc_score(y, p)), 4), "feature_seconds": round(feat_s, 1),
              "pre_declared_threshold": RESCUE_THRESHOLD, "results": rows}
    with open(f"{OUT_DIR}/blocker_rescue_score_{channel}.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["analyze", "recall", "score"])
    ap.add_argument("--channel", choices=list(CHANNELS), default="trigram")
    a = ap.parse_args(argv)
    t0 = time.time()
    if a.stage == "analyze":
        analyze()
    else:
        {"recall": rescue_recall, "score": rescue_score}[a.stage](a.channel)
    print(f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
