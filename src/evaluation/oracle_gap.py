"""Checkpoint B: where does the macro-F0.5 of the current system go? (development labels, diagnostics only)

    python -m src.evaluation.structured_decoder strict_meta --first M3    # writes strict_rows_M3.parquet (once)
    python -m src.evaluation.oracle_gap                                   # ladder + error sources + splits

Input: the strictly nested held-out decisions of the adopted system (M3 first stage + meta decoder, 0.9574), one row
per Hybrid50 candidate with its first-stage probability, meta score (-inf below the floor) and decision.

Oracle ladder (random development S1):
  O0  current decisions
  O1  per S1, the best prefix of the S1's candidates in CURRENT score order (labels choose the cut only)
  O2  per S1, exactly the true candidates present in Hybrid50 (the candidate ceiling)
  O3  per S1, exactly the true candidates present in Hybrid50 or the K=100 run (diagnostic)
  O4  ground truth (1.0)
Per S1 the loss telescopes: 1 - F(O0) = [1 - F(O2)] blocking + [F(O2) - F(O1)] ranking + [F(O1) - F(O0)] decoder.

Pair-level error sources (each error in exactly one class; "recoverable" = macro-F0.5 gain if ONLY that class were
fixed: its false positives removed or its false negatives accepted):
  FN  missing from candidates | below the meta floor | lost to exclusivity | outranked by a false candidate of the S1 |
      correctly ranked but below the threshold
  FP  singleton S1 | outranks a true candidate of the S1 | all present trues rank above it | S1's trues all missing
Never the final holdout.
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd

OUT_DIR = "output/error_analysis"


def f05(tp, npred, nt):
    """Vectorised per-S1 F0.5 (matcher.macro_f05 conventions: singleton = 1 iff nothing predicted)."""
    tp, npred, nt = (np.asarray(x, dtype=float) for x in (tp, npred, nt))
    with np.errstate(divide="ignore", invalid="ignore"):
        p, r = tp / npred, tp / nt
        f = 1.25 * p * r / (0.25 * p + r)
    return np.where(nt == 0, (npred == 0).astype(float), np.where(tp == 0, 0.0, f))


def best_prefix(s1, score, tie, label, n_s1, nt):
    """Per S1 the best F0.5 over prefixes (including the empty one) of its rows ordered by score desc, tie desc."""
    order = np.lexsort((-tie, -score, s1))
    s, lab = s1[order], label[order].astype(float)
    k = pd.Series(np.ones(len(s))).groupby(s).cumsum().to_numpy()
    tp = pd.Series(lab).groupby(s).cumsum().to_numpy()
    f = f05(tp, k, nt[s])
    best = np.where(nt == 0, 1.0, 0.0)  # the empty prefix
    np.maximum.at(best, s, f)
    return best


def categorise(rows, truth_n, owner_other):
    """Error class per row (see the module docstring); '' for correct rows."""
    s1, score = rows["s1"].to_numpy(), rows["meta_score"].to_numpy()
    lab, acc = rows["label"].to_numpy().astype(bool), rows["accepted"].to_numpy().astype(bool)
    n = len(rows)
    max_false = pd.Series(np.where(lab, -np.inf, score)).groupby(s1).transform("max").to_numpy()
    min_true = pd.Series(np.where(lab, score, np.inf)).groupby(s1).transform("min").to_numpy()
    has_true_present = pd.Series(lab.astype(int)).groupby(s1).transform("max").to_numpy() > 0
    cat = np.full(n, "", dtype=object)
    fn = lab & ~acc
    cat[fn & (score == -np.inf)] = "FN below the meta floor (first stage < 0.02)"
    rest = fn & (score > -np.inf)
    cat[rest & owner_other] = "FN lost to exclusivity"
    rest &= ~owner_other
    cat[rest & (max_false >= score)] = "FN outranked by a false candidate of the S1"
    cat[rest & (max_false < score)] = "FN correctly ranked but below the threshold"
    fp = acc & ~lab
    single = truth_n[s1] == 0
    cat[fp & single] = "FP in a singleton S1"
    rest = fp & ~single
    cat[rest & ~has_true_present] = "FP where the S1's true matches are all missing from candidates"
    rest &= has_true_present
    cat[rest & (score > min_true)] = "FP outranks a true candidate of the S1"
    cat[rest & (score <= min_true)] = "FP with all present trues ranked above it (threshold too low)"
    return cat


def run(rows_path=f"{OUT_DIR}/strict_rows_M3.parquet", out=f"{OUT_DIR}/oracle_gap.json", k100="output/candidates_p3/k100"):
    from src.evaluation.blocker_rescue import load_records
    from src.evaluation.hybrid_eval import HybridDev

    t0 = time.time()
    ctx = HybridDev(native=False)
    rows = pd.read_parquet(rows_path)
    s1_ids = ctx.s1.entity_id.to_numpy(dtype=object)
    n_s1 = len(s1_ids)
    rows["s1"] = pd.Index(s1_ids).get_indexer(rows["s1_id"])
    nt = np.array([len(ctx.truth[s]) for s in s1_ids])
    rnd = (ctx.s1.city == "").to_numpy()
    country = ctx.s1.country.to_numpy()
    s1, lab, acc = rows["s1"].to_numpy(), rows["label"].to_numpy().astype(bool), rows["accepted"].to_numpy()
    score, first = rows["meta_score"].to_numpy(), rows["first_prob"].to_numpy()

    # ---------------- ladder
    tp0 = np.bincount(s1, weights=lab & acc, minlength=n_s1)
    np0 = np.bincount(s1, weights=acc, minlength=n_s1)
    F0 = f05(tp0, np0, nt)
    F1 = best_prefix(s1, score, first, lab, n_s1, nt)
    F1_first = best_prefix(s1, first, first, lab, n_s1, nt)
    tp_present = np.bincount(s1, weights=lab, minlength=n_s1)
    F2 = f05(tp_present, tp_present, nt)
    k100_oof = pd.read_parquet(os.path.join(k100, "oof.parquet"), columns=["s1_id", "cand_id", "label"])
    union = pd.concat([rows[["s1_id", "cand_id", "label"]], k100_oof]).drop_duplicates(["s1_id", "cand_id"])
    tp3 = np.bincount(pd.Index(s1_ids).get_indexer(union["s1_id"]), weights=union["label"].astype(float),
                      minlength=n_s1)
    F3 = f05(tp3, tp3, nt)
    m = lambda v, mask=rnd: round(float(np.mean(v[mask])), 5)  # noqa: E731
    ladder = {"O0 current (M3 + meta, strictly nested)": m(F0),
              "O1 perfect cut on the current (meta) score order": m(F1),
              "O1' perfect cut on the first-stage (M3) probability order": m(F1_first),
              "O2 perfect selection among Hybrid50 candidates (ceiling)": m(F2),
              "O3 perfect selection among Hybrid50 + K=100 candidates": m(F3),
              "O4 ground truth": 1.0}
    decomposition = {"total loss 1 - O0": m(1 - F0), "blocking 1 - O2": m(1 - F2),
                     "ranking O2 - O1": m(F2 - F1), "decoder O1 - O0": m(F1 - F0)}

    # ---------------- pair-level error sources
    from src.matching.stream import s1_ranks
    from src.pipeline.meta_decoder import stage_winners

    rows["stage_winner"] = stage_winners(pd.factorize(rows["cand_id"])[0], score, s1_ranks(s1_ids)[s1],
                                         rows["origin"].to_numpy())
    accepted_by = rows.loc[acc, ["cand_id", "s1"]].drop_duplicates("cand_id").set_index("cand_id")["s1"]
    who = rows["cand_id"].map(accepted_by)
    owner_other = (who.notna() & (who.to_numpy() != s1)).to_numpy() | (~rows["stage_winner"].to_numpy())
    cat = categorise(rows, nt, owner_other)
    rows["category"] = cat
    missing = {s: len(ctx.truth[s] - ctx.in_cands.get(s, set())) for s in s1_ids}
    n_missing = np.array([missing[s] for s in s1_ids])

    def fixed(fn_add=None, fp_drop=None, add_missing=False):
        tp, npred = tp0.copy(), np0.copy()
        if fn_add is not None:
            tp += np.bincount(s1, weights=fn_add, minlength=n_s1)
            npred += np.bincount(s1, weights=fn_add, minlength=n_s1)
        if fp_drop is not None:
            npred -= np.bincount(s1, weights=fp_drop, minlength=n_s1)
        if add_missing:
            tp, npred = tp + n_missing, npred + n_missing
        return f05(tp, npred, nt)

    loss = 1 - F0
    table = []
    classes = ["FN missing from candidates (blocking)"] + sorted(set(cat) - {""})
    for c in classes:
        if c.startswith("FN missing"):
            aff = n_missing > 0
            Fc = fixed(add_missing=True)
            n_err = int(n_missing[rnd].sum())
        else:
            sel = cat == c
            aff = np.bincount(s1, weights=sel, minlength=n_s1) > 0
            Fc = fixed(fn_add=sel) if c.startswith("FN") else fixed(fp_drop=sel)
            n_err = int((sel & rnd[s1]).sum())
        table.append({"error source": c, "pairs": n_err, "S1 affected": int((aff & rnd).sum()),
                      "current F0.5 loss of affected S1 (macro share)": round(float(loss[aff & rnd].sum() / rnd.sum()), 5),
                      "max recoverable macro-F0.5": round(float((Fc - F0)[rnd].mean()), 5)})
    multi = (nt >= 2) & (tp0 > 0) & (tp0 < nt)
    Fm = fixed(fn_add=(lab & ~acc) & multi[s1])
    table.append({"error source": "S1-level: multi-match S1 with partial recovery (accept its present FNs)",
                  "pairs": int(((lab & ~acc) & multi[s1] & rnd[s1]).sum()), "S1 affected": int((multi & rnd).sum()),
                  "current F0.5 loss of affected S1 (macro share)": round(float(loss[multi & rnd].sum() / rnd.sum()), 5),
                  "max recoverable macro-F0.5": round(float((Fm - F0)[rnd].mean()), 5)})
    table.sort(key=lambda r: -r["max recoverable macro-F0.5"])

    # ---------------- splits
    bucket = np.minimum(nt, 4)
    splits = {}
    for name, mask_fn in (("country", lambda v: country == v), ("true matches (4 = 4+)", lambda v: bucket == v)):
        vals = sorted(set(country[rnd])) if name == "country" else [0, 1, 2, 3, 4]
        splits[name] = {str(v): {"S1": int((mask_fn(v) & rnd).sum()), "O0": m(F0, mask_fn(v) & rnd),
                                 "O1": m(F1, mask_fn(v) & rnd), "O2": m(F2, mask_fn(v) & rnd),
                                 "blocking": m(1 - F2, mask_fn(v) & rnd), "ranking": m(F2 - F1, mask_fn(v) & rnd),
                                 "decoder": m(F1 - F0, mask_fn(v) & rnd),
                                 "macro share of total loss": round(float(loss[mask_fn(v) & rnd].sum() / rnd.sum()), 5)}
                         for v in vals}
    err = rows[(cat != "") & rnd[s1]].copy()
    fn_missing_pairs = [(s, c) for s in np.array(s1_ids)[rnd] for c in ctx.truth[s] - ctx.in_cands.get(s, set())]
    recs = load_records(set(err["cand_id"]) | {c for _, c in fn_missing_pairs})
    from src.blocking.normalize import has_nonlatin
    native = {e for e, nme in zip(recs.entity_id, recs.business_name.fillna("")) if has_nonlatin(nme)}
    no_addr = set(recs.loc[recs.business_address.isna(), "entity_id"])
    err["native script candidate"] = err["cand_id"].isin(native)
    err["candidate address missing"] = err["cand_id"].isin(no_addr)
    err["blocker rank"] = pd.cut(np.where(err["origin"] == 1, 99, err["rank"]), [0, 1, 3, 10, 20, 1000],
                                 labels=["1", "2-3", "4-10", "11-20", "deep (K=50 only)"])
    by = {}
    for col in ("native script candidate", "candidate address missing", "blocker rank"):
        by[col] = {str(k): v for k, v in err.groupby(col, observed=True)["category"].value_counts().unstack(fill_value=0)
                   .to_dict(orient="index").items()}
    miss = pd.DataFrame(fn_missing_pairs, columns=["s1_id", "cand_id"])
    by["missing-from-candidates pairs"] = {"total": len(miss), "native script": int(miss.cand_id.isin(native).sum()),
                                           "candidate address missing": int(miss.cand_id.isin(no_addr).sum())}
    report = {"ladder": ladder, "decomposition": decomposition, "error_sources": table, "splits": splits,
              "error_class_by_trait": by, "seconds": round(time.time() - t0, 1)}
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
    rows[["s1_id", "cand_id", "origin", "rank", "label", "first_prob", "meta_score", "accepted", "category"]].to_parquet(
        f"{OUT_DIR}/oracle_gap_rows.parquet")  # git-ignored, for targeted follow-ups
    print(json.dumps({k: report[k] for k in ("ladder", "decomposition")}, indent=1))
    print(pd.DataFrame(table).to_string(index=False))
    print(json.dumps(splits, indent=1))
    print(json.dumps(by, indent=1))
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", default=f"{OUT_DIR}/strict_rows_M3.parquet")
    a = ap.parse_args(argv)
    run(a.rows)


if __name__ == "__main__":
    main()
