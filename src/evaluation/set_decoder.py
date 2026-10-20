"""Checkpoint C: per-S1 set decoder on the strictly nested meta scores (development only).

    python -m src.evaluation.set_decoder

The oracle-gap audit (src/evaluation/oracle_gap.py) found the largest recoverable loss is WHERE each S1's ranked
candidate list is cut: correctly ranked true candidates below the threshold, and false ones accepted after all
trues. A fixed threshold ignores the S1's own probability profile, while the F0.5 of an S1 depends on its whole set.

Expected-F0.5 decoder: for every S1, rows sorted by the current score; with calibrated match probabilities p and
independent matches, the number of true matches inside the first k rows and after them are Poisson-binomial. The
decoder picks the k (0 = empty) that maximises
    E[F0.5(k)] = sum_a sum_b P(TP_prefix = a) P(TP_rest = b) * 1.25 a / (0.25 (a + b + miss) + k)
(k = 0: P(no true match at all)), exactly by dynamic programming, vectorised over all S1. The prefix rows that are
stage winners are accepted (deep rows only if the base did not take the candidate), as in production.
Calibration per outer fold: isotonic regression of meta score -> label fitted on the other four folds only.
"""

import argparse
import json
import time

import numpy as np
import pandas as pd

from src.evaluation.hybrid_eval import bootstrap_ci
from src.evaluation.oracle_gap import OUT_DIR, f05

K_MAX, T_MAX = 16, 24  # cut points examined; successes tracked (the rest is folded into the last bucket)


def pb_step(dist, p):
    """One Bernoulli(p) added to per-row count distributions dist (n x T), mass beyond T_MAX kept in the last cell."""
    out = dist * (1 - p)[:, None]
    shifted = dist * p[:, None]
    out[:, 1:] += shifted[:, :-1]
    out[:, -1] += shifted[:, -1]
    return out


def expected_f05(P, miss=None):
    """E[F0.5] of cutting after k = 0..K_MAX rows, for rows P (n_s1 x L, sorted by score, padded with 0)."""
    n, L = P.shape
    T = T_MAX + 1
    pre = np.zeros((K_MAX + 1, n, T))
    d = np.zeros((n, T))
    d[:, 0] = 1
    pre[0] = d
    for k in range(1, K_MAX + 1):
        d = pb_step(d, P[:, k - 1]) if k - 1 < L else d
        pre[k] = d
    suf = np.zeros((K_MAX + 1, n, T))
    d = np.zeros((n, T))
    d[:, 0] = 1
    for j in range(L - 1, -1, -1):  # d = distribution of rows j .. L-1
        d = pb_step(d, P[:, j])
        if j <= K_MAX:
            suf[j] = d
    for j in range(L, K_MAX + 1):  # fewer rows than cut points: empty rest
        suf[j][:, :] = 0
        suf[j][:, 0] = 1
    a = np.arange(T)[:, None]
    b = np.arange(T)[None, :]
    m = 0.0 if miss is None else miss[:, None, None]
    E = np.zeros((n, K_MAX + 1))
    E[:, 0] = suf[0][:, 0] * (1.0 if miss is None else np.exp(-miss))  # nothing predicted: F = 1 iff no match
    for k in range(1, K_MAX + 1):
        W = 1.25 * a / (0.25 * (a + b + m) + k)  # (n or 1) x T x T
        W = np.where(a <= k, W, 0.0)
        E[:, k] = np.einsum("na,nb,nab->n", pre[k], suf[k], np.broadcast_to(W, (n, T, T)))
    return E


def padded(rows, p):
    """(S1 index per S1 row block, P matrix sorted by score) from rows with column s1 (codes) and 'order'."""
    order = np.lexsort((-rows["first_prob"].to_numpy(), -rows["score_sort"].to_numpy(), rows["s1"].to_numpy()))
    s = rows["s1"].to_numpy()[order]
    pos = pd.Series(np.ones(len(s))).groupby(s).cumsum().to_numpy().astype(int) - 1
    uniq, inv = np.unique(s, return_inverse=True)
    P = np.zeros((len(uniq), pos.max() + 1))
    P[inv, pos] = p[order]
    return uniq, P, order, inv, pos


def decode(rows, p, s1_rank, miss=None):
    """Accepted rows of the expected-F0.5 decoder (see the module docstring)."""
    from src.pipeline.meta_decoder import stage_winners

    uniq, P, order, inv, pos = padded(rows, p)
    E = expected_f05(P, miss if miss is None else miss[uniq])
    k_star = E.argmax(axis=1)
    in_prefix = np.zeros(len(rows), dtype=bool)
    in_prefix[order] = pos < k_star[inv]
    cand = pd.factorize(rows["cand_id"])[0]
    origin = rows["origin"].to_numpy()
    win = stage_winners(cand, rows["score_sort"].to_numpy(), s1_rank, origin)
    ok = in_prefix & win
    acc_b = ok & (origin == 0)
    return acc_b | (ok & (origin == 1) & ~np.isin(cand, cand[acc_b])), k_star


def run(rows_path=f"{OUT_DIR}/strict_rows_M3.parquet", out=f"{OUT_DIR}/set_decoder.json"):
    from sklearn.isotonic import IsotonicRegression

    from src.evaluation.hybrid_eval import HybridDev
    from src.matching.stream import s1_ranks

    t0 = time.time()
    ctx = HybridDev(native=False)
    s1_ids = ctx.s1.entity_id.to_numpy(dtype=object)
    n_s1 = len(s1_ids)
    rows = pd.read_parquet(rows_path)
    rows["s1"] = pd.Index(s1_ids).get_indexer(rows["s1_id"])
    s1 = rows["s1"].to_numpy()
    lab = rows["label"].to_numpy().astype(bool)
    score = rows["meta_score"].to_numpy()
    first = rows["first_prob"].to_numpy()
    fold = rows["fold"].to_numpy()
    rows["score_sort"] = np.where(np.isfinite(score), score, -1.0 + first)  # below the floor: after, by first prob
    nt = np.array([len(ctx.truth[s]) for s in s1_ids])
    rnd = (ctx.s1.city == "").to_numpy()
    country = ctx.s1.country.to_numpy()
    s1_rank = s1_ranks(s1_ids)[s1]
    acc0 = rows["accepted"].to_numpy().astype(bool)

    def per_s1(acc):
        return f05(np.bincount(s1, weights=lab & acc, minlength=n_s1), np.bincount(s1, weights=acc, minlength=n_s1), nt)

    ref = per_s1(acc0)

    def summary(acc):
        per = per_s1(acc)
        d = (per - ref)[rnd]
        lo, hi = bootstrap_ci(d)
        rows_r = rnd[s1]
        return {"f05": round(float(per[rnd].mean()), 5), "delta": round(float(d.mean()), 5),
                "delta_se": round(float(d.std(ddof=1) / np.sqrt(len(d))), 5), "delta_ci95": [round(lo, 5), round(hi, 5)],
                "India": round(float(per[rnd & (country == "India")].mean()), 4),
                "US": round(float(per[rnd & (country == "US")].mean()), 4),
                "delta_India": round(float((per - ref)[rnd & (country == "India")].mean()), 5),
                "delta_US": round(float((per - ref)[rnd & (country == "US")].mean()), 5),
                "singletons_empty": round(float((np.bincount(s1, weights=acc, minlength=n_s1)[rnd & (nt == 0)] == 0)
                                                .mean()), 4),
                "fp": int((acc & ~lab & rows_r).sum()), "tp": int((acc & lab & rows_r).sum()),
                "s1_improved": int((d > 1e-12).sum()), "s1_harmed": int((d < -1e-12).sum())}

    res = {"current (M3 + meta, strict)": summary(acc0)}
    # calibrated probabilities, cross-fitted: the map for fold k is fitted on the other folds' rows only
    finite = np.isfinite(score)
    p_cal = first.copy()
    for k in range(5):
        tr, va = finite & (fold != k), finite & (fold == k)
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(score[tr], lab[tr])
        p_cal[va] = iso.predict(score[va])
    p_raw = np.where(finite, score, first)
    variants = {"expected-F0.5 decoder, raw meta scores": (p_raw, None),
                "expected-F0.5 decoder, calibrated (isotonic per fold)": (p_cal, None)}
    # blocking misses the decoder cannot see: expected missing trues per S1 = rate x expected trues present,
    # the rate estimated on the training folds (cross-fitted)
    miss = np.zeros(n_s1)
    exp_present = np.bincount(s1, weights=p_cal, minlength=n_s1)
    s1_fold = np.full(n_s1, -1)
    s1_fold[s1] = fold
    present = np.bincount(s1, weights=lab, minlength=n_s1)
    for k in range(5):
        tr = s1_fold != k
        rate = (nt[tr].sum() - present[tr].sum()) / max(present[tr].sum(), 1)
        miss[s1_fold == k] = rate * exp_present[s1_fold == k]
    variants["expected-F0.5 decoder, calibrated + expected blocking misses"] = (p_cal, miss)
    for name, (p, ms) in variants.items():
        t1 = time.time()
        acc, k_star = decode(rows, p, s1_rank, ms)
        r = summary(acc)
        r["seconds"] = round(time.time() - t1, 1)
        res[name] = r
        print(f"{name}: {json.dumps(r)}", flush=True)
    res["seconds"] = round(time.time() - t0, 1)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args(argv)
    run()


if __name__ == "__main__":
    main()
