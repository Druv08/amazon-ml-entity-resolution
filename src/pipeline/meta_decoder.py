"""Second-stage (meta) decoder of the adopted system: M3 first stage + meta-model on the Hybrid50 rows.

One row per Hybrid50 candidate: every exact K=20 candidate (origin 0, K=20 matcher probability) and every deep
candidate (origin 1, K=50 matcher probability). For the rows whose first-stage probability is >= META_FLOOR the
meta-model sees
    prob, origin, blocker rank                                    (the row)
    S1 context over ALL of the S1's Hybrid50 rows (CONTEXT)       (top-3 probabilities, counts above levels, ...)
    the first-stage matcher's pair features, without 'rank'       (the row's own features)
and its score replaces the probability in the production decision:
    base  the K=20 candidate's best-scoring claimant (ties: smallest s1_id) is accepted if score >= t_base
    deep  the deep candidate's best-scoring claimant is accepted if score >= t_deep and the base did not take it
Rows below the floor are never accepted. The same functions build the training matrix (train_meta.py), the offline
reference (hybrid_equivalence.py) and the streaming inference (predict_hybrid.py). Development evidence:
docs/p3_matching.md "Structured decisions" (strictly nested 0.9574).
"""

import numpy as np
import pandas as pd

META_VERSION = "meta-v1"
META_FLOOR = 0.02
LEVELS = (0.3, 0.5, 0.7, 0.9)
CONTEXT = (["s1_top1", "s1_top2", "s1_top3"] + [f"s1_n_above_{t}" for t in LEVELS]
           + ["s1_prob_sum", "s1_base_top1", "s1_deep_top1", "s1_n_cands", "prob_pos_in_s1", "top1_minus_prob",
              "gap12"])


def context_features(s1, prob, origin):
    """Per-row S1 context (CONTEXT columns, float64) over all rows of each S1. ``s1`` are any per-S1 codes; ties in
    the probability ranking keep the row order (base rows before deep rows, each in rank order)."""
    prob = np.asarray(prob, dtype=np.float64)
    f = pd.DataFrame({"s1": np.asarray(s1), "prob": prob, "origin": np.asarray(origin)})
    f = f.sort_values(["s1", "prob"], ascending=[True, False], kind="mergesort")
    f["pos"] = f.groupby("s1").cumcount()
    uniq = np.unique(f["s1"].to_numpy())
    top = f[f["pos"] < 3].pivot(index="s1", columns="pos", values="prob").reindex(index=uniq, columns=[0, 1, 2])
    top = top.fillna(0.0)
    per = pd.DataFrame({"s1_top1": top[0], "s1_top2": top[1], "s1_top3": top[2]})
    g = f.groupby("s1")["prob"]
    for t in LEVELS:
        per[f"s1_n_above_{t}"] = f.assign(x=f["prob"] >= t).groupby("s1")["x"].sum().reindex(per.index).fillna(0)
    per["s1_prob_sum"] = g.sum().reindex(per.index).fillna(0)
    per["s1_base_top1"] = f[f.origin == 0].groupby("s1")["prob"].max().reindex(per.index).fillna(0)
    per["s1_deep_top1"] = f[f.origin == 1].groupby("s1")["prob"].max().reindex(per.index).fillna(0)
    per["s1_n_cands"] = g.size().reindex(per.index).fillna(0)
    rows = per.iloc[np.searchsorted(uniq, np.asarray(s1))].reset_index(drop=True)
    rows["prob_pos_in_s1"] = f["pos"].reindex(range(len(prob))).to_numpy()  # 0 = the S1's best row
    rows["top1_minus_prob"] = rows["s1_top1"] - prob
    rows["gap12"] = rows["s1_top1"] - rows["s1_top2"]
    return rows[CONTEXT]


def pair_feature_names(first_stage_features):
    return [c for c in first_stage_features if c != "rank"]  # the blocker rank enters as "blocker_rank"


def meta_matrix(prob, origin, rank, context, pair, pair_names):
    """The meta-model's input (float32 DataFrame) for already selected rows: context = context_features() rows,
    pair = first-stage feature rows (in the first-stage feature order, i.e. with 'rank')."""
    cols = {"prob": np.asarray(prob, dtype=np.float64), "origin": np.asarray(origin),
            "blocker_rank": np.asarray(rank)}
    M = pd.concat([pd.DataFrame(cols), context.reset_index(drop=True)], axis=1)
    P = pd.DataFrame(np.asarray(pair, dtype=np.float32), columns=list(pair_names))
    keep = [c for c in P.columns if c != "rank"]
    return pd.concat([M, P[keep]], axis=1).astype(np.float32)


def meta_scores(model, prob, origin, rank, s1, pair_sel, pair_names, floor=META_FLOOR):
    """Meta scores (float64; -inf below the floor) of a block of Hybrid50 rows that holds ALL rows of its S1.
    pair_sel: first-stage feature rows of exactly the rows with float64(prob) >= floor, in row order."""
    prob = np.asarray(prob, dtype=np.float64)
    score = np.full(len(prob), -np.inf)
    sel = np.flatnonzero(prob >= floor)
    if len(pair_sel) != len(sel):
        raise ValueError(f"{len(pair_sel)} feature rows for {len(sel)} rows above the meta floor")
    if len(sel):
        ctx = context_features(s1, prob, origin)
        M = meta_matrix(prob[sel], np.asarray(origin)[sel], np.asarray(rank)[sel], ctx.iloc[sel], pair_sel,
                        pair_names)
        score[sel] = model.predict_proba(M)[:, 1]
    return score


def stage_winners(cand, score, s1_rank, origin):
    """Rows that are their candidate's best claimant within their stage (score desc, then smallest s1_id)."""
    cand, score, s1_rank, origin = map(np.asarray, (cand, score, s1_rank, origin))
    win = np.zeros(len(score), dtype=bool)
    for o in (0, 1):
        idx = np.flatnonzero(origin == o)
        order = idx[np.lexsort((s1_rank[idx], -score[idx], cand[idx]))]
        c = cand[order]
        win[order[np.r_[True, c[1:] != c[:-1]]]] = True
    return win


def decide(cand, origin, score, s1_rank, t_base, t_deep):
    """Accepted rows of the production decision (see the module docstring)."""
    # integer codes: np.isin on object (string) arrays falls back to a Python loop, O(#accepted x #rows)
    cand = pd.factorize(np.asarray(cand))[0]
    origin, score = np.asarray(origin), np.asarray(score)
    ok = stage_winners(cand, score, s1_rank, origin) & (score >= np.where(origin == 0, t_base, t_deep))
    acc_b = ok & (origin == 0)
    taken = np.zeros(cand.max() + 1 if len(cand) else 0, dtype=bool)
    taken[cand[acc_b]] = True
    return acc_b | (ok & (origin == 1) & ~taken[cand])
