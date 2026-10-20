"""Checkpoint E: can targeted rescue lift the Hybrid50 candidate ceiling above 0.990 / 0.993 / 0.995? (dev only)

    python -m src.evaluation.candidate_ceiling

Oracle retrieval, no matcher: four channels built from P2's own encoded fields and vocabulary rules
(src/evaluation/blocker_rescue.py: name trigrams, rare address tokens, rare name tokens, phonetic keys) each give
every development S1 its best 20 candidates that are NOT already in its Hybrid50 list. The ceiling (perfect
selection among the candidates) is measured for +5 / +10 / +20 per channel and for all channels together, on the
random development S1. Also: for every Hybrid50 FN1 pair (a true pair missing from the candidates), which channel,
if any, retrieves it at depth 20, and the traits of the pairs no channel reaches. Never the final holdout.
"""

import argparse
import json
import time

import numpy as np
import pandas as pd

from src.evaluation.blocker_rescue import CHANNELS, OUT_DIR, rescue_candidates
from src.evaluation.hybrid_eval import HybridDev

DEPTH = 20


def run(out=f"{OUT_DIR}/candidate_ceiling.json", log=print):
    from src.pipeline.hybrid import blocking_metrics

    t0 = time.time()
    ctx = HybridDev(native=False)
    country = dict(zip(ctx.s1.entity_id, ctx.s1.country))
    rnd = ctx.rnd
    rnd_set = set(rnd)
    base = ctx.hybrid[["s1_id", "cand_id"]]
    chans, seconds = {}, {}
    for c in CHANNELS:
        t1 = time.time()
        chans[c] = rescue_candidates(ctx, c, depth=DEPTH)
        chans[c]["label"] = [x in ctx.truth.get(s, ()) for s, x in zip(chans[c].s1_id, chans[c].cand_id)]
        seconds[c] = round(time.time() - t1, 1)
        log(f"  {c}: {len(chans[c])} rescue rows, {seconds[c]}s")
    res = {"hybrid50": blocking_metrics(ctx.hybrid, ctx.truth, country, rnd), "retrieval_seconds": seconds}

    def measure(add):
        add = add.drop_duplicates(["s1_id", "cand_id"])
        met = blocking_metrics(pd.concat([base, add[["s1_id", "cand_id"]]], ignore_index=True), ctx.truth,
                               country, rnd)
        a_r = add[add.s1_id.isin(rnd_set)]
        per = a_r.groupby("s1_id").size().reindex(rnd).fillna(0)
        met.update(added_per_s1_mean=round(float(per.mean()), 2), added_per_s1_p95=float(per.quantile(0.95)),
                   added_per_s1_max=int(per.max()), extra_true=int(a_r.label.sum()),
                   extra_false=int((~a_r.label).sum()))
        return met

    for n in (5, 10, 20):
        for c, f in chans.items():
            res[f"{c} +{n}"] = measure(f[f.rescue_rank <= n])
        res[f"all channels +{n} each"] = measure(pd.concat([f[f.rescue_rank <= n] for f in chans.values()]))
        log(f"  +{n}: all channels ceiling {res[f'all channels +{n} each']['ceiling']}")
    # which FN1 pairs does any channel reach at depth 20, and what do the unreachable ones look like
    fn1 = pd.read_parquet(f"{OUT_DIR}/blocker_fn1_pairs.parquet")
    fn1 = fn1[(fn1.reach == "FN1") & fn1.random].copy()
    for c, f in chans.items():
        got = set(zip(f.s1_id[f.label], f.cand_id[f.label]))
        fn1[f"by {c}"] = [(s, x) in got for s, x in zip(fn1.s1_id, fn1.cand_id)]
    by = [f"by {c}" for c in chans]
    fn1["reached"] = fn1[by].any(axis=1)
    traits = ["India", "native script candidate", "missing address (either side)", "no shared name token",
              "address-only evidence (no name token, address >= 0.6)", "house-number conflict (both numbered, none shared)",
              "alias / rebrand (different name, same address)"]
    res["fn1_random_s1"] = {"pairs": int(len(fn1)), "reached_by_any_channel_depth20": int(fn1.reached.sum()),
                            **{k: int(fn1[k].sum()) for k in by},
                            "unreached_traits": {t: round(float(fn1.loc[~fn1.reached, t].mean()), 3) for t in traits},
                            "reached_traits": {t: round(float(fn1.loc[fn1.reached, t].mean()), 3) for t in traits}}
    best = max((k for k in res if "+" in k), key=lambda k: res[k]["ceiling"])
    res["answer"] = {"best_rescue": best, "best_ceiling": res[best]["ceiling"],
                     "added_per_s1_mean": res[best]["added_per_s1_mean"],
                     **{f"ceiling_above_{t}": bool(res[best]["ceiling"] > t) for t in (0.990, 0.993, 0.995)}}
    res["seconds"] = round(time.time() - t0, 1)
    print(json.dumps({k: res[k] for k in ("hybrid50", "answer", "fn1_random_s1")}, indent=1))
    print(pd.DataFrame({k: v for k, v in res.items() if "+" in k}).T[
        ["pair_recall", "ceiling", "added_per_s1_mean", "added_per_s1_p95", "extra_true", "extra_false"]].to_string())
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args(argv)
    run()


if __name__ == "__main__":
    main()
