"""Production Hybrid50 inference -> the official outputs.

    python -m src.blocking.generate_candidates --split test --top-k 20 --out-dir output/candidates_k20 --no-tsv
    python -m src.blocking.generate_candidates --split test --top-k 50 --out-dir output/candidates_k50 --no-tsv
    python -m src.pipeline.train_meta              # once: the production bundle (development artifacts only)
    python -m src.pipeline.predict_hybrid          # -> output/matching_results.tsv + output/candidate_pairs.tsv

Default decoder "meta" (the adopted system, predict_hybrid_meta): both matchers of the bundle score their runs; a
joint pass over the paired K=20 / K=50 shards gives every Hybrid50 row of an S1 its first-stage probability and
features, the meta-model (src/pipeline/meta_decoder.py) re-scores the rows, and the decision is the one below with
the meta score and the bundle's two thresholds. Decoder "threshold" (predict_hybrid) is the earlier rule:

Stage 1  base: the improved K=20 matcher streams the exact K=20 candidates (src/matching/stream.py): global
         exclusivity, then its threshold. These assignments are frozen.
Stage 2  deep: the K=50 matcher streams the K=50 candidates. Only the hybrid's deep rows (K=50 candidates that are
         not in the S1's exact K=20 list, up to the hybrid cap; src/pipeline/hybrid.py) are scored and compete.
         A candidate goes to its global deep winner (highest probability, then the smallest s1_id, then the earliest
         row) when that probability is >= DEEP_THRESHOLD and the base did not already assign the candidate.
Stage 3  outputs, one row per S1 in test_source1 order:
         candidate_pairs.tsv   the hybrid candidate set: every exact K=20 candidate (base rank order), then the
                               K=50-only candidates (K=50 rank order) up to the cap
         matching_results.tsv  base + deep matches, sorted; every match is checked to be in its S1's candidate list
Same decisions as the offline src/pipeline/deep_recovery.py (base_predictions + merge_deep, rule "deep prob >= 0.85")
on the same probabilities; src/pipeline/hybrid_equivalence.py verifies that byte for byte on real smoke runs.
Both runs must come from the same split, S1 limit and shard plan (only top_k differs), so their shards pair up.
"""

import argparse
import heapq
import json
import os
import pickle
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.blocking.handoff import CandidateStore
from src.matching.extra_features import NameStats
from src.matching.matcher import check_top_k
from src.matching.matcher import from_store
from src.matching.stream import (Assignments, CandidateIndex, Winners, _init_role_worker, _ordered, _role_pass2_task,
                                 _scores_features, name_stats_file, pass1_tops, split_ids, stream_winners)
from src.pipeline.deep_recovery import DEEP_THRESHOLD
from src.pipeline.hybrid import hybrid_candidates
from src.pipeline.meta_decoder import META_VERSION, meta_scores

SHARD_COLS = ["s1_row", "rank", "candidate_entity_id"]


def pair_runs(base, deep):
    """{deep shard path: base shard path}; refuses runs whose shards do not cover the same S1 one to one."""
    for key in ("split", "encoding", "s1_limit"):
        if base.meta.get(key) != deep.meta.get(key):
            raise ValueError(f"base and deep candidate runs differ in {key}: {base.meta.get(key)} vs {deep.meta.get(key)}")
    if base.meta["config"]["shard_size"] != deep.meta["config"]["shard_size"]:
        raise ValueError("base and deep candidate runs use different shard sizes")
    rel_b = [os.path.relpath(p, base.run_dir) for p in base.shard_paths]
    rel_d = [os.path.relpath(p, deep.run_dir) for p in deep.shard_paths]
    if rel_b != rel_d or not rel_b:
        raise ValueError("base and deep candidate runs have different (or no) shards")
    return dict(zip(deep.shard_paths, base.shard_paths))


def read_shard(path):
    return pq.read_table(path, columns=SHARD_COLS).to_pandas()


def shard_hybrid(base_df, deep_df, cap):
    """hybrid_candidates() of one aligned shard pair, keyed by s1_row; both shards must hold the same S1 rows."""
    if not np.array_equal(np.unique(base_df["s1_row"]), np.unique(deep_df["s1_row"])):
        raise ValueError("paired base / deep shards hold different S1 rows")
    return hybrid_candidates(base_df[SHARD_COLS], deep_df[SHARD_COLS], cap=cap, s1="s1_row",
                             cand="candidate_entity_id", rank="rank")


class DeepRows:
    """select= callable for stream_winners() over the deep run: the hybrid's deep rows of each deep shard."""

    def __init__(self, pairing, cap):
        self.pairing, self.cap = pairing, cap

    def __call__(self, path, df):
        h = shard_hybrid(read_shard(self.pairing[path]), df, self.cap)
        d = h[h["origin"] == "deep"]
        rows = pd.MultiIndex.from_arrays([df["s1_row"].to_numpy(), df["candidate_entity_id"].to_numpy()])
        return rows.isin(pd.MultiIndex.from_arrays([d["s1_row"].to_numpy(), d["candidate_entity_id"].to_numpy()]))


def iter_hybrid_lists(pairing, cap, n_s1):
    """(s1_row, [candidate ids in hybrid order]) for S1 rows 0 .. n_s1-1 in file order ([] = no candidate).
    Shards are grouped per country (each country's shards hold ascending S1 rows) and merged on s1_row, so only one
    shard pair per country is in memory."""
    by_country = {}
    for deep_path, base_path in sorted(pairing.items()):
        by_country.setdefault(os.path.dirname(deep_path), []).append((base_path, deep_path))

    def country_stream(shards):
        for base_path, deep_path in shards:
            h = shard_hybrid(read_shard(base_path), read_shard(deep_path), cap)
            rows, ids = h["s1_row"].to_numpy(), h["candidate_entity_id"].to_numpy()
            starts = np.flatnonzero(np.r_[True, rows[1:] != rows[:-1]]) if len(rows) else np.zeros(0, int)
            for a, b in zip(starts, np.r_[starts[1:], len(rows)]):
                yield int(rows[a]), ids[a:b].tolist()

    merged = heapq.merge(*(country_stream(v) for v in by_country.values()), key=lambda t: t[0])
    nxt = next(merged, None)
    for r in range(n_s1):
        cands = []
        if nxt is not None and nxt[0] == r:
            cands, nxt = nxt[1], next(merged, None)
        if nxt is not None and nxt[0] <= r:
            raise ValueError(f"hybrid candidates are not in S1 file order at S1 row {r}")
        yield r, cands
    if nxt is not None:
        raise ValueError(f"candidates for S1 row {nxt[0]} beyond the {n_s1} S1 of the run")


def hybrid_assignments(base, base_threshold, deep, deep_threshold):
    """Final matches from the two global Winners states -> (candidate keys, owner S1 ranks, is_deep).
    Base winners that clear the base threshold are kept as they are; a deep winner is added only when it clears the
    deep threshold and the base did not assign that candidate to any S1 (base priority)."""
    b = base.accepted(base_threshold)
    d = deep.accepted(deep_threshold)
    d = d[~np.isin(d, b)]
    return (np.concatenate([b, d]), np.concatenate([base.s1[b], deep.s1[d]]),
            np.r_[np.zeros(len(b), dtype=bool), np.ones(len(d), dtype=bool)])


def write_outputs(matching_path, candidate_path, assignments, hybrid_lists, s1_ids, s1_rank):
    """Both official TSVs in one pass over the S1 (file order); checks matches are a subset of the candidates."""
    st = {"s1": 0, "s1_matched": 0, "matches": 0, "candidates": 0, "empty_candidate_rows": 0}
    for path in (matching_path, candidate_path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(matching_path, "w", encoding="utf-8", newline="") as fm, \
            open(candidate_path, "w", encoding="utf-8", newline="") as fc:
        fm.write("source1_entity_id\tmatched_entity_ids\n")
        fc.write("source1_entity_id\tcandidate_entity_ids\n")
        for i, ((row, cands), s, r) in enumerate(zip(hybrid_lists, s1_ids, s1_rank)):
            if row != i:
                raise ValueError(f"candidate list for S1 row {row} where row {i} was expected")
            if len(set(cands)) != len(cands):
                raise ValueError(f"{s}: duplicate candidates")
            m = assignments.of(r)
            if len(m) and not set(m) <= set(cands):
                raise ValueError(f"{s}: a match is not among its candidates")
            fm.write(f"{s}\t{','.join(m)}\n")
            fc.write(f"{s}\t{','.join(cands)}\n")
            st["s1"] += 1
            st["s1_matched"] += len(m) > 0
            st["matches"] += len(m)
            st["candidates"] += len(cands)
            st["empty_candidate_rows"] += not cands
    if st["matches"] != len(assignments.owner):
        raise ValueError("matches assigned to S1 outside the run")
    return st


def predict_hybrid(base_store, deep_store, base_model, deep_model, matching_path, candidate_path, cap=50,
                   deep_threshold=DEEP_THRESHOLD, workers=2, log=print):
    t0 = time.time()
    pairing = pair_runs(base_store, deep_store)
    check_top_k(base_model["top_k"], base_store.meta["config"]["top_k"], base_store.split)
    check_top_k(deep_model["top_k"], deep_store.meta["config"]["top_k"], deep_store.split)
    tables = base_store.source_tables()
    groups = set(base_model.get("feature_groups", ())) | set(deep_model.get("feature_groups", ()))
    name_stats = NameStats.from_tables(tables) if "E" in groups else None  # the split's own statistics, once

    log("stage 1: base (exact K=%d candidates)" % base_store.meta["config"]["top_k"])
    base_w, n_base, _ = stream_winners(base_store, base_model, log, workers, name_stats=name_stats)
    log("stage 2: deep (K=%d-only candidates, cap %d)" % (deep_store.meta["config"]["top_k"], cap))
    deep_w, n_deep, n_deep_scored = stream_winners(deep_store, deep_model, log, workers,
                                                   select=DeepRows(pairing, cap), name_stats=name_stats)
    keys, owner, is_deep = hybrid_assignments(base_w, base_model["threshold"], deep_w, deep_threshold)

    log("stage 3: outputs")
    cand_ids, s1_all, rank = split_ids(tables)
    n = base_store.meta["s1_limit"] or len(s1_all)
    st = write_outputs(matching_path, candidate_path, Assignments(owner, cand_ids[keys]),
                       iter_hybrid_lists(pairing, cap, n), s1_all[:n], rank[:n])
    return {**st, "base_pairs": n_base, "deep_pairs": n_deep, "deep_rows_scored": n_deep_scored,
            "base_matches": int((~is_deep).sum()), "deep_matches": int(is_deep.sum()),
            "base_threshold": base_model["threshold"], "deep_threshold": deep_threshold, "cap": cap,
            "seconds": round(time.time() - t0, 1)}


def predict_hybrid_meta(base_store, deep_store, bundle, matching_path, candidate_path, workers=2, log=print):
    """The adopted system: M3 first stage + meta-model decoder on the Hybrid50 rows (see the module docstring)."""
    from concurrent.futures import ProcessPoolExecutor

    t0 = time.time()
    if bundle.get("meta_version") != META_VERSION:
        raise ValueError(f"bundle version {bundle.get('meta_version')} != {META_VERSION}; re-run train_meta")
    models = {"base": bundle["base_matcher"], "deep": bundle["deep_matcher"]}
    cap, floor, extra = bundle["hybrid"]["cap"], bundle["meta_floor"], tuple(bundle.get("meta_extra", ()))
    t_base, t_deep = bundle["thresholds"]["base"], bundle["thresholds"]["deep"]
    pairing = pair_runs(base_store, deep_store)
    check_top_k(models["base"]["top_k"], base_store.meta["config"]["top_k"], base_store.split)
    check_top_k(models["deep"]["top_k"], deep_store.meta["config"]["top_k"], deep_store.split)
    tables = base_store.source_tables()
    index = CandidateIndex(len(tables[2]), len(tables[3]))
    cand_ids, s1_all, rank = split_ids(tables)
    groups = set(models["base"].get("feature_groups", ())) | set(models["deep"].get("feature_groups", ()))
    name_stats = NameStats.from_tables(tables) if "E" in groups else None
    executor, ns_path = None, None
    if workers > 1:
        ns_path = name_stats_file(name_stats)
        executor = ProcessPoolExecutor(max_workers=workers, initializer=_init_role_worker, initargs=(models, ns_path))
    window = 2 * workers
    try:
        tops = {}
        for role, store in (("base", base_store), ("deep", deep_store)):
            log(f"pass 1: {role} run (K={store.meta['config']['top_k']})")
            tops[role] = pass1_tops(store, models[role], index, executor, window, log,
                                    role=role if executor is not None else None)
        select = DeepRows(pairing, cap)

        def tasks():
            it = zip(base_store.shard_paths, deep_store.shard_paths, base_store.iter_frames(with_records=True),
                     deep_store.iter_frames(with_records=True))
            for base_path, deep_path, dfb, dfd in it:
                if pairing[deep_path] != base_path:
                    raise ValueError("base and deep shards are not aligned")
                for role, df, path in (("base", dfb, base_path), ("deep", dfd, deep_path)):
                    keys = index.keys(df["candidate_source"], df["candidate_row"])
                    rows = None if role == "base" else np.asarray(select(path, df), dtype=bool)
                    pairs, records = from_store(df)
                    xt = {c: t.frame(keys, pairs["cand_id"]) for c, t in tops[role].items()}
                    meta = (role, keys, df["s1_row"].to_numpy(), df["rank"].to_numpy(), rows)
                    yield meta, (role, pairs, records, xt, rows, floor, extra)

        if executor is None:
            results = ((m, _scores_features(models[m[0]], pr, rec, xt, name_stats, rows, fl, ex))
                       for m, (_, pr, rec, xt, rows, fl, ex) in tasks())
        else:
            results = _ordered(executor, _role_pass2_task, tasks(), window)
        log("pass 2: paired shards -> first stage -> meta-model")
        wins = {0: Winners(index.n, np.float64), 1: Winners(index.n, np.float64)}
        row0, n = 0, {"base_pairs": 0, "deep_pairs": 0, "deep_rows_scored": 0, "meta_rows": 0}
        it = iter(results)
        for i, ((mb, (pb, _, xb)), (md, (pd_, _, xd))) in enumerate(zip(it, it)):
            if mb[0] != "base" or md[0] != "deep":
                raise RuntimeError("paired results out of order")
            rows = md[4]
            n["base_pairs"] += len(mb[1])
            n["deep_pairs"] += len(md[1])
            n["deep_rows_scored"] += int(rows.sum())
            keys = np.concatenate([mb[1], md[1][rows]])
            s1 = np.concatenate([mb[2], md[2][rows]])
            rk = np.concatenate([mb[3], md[3][rows]])
            prob = np.concatenate([pb, pd_]).astype(np.float64)
            origin = np.r_[np.zeros(len(pb), dtype=np.int64), np.ones(len(pd_), dtype=np.int64)]
            score = meta_scores(bundle["meta_model"], prob, origin, rk, s1, np.vstack([xb, xd]),
                                bundle["pair_features"], floor)
            at = row0 + np.arange(len(prob))
            row0 += len(prob)
            ok = score > -np.inf
            n["meta_rows"] += int(ok.sum())
            for o in (0, 1):
                m = ok & (origin == o)
                wins[o].update(keys[m], score[m], rank[s1[m]], at[m])
            if i % 100 == 0:
                log(f"  pass 2: shard pair {i}/{len(pairing)}")
    finally:
        if executor is not None:
            executor.shutdown()
        if ns_path:
            os.remove(ns_path)
    keys, owner, is_deep = hybrid_assignments(wins[0], t_base, wins[1], t_deep)
    log("outputs")
    n_s1 = base_store.meta["s1_limit"] or len(s1_all)
    st = write_outputs(matching_path, candidate_path, Assignments(owner, cand_ids[keys]),
                       iter_hybrid_lists(pairing, cap, n_s1), s1_all[:n_s1], rank[:n_s1])
    return {**st, **n, "base_matches": int((~is_deep).sum()), "deep_matches": int(is_deep.sum()),
            "decoder": "meta", "thresholds": bundle["thresholds"], "cap": cap, "seconds": round(time.time() - t0, 1)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="test")
    ap.add_argument("--base-cands", default="output/candidates_k20", help="out_dir of the K=20 generate_candidates run")
    ap.add_argument("--deep-cands", default="output/candidates_k50", help="out_dir of the K=50 generate_candidates run")
    ap.add_argument("--base-model", default="output/candidates_p3/k20/matcher.pkl")
    ap.add_argument("--deep-model", default="output/candidates_p3/k50/matcher.pkl")
    ap.add_argument("--cap", type=int, default=50)
    ap.add_argument("--deep-threshold", type=float, default=DEEP_THRESHOLD)
    ap.add_argument("--workers", type=int, default=2, help="scoring processes; output identical for any value")
    ap.add_argument("--out", default="output")
    ap.add_argument("--decoder", choices=["meta", "threshold"], default="meta",
                    help="meta: the adopted M3 + meta-model system (needs --bundle); threshold: the earlier rule")
    ap.add_argument("--bundle", default="output/candidates_p3/hybrid_meta.pkl", help="written by train_meta")
    a = ap.parse_args(argv)
    t0 = time.time()
    log = lambda msg: print(f"{msg}, {time.time() - t0:.0f}s", flush=True)  # noqa: E731
    base = CandidateStore(a.split, out_dir=a.base_cands)
    deep = CandidateStore(a.split, out_dir=a.deep_cands, data_dir=base.data_dir, tables=base.source_tables())
    paths = os.path.join(a.out, "matching_results.tsv"), os.path.join(a.out, "candidate_pairs.tsv")
    if a.decoder == "meta":
        with open(a.bundle, "rb") as fh:
            bundle = pickle.load(fh)
        r = predict_hybrid_meta(base, deep, bundle, *paths, workers=a.workers, log=log)
    else:
        models = []
        for path in (a.base_model, a.deep_model):
            with open(path, "rb") as fh:
                models.append(pickle.load(fh))
        r = predict_hybrid(base, deep, *models, *paths, cap=a.cap, deep_threshold=a.deep_threshold,
                           workers=a.workers, log=log)
    with open(os.path.join(a.out, "hybrid_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(r, fh, indent=2)
    print(json.dumps(r))


if __name__ == "__main__":
    main()
