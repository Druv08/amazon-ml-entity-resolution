"""Bounded-memory matcher inference over a P2 CandidateStore (used by predict.py).

Same decisions as scoring every pair in memory and applying matcher.exclusive() + to_matches(), but the memory is
O(#S2+S3 records) + one shard instead of O(#pairs):

  pass 1  per shard: cross-entity scores -> TopTwo (each candidate's two best scores over ALL S1, = global top2())
  pass 2  per shard: features (with the global top-2) -> model probability -> Winners (global exclusivity state)
  emit    candidates whose global winner has prob >= threshold, grouped by S1, one row per S1 in file order

Candidates are keyed by P2's stable (candidate_source, candidate_row) columns, S1 by s1_row, so no id->row dicts
are built. Exclusivity ties are resolved exactly as matcher.exclusive(): highest prob, then smallest s1_id, then the
earliest row (shard order, then row order within the shard).
"""

import os
import pickle
import tempfile
from collections import deque
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from .extra_features import NameStats
from .matcher import build_features, from_store, xscores

_BIG = np.iinfo(np.int64).max


class CandidateIndex:
    """Integer key per S2/S3 record: candidate_row, offset by len(source 2) for source 3."""

    def __init__(self, n2, n3):
        self.n2, self.n = int(n2), int(n2) + int(n3)

    def keys(self, source, row):
        source, row = np.asarray(source), np.asarray(row, dtype=np.int64)
        if not np.isin(source, (2, 3)).all():
            raise ValueError("candidate_source must be 2 or 3")
        return np.where(source == 2, 0, self.n2) + row


class TopTwo:
    """Each candidate's two best scores over every S1 seen so far; -inf = none yet. Stored in the scores' own
    float dtype (float32 for P3's scores), so the values are exact and memory is 2 x 4 bytes per candidate."""

    def __init__(self, n, dtype=np.float64):
        self.t1 = np.full(n, -np.inf, dtype=dtype)
        self.t2 = np.full(n, -np.inf, dtype=dtype)

    def update(self, keys, scores):
        keys, s = np.asarray(keys, dtype=np.int64), np.asarray(scores)
        if s.dtype != self.t1.dtype:
            raise TypeError(f"scores are {s.dtype}, this TopTwo stores {self.t1.dtype}")
        if not len(keys):
            return
        o = np.lexsort((-s, keys))
        k, v = keys[o], s[o]
        first = np.r_[True, k[1:] != k[:-1]]
        second = np.r_[False, first[:-1] & ~first[1:]]  # the row right after a key's best, same key
        uk, b1 = k[first], v[first]
        b2 = np.full(len(uk), -np.inf, dtype=self.t1.dtype)
        b2[np.searchsorted(uk, k[second])] = v[second]
        t1, t2 = self.t1[uk], self.t2[uk]
        # top-2 of the union of two sorted pairs (t1 >= t2, b1 >= b2)
        self.t1[uk] = np.maximum(t1, b1)
        self.t2[uk] = np.maximum(np.minimum(t1, b1), np.maximum(t2, b2))

    def frame(self, keys, cand_ids):
        """matcher.top2()-style DataFrame(c, s) for these candidates, for build_features(xtop=...)."""
        uk, pos = np.unique(np.asarray(keys, dtype=np.int64), return_index=True)
        c = np.asarray(cand_ids, dtype=object)[pos]
        t1, t2 = self.t1[uk], self.t2[uk]
        has2 = t2 > -np.inf
        return pd.DataFrame({"c": np.concatenate([c, c[has2]]), "s": np.concatenate([t1, t2[has2]])})


class Winners:
    """Global exclusivity state: per candidate the best (prob desc, s1_rank asc, row asc) seen so far.
    dtype: float32 for first-stage probabilities (as matcher.to_matches), float64 for meta-model scores."""

    def __init__(self, n, dtype=np.float32):
        self.prob = np.full(n, -np.inf, dtype=dtype)
        self.s1 = np.full(n, _BIG, dtype=np.int64)
        self.row = np.full(n, _BIG, dtype=np.int64)

    def update(self, keys, prob, s1_rank, rows):
        keys = np.asarray(keys, dtype=np.int64)
        if not len(keys):
            return
        p = np.asarray(prob, dtype=self.prob.dtype)
        r, i = np.asarray(s1_rank, dtype=np.int64), np.asarray(rows, dtype=np.int64)
        o = np.lexsort((i, r, -p, keys))  # per key: best prob, then smallest s1, then earliest row
        first = np.r_[True, keys[o][1:] != keys[o][:-1]]
        sel = o[first]
        k, np_, ns, nr = keys[sel], p[sel], r[sel], i[sel]
        bp, bs, br = self.prob[k], self.s1[k], self.row[k]
        better = (np_ > bp) | ((np_ == bp) & ((ns < bs) | ((ns == bs) & (nr < br))))
        k = k[better]
        self.prob[k], self.s1[k], self.row[k] = np_[better], ns[better], nr[better]

    def accepted(self, threshold):
        """Keys whose global winner clears the threshold, compared in the stored dtype (float32 for probabilities,
        as matcher.to_matches)."""
        if threshold <= 0:
            raise ValueError("threshold must be > 0 (a non-winner has probability 0 in exclusive())")
        return np.flatnonzero(self.prob >= self.prob.dtype.type(threshold))


def s1_ranks(s1_ids):
    """Rank of every S1 id in lexicographic order (the exclusivity tie-breaker), indexed by S1 file row."""
    ids = np.asarray(s1_ids, dtype=str)
    rank = np.empty(len(ids), dtype=np.int64)
    rank[np.argsort(ids, kind="stable")] = np.arange(len(ids))
    return rank


def write_matches(path, winners, threshold, cand_ids, s1_ids, s1_rank, col="matched_entity_ids"):
    """matching_results.tsv: one row per S1 in the given (file) order, accepted candidates sorted and comma-joined.
    Same bytes as matcher.write_matching_results(to_matches(...)). Returns the number of S1 with a match."""
    acc = winners.accepted(threshold)
    return write_assignments(path, winners.s1[acc], np.asarray(cand_ids, dtype=object)[acc], s1_ids, s1_rank, col)


class Assignments:
    """Accepted (S1 rank, candidate id) pairs, looked up per S1 in sorted candidate order."""

    def __init__(self, owner_rank, cand_ids):
        owner, cids = np.asarray(owner_rank, dtype=np.int64), np.asarray(cand_ids, dtype=str)
        o = np.lexsort((cids, owner))
        self.owner, self.cids = owner[o], cids[o]

    def of(self, s1_rank):
        a, b = np.searchsorted(self.owner, s1_rank, "left"), np.searchsorted(self.owner, s1_rank, "right")
        return self.cids[a:b]


def write_assignments(path, owner_rank, cand_ids, s1_ids, s1_rank, col="matched_entity_ids"):
    """write_matches() for explicit (owner S1 rank, candidate id) arrays. Returns the number of S1 with a match."""
    acc = Assignments(owner_rank, cand_ids)
    n_matched = 0
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(f"source1_entity_id\t{col}\n")
        for s, r in zip(s1_ids, s1_rank):
            ids = acc.of(r)
            n_matched += len(ids) > 0
            fh.write(f"{s}\t{','.join(ids)}\n")
    return n_matched


def _cross_scores(model, pairs, records):
    return {c: np.asarray(v) for c, v in xscores(pairs, records, model["tfidf"]).items()}


def _probabilities(model, pairs, records, xt, name_stats, rows=None):
    """Model probabilities of the shard's pairs, or of the ``rows`` subset (boolean mask). Features are built on the
    whole shard (within-S1 ranks and gaps see every candidate) and only then subset, so a row gets the same
    probability either way."""
    groups = tuple(model.get("feature_groups", ()))
    X = build_features(pairs, records, model["tfidf"], xt, groups, name_stats)[model["features"]]
    if rows is not None:
        X = X[np.asarray(rows, dtype=bool)]
    if not len(X):
        return np.zeros(0, dtype=np.float32)
    return model["model"].predict_proba(X)[:, 1].astype(np.float32)


def _scores_features(model, pairs, records, xt, name_stats, rows=None, floor=0.0, extra=()):
    """_probabilities() plus the feature rows (float32, model["features"] order, then the ``extra`` groups: "decoy" =
    src/matching/decoy_features.py) of the scored rows whose probability is >= floor
    -> (prob, keep mask over the scored rows, X[keep]). Used by the second-stage (meta) decoder."""
    groups = tuple(model.get("feature_groups", ()))
    X = build_features(pairs, records, model["tfidf"], xt, groups, name_stats)[model["features"]]
    sel = pairs
    if rows is not None:
        X, sel = X[np.asarray(rows, dtype=bool)], pairs[np.asarray(rows, dtype=bool)]
    n_extra = 0
    if "decoy" in extra:
        from .decoy_features import FEATURES as DECOY

        n_extra = len(DECOY)
    if not len(X):
        return (np.zeros(0, dtype=np.float32), np.zeros(0, dtype=bool),
                np.zeros((0, X.shape[1] + n_extra), dtype=np.float32))
    p = model["model"].predict_proba(X)[:, 1].astype(np.float32)
    keep = p.astype(np.float64) >= floor  # the same float64 comparison as meta_decoder.meta_scores
    out = X.to_numpy(dtype=np.float32)[keep]
    if n_extra:
        from .decoy_features import decoy_features

        out = np.hstack([out, decoy_features(sel[keep], records).to_numpy(dtype=np.float32)])
    return p, keep, out


# ---------------------------------------------------------------- worker processes (spawn-safe, module level)
_W = {}


def _init_worker(model, name_stats_path):
    _W["model"] = model
    _W["name_stats"] = None
    if name_stats_path:
        with open(name_stats_path, "rb") as fh:
            _W["name_stats"] = pickle.load(fh)


def _pass1_task(args):
    return _cross_scores(_W["model"], *args)


def _pass2_task(args):
    pairs, records, xt, rows = args
    return _probabilities(_W["model"], pairs, records, xt, _W["name_stats"], rows)


def _init_role_worker(models, name_stats_path):
    """Workers holding several matchers by role (e.g. {"base": K=20 matcher, "deep": K=50 matcher})."""
    _init_worker(None, name_stats_path)
    _W["models"] = models


def _role_pass1_task(args):
    role, pairs, records = args
    return _cross_scores(_W["models"][role], pairs, records)


def _role_pass2_task(args):
    role, pairs, records, xt, rows, floor, extra = args
    return _scores_features(_W["models"][role], pairs, records, xt, _W["name_stats"], rows, floor, extra)


def _ordered(executor, fn, tasks, window):
    """(meta, result) in task order with at most ``window`` tasks in flight (bounded memory, deterministic order)."""
    queue = deque()
    for meta, args in tasks:
        queue.append((meta, executor.submit(fn, args)))
        if len(queue) >= window:
            meta0, fut = queue.popleft()
            yield meta0, fut.result()
    while queue:
        meta0, fut = queue.popleft()
        yield meta0, fut.result()


def split_ids(tables):
    """(candidate id per CandidateIndex key, S1 ids in file order, S1 exclusivity ranks) of a split's source tables."""
    cand_ids = np.concatenate([tables[2]["entity_id"].to_numpy(dtype=object),
                               tables[3]["entity_id"].to_numpy(dtype=object)])
    s1_all = tables[1]["entity_id"].to_numpy(dtype=object)
    return cand_ids, s1_all, s1_ranks(s1_all)


def pass1_tops(store, model, index, executor=None, window=2, log=print, role=None):
    """Pass 1: {score column: TopTwo} = every candidate's two best cross-entity scores over ALL S1 of the store.
    role: the matcher's key when the executor's workers hold several matchers (_init_role_worker)."""
    n_shards = len(store.shard_paths)

    def tasks():
        for df in store.iter_frames(with_records=True):
            pairs, records = from_store(df)
            args = (pairs, records) if role is None else (role, pairs, records)
            yield index.keys(df["candidate_source"], df["candidate_row"]), args

    if executor is None:
        results = ((keys, _cross_scores(model, *args[-2:])) for keys, args in tasks())
    else:
        results = _ordered(executor, _pass1_task if role is None else _role_pass1_task, tasks(), window)
    tops = {}
    for i, (keys, scores) in enumerate(results):
        for c, v in scores.items():
            if c not in tops:
                tops[c] = TopTwo(index.n, v.dtype if v.dtype.kind == "f" else np.float64)
            tops[c].update(keys, v)  # raises if a shard's dtype differs (values must stay exact)
        if i % 100 == 0:
            log(f"  pass 1: shard {i}/{n_shards}")
    return tops


def name_stats_file(name_stats):
    """Temp pickle of the NameStats for spawn workers (None -> None). The caller removes it."""
    if name_stats is None:
        return None
    fd, path = tempfile.mkstemp(suffix=".pkl", prefix="namestats_")
    with os.fdopen(fd, "wb") as fh:
        pickle.dump(name_stats, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def stream_winners(store, model, log=print, workers=1, select=None, name_stats=None):
    """Pass 1 + pass 2 over a CandidateStore with a matcher.pkl dict -> (Winners, number of pairs, number scored).

    select: optional callable(shard_path, frame) -> boolean mask of the shard rows that are scored and compete for
    candidates. Every row still feeds pass 1 and the within-S1 features, so a selected row gets the same probability
    as when the whole shard is scored. Exclusivity ties keep the global row number (shard order, then row order).
    name_stats: precomputed NameStats of this split (computed here when the model needs feature group E).

    workers > 1 scores shards in that many processes. The parent keeps the source tables and all global state
    (TopTwo, Winners) and consumes results strictly in shard order, so the result is identical to workers=1.
    Workers receive only each shard's compact pairs / records frames, so they never load the source tables."""
    tables = store.source_tables()
    index = CandidateIndex(len(tables[2]), len(tables[3]))
    rank = s1_ranks(tables[1]["entity_id"].to_numpy(dtype=object))
    paths = store.shard_paths
    n_shards = len(paths)
    groups = tuple(model.get("feature_groups", ()))
    if "E" in groups and name_stats is None:
        name_stats = NameStats.from_tables(tables)  # this split's own statistics
    if "E" not in groups:
        name_stats = None

    executor, ns_path = None, None
    if workers > 1:
        ns_path = name_stats_file(name_stats)
        executor = ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(model, ns_path))
    window = 2 * workers
    try:
        tops = pass1_tops(store, model, index, executor, window, log)

        def pass2_tasks():
            for path, df in zip(paths, store.iter_frames(with_records=True)):
                keys = index.keys(df["candidate_source"], df["candidate_row"])
                rows = None if select is None else np.asarray(select(path, df), dtype=bool)
                pairs, records = from_store(df)
                xt = {c: t.frame(keys, pairs["cand_id"]) for c, t in tops.items()}
                yield (keys, rank[df["s1_row"].to_numpy()], rows), (pairs, records, xt, rows)

        if executor is None:
            results2 = ((meta, _probabilities(model, pr, rec, xt, name_stats, rows))
                        for meta, (pr, rec, xt, rows) in pass2_tasks())
        else:
            results2 = _ordered(executor, _pass2_task, pass2_tasks(), window)
        winners, row0, n_scored = Winners(index.n), 0, 0
        for i, ((keys, s1_rank_rows, rows), prob) in enumerate(results2):
            at = row0 + np.arange(len(keys))
            if rows is not None:
                keys, s1_rank_rows, at = keys[rows], s1_rank_rows[rows], at[rows]
            winners.update(keys, prob, s1_rank_rows, at)
            row0 += len(rows) if rows is not None else len(at)
            n_scored += len(keys)
            if i % 100 == 0:
                log(f"  pass 2: shard {i}/{n_shards}")
    finally:
        if executor is not None:
            executor.shutdown()
        if ns_path:
            os.remove(ns_path)
    return winners, row0, n_scored


def stream_predict(store, model, out_path, s1_limit=None, log=print, workers=1):
    """stream_winners() + emit: matching_results.tsv of one candidate run (exclusivity + the model's threshold)."""
    winners, n_pairs, _ = stream_winners(store, model, log, workers)
    cand_ids, s1_all, rank = split_ids(store.source_tables())
    n = s1_limit or len(s1_all)
    n_matched = write_matches(out_path, winners, model["threshold"], cand_ids, s1_all[:n], rank[:n])
    return {"pairs": n_pairs, "s1": n, "s1_matched": int(n_matched)}
