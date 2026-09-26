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
    """Global exclusivity state: per candidate the best (prob desc, s1_rank asc, row asc) seen so far."""

    def __init__(self, n):
        self.prob = np.full(n, -np.inf, dtype=np.float32)
        self.s1 = np.full(n, _BIG, dtype=np.int64)
        self.row = np.full(n, _BIG, dtype=np.int64)

    def update(self, keys, prob, s1_rank, rows):
        keys = np.asarray(keys, dtype=np.int64)
        if not len(keys):
            return
        p = np.asarray(prob, dtype=np.float32)
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
        """Keys whose global winner clears the threshold (float32 comparison, as matcher.to_matches)."""
        if threshold <= 0:
            raise ValueError("threshold must be > 0 (a non-winner has probability 0 in exclusive())")
        return np.flatnonzero(self.prob >= np.float32(threshold))


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
    owner, cids = winners.s1[acc], np.asarray(np.asarray(cand_ids, dtype=object)[acc], dtype=str)
    o = np.lexsort((cids, owner))
    owner, cids = owner[o], cids[o]
    n_matched = 0
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(f"source1_entity_id\t{col}\n")
        for s, r in zip(s1_ids, s1_rank):
            a, b = np.searchsorted(owner, r, "left"), np.searchsorted(owner, r, "right")
            n_matched += b > a
            fh.write(f"{s}\t{','.join(cids[a:b])}\n")
    return n_matched


def stream_predict(store, model, out_path, s1_limit=None, log=print):
    """Pass 1 + pass 2 + emit over a CandidateStore with a matcher.pkl dict (model, threshold, features, tfidf)."""
    tables = store.source_tables()
    index = CandidateIndex(len(tables[2]), len(tables[3]))
    cand_ids = np.concatenate([tables[2]["entity_id"].to_numpy(dtype=object),
                               tables[3]["entity_id"].to_numpy(dtype=object)])
    s1_all = tables[1]["entity_id"].to_numpy(dtype=object)
    rank = s1_ranks(s1_all)
    n_shards = len(store.shard_paths)
    groups = tuple(model.get("feature_groups", ()))
    name_stats = NameStats.from_tables(tables) if "E" in groups else None  # this split's own statistics

    tops = {}
    for i, df in enumerate(store.iter_frames(with_records=True)):
        keys = index.keys(df["candidate_source"], df["candidate_row"])
        for c, v in xscores(*from_store(df), model["tfidf"]).items():
            v = np.asarray(v)
            if c not in tops:
                tops[c] = TopTwo(index.n, v.dtype if v.dtype.kind == "f" else np.float64)
            tops[c].update(keys, v)  # raises if a shard's dtype differs (values must stay exact)
        if i % 100 == 0:
            log(f"  pass 1: shard {i}/{n_shards}")

    winners, row0, n_pairs = Winners(index.n), 0, 0
    for i, df in enumerate(store.iter_frames(with_records=True)):
        keys = index.keys(df["candidate_source"], df["candidate_row"])
        pairs, records = from_store(df)
        xt = {c: t.frame(keys, pairs["cand_id"]) for c, t in tops.items()}
        X = build_features(pairs, records, model["tfidf"], xt, groups, name_stats)[model["features"]]
        prob = model["model"].predict_proba(X)[:, 1].astype(np.float32)
        winners.update(keys, prob, rank[df["s1_row"].to_numpy()], row0 + np.arange(len(df)))
        row0 += len(df)
        n_pairs += len(df)
        if i % 100 == 0:
            log(f"  pass 2: shard {i}/{n_shards}")

    n = s1_limit or len(s1_all)
    n_matched = write_matches(out_path, winners, model["threshold"], cand_ids, s1_all[:n], rank[:n])
    return {"pairs": n_pairs, "s1": n, "s1_matched": int(n_matched)}
