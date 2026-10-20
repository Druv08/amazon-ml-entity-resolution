"""Candidate feature groups for the matcher, motivated by docs/p3_matching.md "Error analysis".

Every group is computable identically for train and test from the pairs, their records and (group E) statistics of
the split's own source files. Nothing external is used.

  A  address numbers     typo-tolerant house-number agreement, shared / conflicting counts, word-only address overlap
  C  transliteration     P2's offline Indic->Latin transliteration and phonetic keys (src/blocking/features.py)
  D  blocker evidence    address part of P2's score (block_score - name_score)
  E  name rarity         how common the core name / its words are in the split (chains vs unique businesses)
"""

import hashlib
import math
import os
import pickle
from collections import Counter

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

from src.blocking.features import latin_name, record_features
from src.blocking.normalize import has_nonlatin

from .matcher import _STOP, _jaccard, _norm

GROUPS = ("A", "C", "D", "E")
# A, C, E: docs/p3_matching.md "Targeted features" (D did not help); T: "Token-alignment features"
PRODUCTION_GROUPS = ("A", "C", "E", "T")


def _texts(pairs, records, col):
    rec = records.set_index("entity_id")[col]
    return rec.reindex(pairs["s1_id"]).fillna("").tolist(), rec.reindex(pairs["cand_id"]).fillna("").tolist()


def _cached(values, fn):
    cache = {v: fn(v) for v in set(values)}
    return [cache[v] for v in values]


# ---------------------------------------------------------------- A: address numbers
def address_number_features(pairs, records):
    sa, sb = _texts(pairs, records, "business_address")
    ta, tb = _cached(sa, _norm), _cached(sb, _norm)
    f = pd.DataFrame(index=pairs.index)
    na = [[t for t in w if t.isdigit()] for w in ta]
    nb = [[t for t in w if t.isdigit()] for w in tb]
    wa = [{t for t in w if not t.isdigit()} for w in ta]
    wb = [{t for t in w if not t.isdigit()} for w in tb]
    f["num_shared_n"] = [len(set(x) & set(y)) for x, y in zip(na, nb)]
    f["num_only_s1_n"] = [len(set(x) - set(y)) for x, y in zip(na, nb)]
    f["num_only_cand_n"] = [len(set(y) - set(x)) for x, y in zip(na, nb)]
    best, prefix = [], []
    for x, y in zip(na, nb):
        if not x or not y:
            best.append(-1.0)
            prefix.append(-1)
            continue
        best.append(max(fuzz.ratio(u, v) for u in x for v in y) / 100)
        prefix.append(int(any(u != v and min(len(u), len(v)) >= 2 and (u.startswith(v) or v.startswith(u))
                              for u in x for v in y)))
    f["num_best_ratio"] = best  # 1.0 = a shared number; ~0.75-0.9 = one-digit typo; -1 = a side has no number
    f["num_prefix"] = prefix
    f["addr_word_jacc"] = [_jaccard(x, y) for x, y in zip(wa, wb)]
    return f


# ---------------------------------------------------------------- C: transliteration / phonetics
def translit_features(pairs, records):
    sa, sb = _texts(pairs, records, "business_name")

    def keys(v):
        r = record_features(v, "", with_fallback=False)
        return r["p"], r["pb"]

    ka, kb = _cached(sa, keys), _cached(sb, keys)
    la = _cached(sa, lambda v: " ".join(_norm(latin_name(v))))
    lb = _cached(sb, lambda v: " ".join(_norm(latin_name(v))))
    nla, nlb = _cached(sa, has_nonlatin), _cached(sb, has_nonlatin)
    f = pd.DataFrame(index=pairs.index)
    f["ph_jacc"] = [_jaccard(x[0], y[0]) for x, y in zip(ka, kb)]
    f["ph_bigram_hit"] = [int(bool(x[1] & y[1])) for x, y in zip(ka, kb)]
    f["latin_tsort"] = process.cpdist(la, lb, scorer=fuzz.token_sort_ratio, workers=-1) / 100
    f["script_mismatch"] = [int(x != y) for x, y in zip(nla, nlb)]
    return f


# ---------------------------------------------------------------- D: blocker evidence
def blocker_features(pairs):
    f = pd.DataFrame(index=pairs.index)
    f["block_addr_score"] = (pairs["block_score"] - pairs["name_score"]).to_numpy()
    return f


# ---------------------------------------------------------------- E: name rarity
def _core(name):
    return tuple(sorted({t for t in _norm(name) if t not in _STOP}))


def core_key(country, core):
    """Stable 64-bit key of (country, core-name tokens); Python's hash() is randomised per process."""
    return int.from_bytes(hashlib.blake2b("\x1f".join((country,) + tuple(core)).encode("utf-8"),
                                         digest_size=8).digest(), "little")


class NameStats:
    """Core-name and name-word frequencies of ONE split (train or test), per country.
    Core names are stored as core_key() integers to keep memory low (~10M distinct names per split)."""

    def __init__(self, s1_core, cand_core, word_df, n_docs):
        self.s1_core, self.cand_core, self.word_df, self.n_docs = s1_core, cand_core, word_df, n_docs

    @classmethod
    def from_tables(cls, tables, chunk=500_000):
        """tables: {1: S1 frame, 2: S2 frame, 3: S3 frame} with business_name and country (all records of the split)."""
        s1_core, cand_core, word_df, n_docs = Counter(), Counter(), Counter(), Counter()
        for s in (1, 2, 3):
            names, countries = tables[s]["business_name"].fillna(""), tables[s]["country"]
            target = s1_core if s == 1 else cand_core
            for a in range(0, len(names), chunk):  # a bounded name cache per chunk
                part = names.iloc[a:a + chunk].tolist()
                for c, core in zip(countries.iloc[a:a + chunk].tolist(), _cached(part, _core)):
                    target[core_key(c, core)] += 1
                    n_docs[c] += 1
                    for w in core:
                        word_df[c, w] += 1
        return cls(s1_core, cand_core, word_df, n_docs)

    def s1_count(self, country, core):
        return self.s1_core.get(core_key(country, core), 0)

    def cand_count(self, country, core):
        return self.cand_core.get(core_key(country, core), 0)

    def idf(self, country, word):
        return math.log((self.n_docs[country] + 1) / (self.word_df[country, word] + 1))


def load_name_stats(split="train", data_dir=None, cache_dir="output/cache"):
    """NameStats of a split's three raw source files, cached (git-ignored) because it scans every record."""
    path = os.path.join(cache_dir, f"name_stats_v2_{split}.pkl")
    if os.path.exists(path):
        with open(path, "rb") as fh:
            return pickle.load(fh)
    from src.blocking.data_io import source_path
    from src.blocking.handoff import load_source_table

    data_dir = data_dir or os.path.join("data", "raw", split)
    stats = NameStats.from_tables({s: load_source_table(source_path(data_dir, split, s)) for s in (1, 2, 3)})
    os.makedirs(cache_dir, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(stats, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return stats


def name_rarity_features(pairs, records, stats):
    sa, sb = _texts(pairs, records, "business_name")
    ca, cb = _cached(sa, _core), _cached(sb, _core)
    country = records.set_index("entity_id")["country"].reindex(pairs["s1_id"]).tolist()
    f = pd.DataFrame(index=pairs.index)
    f["s1_core_freq"] = [math.log1p(stats.s1_count(c, k)) for c, k in zip(country, ca)]
    f["cand_core_freq"] = [math.log1p(stats.cand_count(c, k)) for c, k in zip(country, cb)]
    jac, rarest = [], []
    for c, x, y in zip(country, ca, cb):
        x, y = set(x), set(y)
        union = x | y
        w_union = sum(stats.idf(c, w) for w in union)
        shared = [stats.idf(c, w) for w in x & y]
        jac.append(sum(shared) / w_union if w_union else 0.0)
        rarest.append(max(shared) if shared else 0.0)
    f["core_idf_jacc"] = jac  # word overlap weighted by rarity: sharing "sai" + "enterprises" counts little
    f["core_rarest_shared_idf"] = rarest
    return f


def extra_features(groups, pairs, records, name_stats=None):
    parts = []
    for g in groups:
        if g == "A":
            parts.append(address_number_features(pairs, records))
        elif g == "C":
            parts.append(translit_features(pairs, records))
        elif g == "D":
            parts.append(blocker_features(pairs))
        elif g == "E":
            if name_stats is None:
                raise ValueError("group E needs NameStats of the split")
            parts.append(name_rarity_features(pairs, records, name_stats))
        elif g == "T":
            from .decoy_features import decoy_features

            parts.append(decoy_features(pairs, records))
        else:
            raise ValueError(f"unknown feature group {g!r}")
    return pd.concat(parts, axis=1).astype(np.float32) if parts else pd.DataFrame(index=pairs.index)
