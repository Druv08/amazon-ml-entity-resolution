"""Fingerprinted cache of a development run's matcher feature matrix, so model / decoder experiments reuse it instead
of recomputing RapidFuzz, TF-IDF, transliteration and name rarity.

    meta, X = feature_cache("output/candidates_p3/k20")      # meta: s1_id, cand_id, rank, label (OOF row order)

The cache (<cands>/X_ACE.parquet, git-ignored) is valid only for one fingerprint:
  feature version + feature groups, the candidate run's config (run.json), the S1 sample ids, and the identity
  (size + SHA-256 of the first MiB) of the training source files. A mismatching cache is refused, never reused.
"""

import hashlib
import json
import os
import time

import pandas as pd

FEATURE_VERSION = "matcher-61-ACE-v1"  # bump whenever build_features / extra_features change their output
GROUPS = ("A", "C", "E")
DATA_FILES = ("train_source1.tsv", "train_source2.tsv", "train_source3.tsv", "train_ground_truth.tsv")


def _file_identity(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        h.update(fh.read(2 ** 20))
    return {"size": os.path.getsize(path), "head_sha256": h.hexdigest()}


def fingerprint(cands, groups=GROUPS, data_dir="data/raw/train"):
    from src.matching.sample_candidates import OUT

    with open(os.path.join(cands, "train", "run.json"), encoding="utf-8") as fh:
        run = json.load(fh)
    s1 = sorted(pd.read_parquet(f"{OUT}/sample_s1.parquet", columns=["entity_id"])["entity_id"])
    return {"feature_version": FEATURE_VERSION, "groups": list(groups),
            "candidates": {k: run.get(k) for k in ("split", "encoding", "config", "s1_limit")},
            "sample_s1": {"n": len(s1), "sha256": hashlib.sha256("\n".join(s1).encode()).hexdigest()},
            "sources": {f: _file_identity(os.path.join(data_dir, f)) for f in DATA_FILES}}


def _paths(cands, groups=GROUPS):
    g = "".join(groups)
    return os.path.join(cands, f"X_{g}.parquet"), os.path.join(cands, f"X_{g}.fingerprint.json")


def write_cache(cands, groups, pairs, X, seconds=None):
    """Store a feature matrix built elsewhere (train.py) with its fingerprint, for later experiments / train_meta."""
    path, fp_path = _paths(cands, groups)
    X.assign(**{"__" + c: pairs[c].to_numpy() for c in ("s1_id", "cand_id", "rank", "label")}).to_parquet(path)
    with open(fp_path, "w", encoding="utf-8") as fh:
        json.dump({**fingerprint(cands, groups), "build": {"seconds": seconds, "by": "train.py"}}, fh, indent=2)


def _split(frame):
    meta_cols = [c for c in frame.columns if c.startswith("__")]
    return frame[meta_cols].rename(columns=lambda c: c[2:]), frame.drop(columns=meta_cols)


def _legacy_ok(cands, frame):
    """A cache written before fingerprints existed is adopted only if its rows are the run's OOF rows and its columns
    are the saved matcher's features."""
    import pickle

    meta, X = _split(frame)
    oof = pd.read_parquet(os.path.join(cands, "oof.parquet"), columns=["s1_id", "cand_id", "label"])
    with open(os.path.join(cands, "matcher.pkl"), "rb") as fh:
        feats = pickle.load(fh)["features"]
    return (meta[["s1_id", "cand_id"]].reset_index(drop=True).equals(oof[["s1_id", "cand_id"]])
            and (meta["label"].to_numpy() == oof["label"].to_numpy()).all() and list(X.columns) == list(feats))


def feature_cache(cands, groups=GROUPS, rebuild=False, log=print):
    """(meta frame s1_id/cand_id/rank/label, X float32) of a development run, built once with train.featurize."""
    path, fp_path = _paths(cands, groups)
    want = fingerprint(cands, groups)
    if os.path.exists(path) and not rebuild:
        if os.path.exists(fp_path):
            with open(fp_path, encoding="utf-8") as fh:
                have = json.load(fh)
            if {k: v for k, v in have.items() if k != "build"} != want:
                raise SystemExit(f"{path}: incompatible feature cache (fingerprint differs); rebuild it with "
                                 f"feature_cache(..., rebuild=True)")
            return _split(pd.read_parquet(path))
        frame = pd.read_parquet(path)
        if not _legacy_ok(cands, frame):
            raise SystemExit(f"{path}: cache without fingerprint does not match the run; rebuild it")
        with open(fp_path, "w", encoding="utf-8") as fh:
            json.dump({**want, "build": {"adopted_legacy_cache": True}}, fh, indent=2)
        log(f"adopted existing cache {path} after row / column verification")
        return _split(frame)
    from src.matching.train import featurize, load

    t0 = time.time()
    pairs, records, truth, s1, claimed, _ = load(cands)
    X, y, _, _ = featurize(pairs, records, claimed, groups)
    X.assign(**{"__" + c: pairs[c].to_numpy() for c in ("s1_id", "cand_id", "rank", "label")}).to_parquet(path)
    with open(fp_path, "w", encoding="utf-8") as fh:
        json.dump({**want, "build": {"seconds": round(time.time() - t0, 1)}}, fh, indent=2)
    return _split(pd.read_parquet(path))
