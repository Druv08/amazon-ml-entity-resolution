"""Consumer API for the matcher (P3). No blocking internals needed.

    from src.blocking.handoff import CandidateStore

    store = CandidateStore("train")                  # output of generate_candidates
    for df in store.iter_frames(with_records=True, with_labels=True):
        ...  # one pandas DataFrame per shard (~200k rows), columns below

Columns of every frame
    s1_entity_id, candidate_entity_id, rank (1 = best), score, name_score,
    candidate_source (2/3), candidate_row, s1_row
  + with_records: s1_name, s1_address, s1_country,
                  candidate_name, candidate_address, candidate_country
  + with_labels (train only): label (1 if the pair is in train_ground_truth.tsv)

Frames are streamed shard by shard, so memory stays bounded; source tables are
loaded once (about 1-2 GB for the training split) when ``with_records`` is used.
"""

import csv
import glob
import json
import os

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .data_io import ground_truth_path, iter_ground_truth, source_path


def read_candidate_pairs(path):
    """Yield ``(source1_entity_id, [candidate ids])`` from a candidate_pairs.tsv."""
    with open(path, encoding="utf-8") as f:
        header = f.readline().rstrip("\r\n").split("\t")
        if header != ["source1_entity_id", "candidate_entity_ids"]:
            raise ValueError(f"{path}: unexpected header {header}")
        for line in f:
            s1, ids = line.rstrip("\r\n").split("\t")
            yield s1, (ids.split(",") if ids else [])


def load_source_table(path):
    """One source TSV as a DataFrame (all columns as strings, no NA parsing)."""
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, na_filter=False,
                       quoting=csv.QUOTE_NONE, engine="c")


def load_ground_truth_pairs(data_dir="data/raw/train"):
    """Set of true ``(s1_id, candidate_id)`` pairs of the training split."""
    return {(s1, c) for s1, ids in iter_ground_truth(ground_truth_path(data_dir)) for c in ids}


class CandidateStore:
    def __init__(self, split, out_dir="output/candidates", data_dir=None):
        self.split = split
        self.run_dir = os.path.join(out_dir, split)
        self.data_dir = data_dir or os.path.join("data", "raw", split)
        meta = os.path.join(self.run_dir, "run.json")
        if not os.path.exists(meta):
            raise FileNotFoundError(f"no candidates for split '{split}' in {self.run_dir}; run "
                                    f"python -m src.blocking.generate_candidates --split {split}")
        with open(meta, encoding="utf-8") as f:
            self.meta = json.load(f)
        self._tables = None

    @property
    def shard_paths(self):
        return sorted(glob.glob(os.path.join(self.run_dir, "shards", "*", "*.parquet")))

    def iter_shards(self):
        """Raw long-format candidate frames, one per shard."""
        for p in self.shard_paths:
            yield pq.read_table(p).to_pandas()

    def source_tables(self):
        if self._tables is None:
            self._tables = {s: load_source_table(source_path(self.data_dir, self.split, s)) for s in (1, 2, 3)}
        return self._tables

    def iter_frames(self, with_records=True, with_labels=False):
        truth = None
        if with_labels:
            if self.split != "train":
                raise ValueError("labels exist only for the train split")
            truth = load_ground_truth_pairs(self.data_dir)
        tables = self.source_tables() if with_records else None
        for df in self.iter_shards():
            if with_records:
                s1 = tables[1].iloc[df["s1_row"].to_numpy()]
                df["s1_name"] = s1["business_name"].to_numpy()
                df["s1_address"] = s1["business_address"].to_numpy()
                df["s1_country"] = s1["country"].to_numpy()
                for col, src_col in (("candidate_name", "business_name"), ("candidate_address", "business_address"),
                                     ("candidate_country", "country")):
                    out = np.empty(len(df), dtype=object)
                    for s in (2, 3):
                        m = (df["candidate_source"] == s).to_numpy()
                        # take the shard's rows first: converting the whole multi-million-row column per shard
                        # made every shard read cost ~2 s regardless of its size
                        out[m] = tables[s][src_col].take(df.loc[m, "candidate_row"].to_numpy()).to_numpy()
                    df[col] = out
            if truth is not None:
                df["label"] = [int((a, b) in truth) for a, b in zip(df["s1_entity_id"], df["candidate_entity_id"])]
            yield df
