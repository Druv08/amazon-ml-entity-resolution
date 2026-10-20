"""P2's blocker (src.blocking, default config) on the matcher's train S1 sample only, in CandidateStore layout.
Scoring is per S1 row ("output is identical for any shard size", blocking_data_analysis.md 10.6), so these are
exactly the pairs the full `generate_candidates --split train` run would give these S1, in minutes instead of hours.

python -m src.matching.sample_candidates --top-k 20       # -> output/candidates_p3/k20/train/...
then CandidateStore("train", out_dir="output/candidates_p3/k20").iter_frames(with_records=True, with_labels=True)

Sample (S1 list in output/candidates_p3/sample_s1.parquet): a hash-random ~20k S1 plus EVERY S1 in Bhopal (India) and
Tucson (US), so S1 entities competing for the same record are scored together (exclusivity, cross-entity features).
P2's blocking-validation hash sample is excluded: P2 tuned on it, so P3 must not evaluate on it.
"""
import argparse
import os

import numpy as np
import pandas as pd

from src.blocking import generate_candidates as gc
from src.blocking.engine import CandidateConfig
from src.blocking.handoff import load_source_table
from src.blocking.sampling import hash_fraction, in_sample

OUT = "output/candidates_p3"
CITIES = {"bhopal": "India", "tucson": "US"}


def sample_s1(data_dir="data/raw/train", rate=0.009):
    s1 = load_source_table(f"{data_dir}/train_source1.tsv")
    addr = s1["business_address"].str.lower()
    city = pd.Series("", index=s1.index)
    for c, country in CITIES.items():
        city[addr.str.contains(rf"\b{c}\b") & (s1["country"] == country)] = c
    rnd = np.array([hash_fraction(i, "p3-matcher-v1") < rate for i in s1["entity_id"]])
    p2_val = np.array([in_sample(i, 0.0025) for i in s1["entity_id"]])
    keep = (rnd | (city != "")) & ~p2_val
    return pd.DataFrame({"entity_id": s1["entity_id"], "country": s1["country"], "city": city,
                         "s1_row": np.arange(len(s1))})[keep].reset_index(drop=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    path = f"{OUT}/sample_s1.parquet"
    if not os.path.exists(path):
        os.makedirs(OUT, exist_ok=True)
        sample_s1().to_parquet(path)
    s = pd.read_parquet(path)
    print(f"sample: {len(s)} S1 ({(s.city != '').sum()} in {'/'.join(CITIES)})", s.country.value_counts().to_dict())

    config = CandidateConfig.from_dict({**CandidateConfig().to_dict(), "top_k": a.top_k})
    plan, n = [], config.shard_size
    for c, g in s.groupby("country"):
        rows = g.s1_row.to_numpy()
        plan += [(c, f"{i:05d}", rows[j:j + n]) for i, j in enumerate(range(0, len(rows), n))]
    gc.plan_shards = lambda enc, config, s1_limit=None: plan  # ponytail: reuse run() whole; only the row plan differs
    gc.run("train", "data/raw/train", "data/processed/blocking", f"{OUT}/k{a.top_k}", config,
           workers=a.workers, write_tsv=False)
