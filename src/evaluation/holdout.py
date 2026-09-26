"""Untouched FINAL holdout of training S1 entities.

    !!! DO NOT use this holdout for feature selection, K selection, threshold tuning, model selection or any
    !!! repeated development evaluation. Score it ONCE, after the model and every setting are frozen.
    !!! Looking at its macro-F0.5 earlier spoils it as an unbiased estimate.

Selection (deterministic, order-independent, see select_holdout):
  * hash:     md5("final-holdout-v1:" + s1_id) mapped to [0, 1) < HOLDOUT_RATE (src.blocking.sampling.hash_fraction)
  * rate:     1.35% of the 2,206,821 training S1 -> about 29-30k S1 after exclusions
  * excluded: every S1 of an earlier development sample, so the holdout never overlaps them:
      - P2 blocking validation sample   hash salt "blocking-validation-v1", rate 0.0025 (P2 tuned the blocker on it)
      - P3 matcher development sample   hash salt "p3-matcher-v1", rate 0.009 (features, K, threshold tuned on it)
      - P3 dense city sample            every S1 in Bhopal (India) / Tucson (US), part of the P3 training sample
    The new salt is used nowhere else in the repository.

    python -m src.evaluation.holdout          # writes IDs + metadata to output/holdout/ (git-ignored) and prints
                                               # the overlap proof; it never reads ground truth or scores anything
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

from src.blocking.handoff import load_source_table
from src.blocking.sampling import DEFAULT_SALT as P2_VALIDATION_SALT, hash_fraction
from src.matching.sample_candidates import CITIES

HOLDOUT_SALT = "final-holdout-v1"
HOLDOUT_RATE = 0.0135
P2_VALIDATION_RATE = 0.0025  # src/blocking/run_experiments.py / evaluate.py --sample-rate default
P3_DEV_SALT, P3_DEV_RATE = "p3-matcher-v1", 0.009  # src/matching/sample_candidates.sample_s1


def _hash_mask(ids, salt, rate):
    return np.fromiter((hash_fraction(i, salt) < rate for i in ids), dtype=bool, count=len(ids))


def development_masks(s1):
    """Boolean masks (over the rows of an S1 frame) of every earlier development sample."""
    ids = s1["entity_id"].tolist()
    addr = s1["business_address"].str.lower()
    city = np.zeros(len(s1), dtype=bool)
    for c, country in CITIES.items():
        city |= (addr.str.contains(rf"\b{c}\b") & (s1["country"] == country)).to_numpy()
    return {
        "p2_blocking_validation": _hash_mask(ids, P2_VALIDATION_SALT, P2_VALIDATION_RATE),
        "p3_random_dev": _hash_mask(ids, P3_DEV_SALT, P3_DEV_RATE),
        "p3_city_dev": city,
    }


def select_holdout(s1):
    """-> DataFrame(entity_id, country, s1_row) of the final holdout, from an S1 frame in file order
    (entity_id, business_address, country)."""
    dev = development_masks(s1)
    keep = _hash_mask(s1["entity_id"].tolist(), HOLDOUT_SALT, HOLDOUT_RATE)
    for m in dev.values():
        keep &= ~m
    return pd.DataFrame({"entity_id": s1["entity_id"].to_numpy(), "country": s1["country"].to_numpy(),
                         "s1_row": np.arange(len(s1))})[keep].reset_index(drop=True)


def overlap_report(s1, holdout):
    """Counts proving the holdout is disjoint from every development sample (all must be 0)."""
    dev = development_masks(s1)
    in_holdout = s1["entity_id"].isin(holdout["entity_id"]).to_numpy()
    return {name: int((m & in_holdout).sum()) for name, m in dev.items()}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data/raw/train")
    ap.add_argument("--out-dir", default="output/holdout")
    a = ap.parse_args(argv)
    s1 = load_source_table(os.path.join(a.data_dir, "train_source1.tsv"))
    h = select_holdout(s1)
    overlap = overlap_report(s1, h)
    if any(overlap.values()) or not h["entity_id"].is_unique:
        raise SystemExit(f"holdout overlaps a development sample: {overlap}")
    os.makedirs(a.out_dir, exist_ok=True)
    h.to_parquet(os.path.join(a.out_dir, "final_holdout_s1.parquet"))  # IDs only, no records / ground truth
    meta = {"salt": HOLDOUT_SALT, "rate": HOLDOUT_RATE, "s1": len(h), "s1_total": len(s1),
            "by_country": h["country"].value_counts().to_dict(), "overlap_with_development_samples": overlap,
            "warning": "DO NOT use for feature selection, K selection, threshold tuning or repeated evaluation"}
    with open(os.path.join(a.out_dir, "final_holdout_meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
