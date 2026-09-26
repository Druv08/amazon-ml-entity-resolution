"""Test inference on P2's test candidates: features -> model -> exclusivity -> threshold -> matching_results.tsv.
output/candidate_pairs.tsv is P2's file (generate_candidates writes it); the matches are a subset of it.

python -m src.blocking.generate_candidates --split test --top-k 20      # K must equal the K the model was trained on
python -m src.matching.predict --model output/candidates_p3/k20/matcher.pkl

Two passes over the shards: pass 1 records each candidate's top-2 cross-entity scores over ALL S1, pass 2 builds
features with them, so a competing S1 in another shard still counts (as it does in training).
"""
import argparse
import os
import pickle
import time

import numpy as np
import pandas as pd

from src.blocking.handoff import CandidateStore
from src.matching.matcher import build_features, exclusive, from_store, to_matches, top2, write_matching_results, xtop

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="output/candidates_p3/k20/matcher.pkl")
ap.add_argument("--split", default="test")
ap.add_argument("--cands", default="output/candidates", help="out_dir of generate_candidates")
ap.add_argument("--out", default="output")
a = ap.parse_args()
t0 = time.time()

with open(a.model, "rb") as fh:
    m = pickle.load(fh)
store = CandidateStore(a.split, out_dir=a.cands)
k = store.meta["config"]["top_k"]
if k != m["top_k"]:  # rank/gap/cross-entity features depend on the candidate list length
    raise SystemExit(f"model trained on K={m['top_k']} candidates, {a.split} candidates have K={k}")
tf = m["tfidf"]

tops = []
for i, df in enumerate(store.iter_frames(with_records=True)):
    tops.append(xtop(*from_store(df), tf))
    if i % 100 == 0:
        print(f"  pass 1: shard {i}/{len(store.shard_paths)}, {time.time() - t0:.0f}s", flush=True)
xt = {}
for c in tops[0]:
    d = pd.concat(t[c] for t in tops)
    xt[c] = top2(d["s"], d["c"])
del tops, d

parts = []  # compact columns only: exclusivity needs every pair's prob before deciding
for i, df in enumerate(store.iter_frames(with_records=True)):
    pairs, records = from_store(df)
    X = build_features(pairs, records, tf, xt)[m["features"]]
    parts.append(pairs[["s1_id", "cand_id"]].assign(prob=m["model"].predict_proba(X)[:, 1].astype(np.float32)))
    if i % 100 == 0:
        print(f"  pass 2: shard {i}/{len(store.shard_paths)}, {time.time() - t0:.0f}s", flush=True)
scored = pd.concat(parts, ignore_index=True)
del parts

s1_ids = store.source_tables()[1]["entity_id"]  # every S1, file order; a run with --s1-limit covers its first N
s1_ids = s1_ids.iloc[:store.meta["s1_limit"]] if store.meta["s1_limit"] else s1_ids
matches = to_matches(scored, exclusive(scored, scored["prob"].to_numpy()), m["threshold"], s1_ids)
os.makedirs(a.out, exist_ok=True)
write_matching_results(matches, f"{a.out}/matching_results.tsv")
print(f"wrote {a.out}/matching_results.tsv in {time.time() - t0:.0f}s | S1 with a match: "
      f"{np.mean([bool(v) for v in matches.values()]):.3f}")
