"""Test inference on P2's test candidates: features -> model -> exclusivity -> threshold -> matching_results.tsv.
output/candidate_pairs.tsv is P2's file (generate_candidates writes it); the matches are a subset of it.

python -m src.blocking.generate_candidates --split test --top-k 20      # K must equal the K the model was trained on
python -m src.matching.predict --model output/candidates_p3/k20/matcher.pkl

Streams the shards twice with memory bounded by one shard plus per-candidate arrays (src/matching/stream.py):
pass 1 records each candidate's top-2 cross-entity scores over ALL S1, pass 2 builds features with them (so a
competing S1 in another shard still counts, as in training) and keeps a global exclusivity winner per candidate.
"""
import argparse
import os
import pickle
import time

from src.blocking.handoff import CandidateStore
from src.matching.matcher import check_top_k
from src.matching.stream import stream_predict

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="output/candidates_p3/k20/matcher.pkl")
    ap.add_argument("--split", default="test")
    ap.add_argument("--cands", default="output/candidates", help="out_dir of generate_candidates")
    ap.add_argument("--out", default="output")
    ap.add_argument("--workers", type=int, default=2,
                    help="scoring processes; output is byte-identical for any value. 2 is the measured best "
                         "trade-off (docs/p3_matching.md 'Inference performance'); each extra worker costs ~1 GB")
    a = ap.parse_args()
    t0 = time.time()

    with open(a.model, "rb") as fh:
        m = pickle.load(fh)
    store = CandidateStore(a.split, out_dir=a.cands)
    check_top_k(m["top_k"], store.meta["config"]["top_k"], a.split)
    os.makedirs(a.out, exist_ok=True)
    path = f"{a.out}/matching_results.tsv"
    r = stream_predict(store, m, path, s1_limit=store.meta["s1_limit"], workers=a.workers,
                       log=lambda msg: print(f"{msg}, {time.time() - t0:.0f}s", flush=True))
    print(f"wrote {path} in {time.time() - t0:.0f}s | {r['pairs']} pairs | S1 with a match: "
          f"{r['s1_matched'] / max(r['s1'], 1):.3f}")
