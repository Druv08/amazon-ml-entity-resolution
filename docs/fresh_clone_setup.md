# Fresh-clone production handoff

This document records the runtime dependency audit for the adopted Hybrid50 production pipeline on branch
`updated-p2-p3`.

## Dependency classification

### A. Tracked: supplied by `git clone`

- Raw TSV loading and path rules: `src/blocking/data_io.py`
- Blocking tokenisation, normalisation, transliteration, encoding, and scoring: `src/blocking/`
- K=20/K=50 generation and official candidate writer: `src/blocking/generate_candidates.py`
- CandidateStore: `src/blocking/handoff.py`
- M3 matcher and A+C+E+T features: `src/matching/matcher.py`, `extra_features.py`, and `decoy_features.py`
- Streaming scoring and global exclusivity: `src/matching/stream.py`
- Hybrid50 candidate construction: `src/pipeline/hybrid.py`
- Meta-decoder and production orchestration: `src/pipeline/meta_decoder.py` and `predict_hybrid.py`
- Bundle rebuild path: `src/matching/sample_candidates.py`, `train.py`, `feature_cache.py`, and
  `src/pipeline/train_meta.py`
- Official output validation: `resources/utils/validate_submission.py`
- Pinned runtime dependencies: `requirements.txt`
- Fresh-clone preflight: `scripts/check_environment.py`

All Python source files referenced by the production imports are tracked. Production does not import anything from
`data/clean/` or from an untracked local source directory.

### B. Official dataset: extract separately

```text
data/raw/train/train_source1.tsv
data/raw/train/train_source2.tsv
data/raw/train/train_source3.tsv
data/raw/train/train_ground_truth.tsv
data/raw/test/test_source1.tsv
data/raw/test/test_source2.tsv
data/raw/test/test_source3.tsv
```

The test files are sufficient for candidate generation and inference when the trained bundle is transferred. The
train files are additionally required to rebuild the bundle.

### C. Generated/rebuildable: do not transfer or commit

- `data/processed/blocking/`: encoded sources and memory-mapped country indices; rebuilt by candidate generation.
- `output/candidates_k20/` and `output/candidates_k50/`: CandidateStore shards; rebuilt from official test data.
- `output/candidates_p3/`: development samples, train CandidateStores, feature caches, OOF predictions, and matcher
  artifacts; needed only to retrain/rebuild the final bundle.
- `output/cache/`: optional computed statistics.
- `output/candidate_pairs.tsv`, `output/matching_results.tsv`, and `output/hybrid_summary.json`: final generated
  outputs.
- Smoke-test and analysis outputs under `output/`.

### D. Trained artifact: transfer separately or rebuild

```text
artifact:      hybrid_meta.pkl
required path: output/candidates_p3/hybrid_meta.pkl
size:          14,713,222 bytes
SHA-256:       f9bb2c379e95a6157ea04535c33bc7b2b82c343af6a93e49a9363cea0b3e1a89
```

The inspected production bundle contains:

- K=20 and K=50 M3 `HistGradientBoostingClassifier` matchers;
- each matcher's fitted name/address TF-IDF vectorizers;
- first-stage thresholds, K values, 74-feature lists, `max_leaf_nodes=63`, and feature groups A+C+E+T;
- the fitted meta-model, 90 meta-feature names, meta floor, and meta version;
- production meta thresholds (`base=0.74`, `deep=0.63`);
- Hybrid50 configuration (`base_top_k=20`, `deep_top_k=50`, `cap=50`);
- first-stage SHA-256 values and source/cache fingerprints.

The bundle is the only trained artifact read by the default `python -m src.pipeline.predict_hybrid` path. The
separate K20/K50 `matcher.pkl` defaults are used only with `--decoder threshold`; they are already embedded in the
bundle for the default `--decoder meta` path.

The provided competition text requires a reproducible source package and a permissively licensed model, but it does
not explicitly say that learned artifacts may be published. The bundle is small enough for normal Git and its
scikit-learn model type is BSD-licensed, but its fitted TF-IDF vectorizers contain vocabulary derived from the
restricted training text. It therefore remains ignored instead of being placed in this public repository. Keep it in
the team-approved private transfer channel or rebuild it from each teammate's official dataset. Never load a pickle
received from an untrusted source.

### E. Unnecessary for production

- `data/clean/` and the old P1 cleaned CSVs.
- Standalone experimental models, OOF files, feature matrices, and error-analysis outputs once the final bundle is
  available.
- `xgboost`, which is imported only by optional development experiments and is not part of the production path.
- `psutil`, which is optional candidate-generation memory telemetry guarded by `try/except`.

## Why no other laptop files are required

At inference, both CandidateStores are regenerated from the official raw test TSVs. `predict_hybrid` loads the two
source tables from those TSVs, computes split-specific Group E name statistics, reads both trained matchers and the
meta-model from `hybrid_meta.pkl`, performs streaming scoring/exclusivity, and writes both official TSVs. Therefore,
apart from the official dataset, the only file that must be transferred is `hybrid_meta.pkl`.

## Preflight and production commands

After environment setup, dataset extraction, and bundle restoration:

```powershell
python scripts/check_environment.py
python -m src.blocking.generate_candidates --split test --top-k 20 --out-dir output/candidates_k20 --no-tsv
python -m src.blocking.generate_candidates --split test --top-k 50 --out-dir output/candidates_k50 --no-tsv
python -m src.pipeline.predict_hybrid
python resources/utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir data/raw/test
```

The preflight treats missing candidate directories as pending generated work. Missing packages, test data, or the
production bundle are blockers for inference.
