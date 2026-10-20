# Amazon ML Entity Resolution

Production solution for the Amazon ML Challenge 2026 Business Entity Resolution problem. The adopted system is
Hybrid50: exact K=20 base candidates plus selected K=50 deep candidates, two M3 matchers with feature groups
A+C+E+T, streaming inference, global candidate exclusivity, and a learned meta-decoder.

Raw competition data and generated outputs are intentionally excluded from Git.

## Fresh machine setup

The commands below are the supported production path. Run them from the repository root.

```powershell
git clone https://github.com/Druv08/amazon-ml-entity-resolution.git
cd amazon-ml-entity-resolution
git checkout updated-p2-p3

py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On macOS/Linux, create and activate the environment with:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Extract the untouched official package into the exact paths documented in [data/README.md](data/README.md). Do not
use `data/clean/` for the production pipeline.

Restore the trained production bundle at:

```text
output/candidates_p3/hybrid_meta.pkl
```

The production bundle is intentionally ignored by Git. It contains both trained K=20/K=50 M3 matchers, their
TF-IDF state and A+C+E+T feature configuration, the meta-model, thresholds, and Hybrid50 configuration. No separate
`matcher.pkl` files are required by the default meta-decoder. See
[docs/fresh_clone_setup.md](docs/fresh_clone_setup.md) for transfer integrity, rebuild instructions, and a complete
dependency classification.

Check the machine before starting the long runs:

```powershell
python scripts/check_environment.py
```

Generate the two candidate runs. They share the rebuildable cache under `data/processed/blocking/`:

```powershell
python -m src.blocking.generate_candidates --split test --top-k 20 --out-dir output/candidates_k20 --no-tsv
python -m src.blocking.generate_candidates --split test --top-k 50 --out-dir output/candidates_k50 --no-tsv
```

Run the adopted meta-decoder. It writes both official files from the Hybrid50 candidate set:

```powershell
python -m src.pipeline.predict_hybrid
```

Validate the result:

```powershell
python resources/utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir data/raw/test
```

The commands are identical in bash; only virtual-environment activation differs.

## Rebuilding the production bundle

Transferring `hybrid_meta.pkl` is the fastest and safest handoff. If it cannot be transferred, it can be rebuilt
from the official training data and tracked code:

```powershell
python -m src.matching.sample_candidates --top-k 20
python -m src.matching.train --cands output/candidates_p3/k20
python -m src.matching.sample_candidates --top-k 50
python -m src.matching.train --cands output/candidates_p3/k50
python -m src.pipeline.train_meta
```

These are training runs, not prerequisites when the production bundle has been restored.

## Repository layout

```text
data/               official local data and rebuildable caches (ignored except data/README.md)
src/blocking/       loading, tokenisation, transliteration, K20/K50 candidate generation, CandidateStore
src/matching/       M3 features A+C+E+T, training, streaming inference, global exclusivity
src/pipeline/       Hybrid50 construction, meta-decoder training, production prediction
src/evaluation/     validation and analysis utilities
resources/utils/    official submission validator
scripts/            fresh-clone preflight checks
tests/              synthetic/unit tests that do not require the full dataset
output/             generated candidates, models, and official TSVs (ignored)
```

Detailed implementation notes are in [docs/blocking_handoff.md](docs/blocking_handoff.md) and
[docs/p3_matching.md](docs/p3_matching.md).

## Development checks

```powershell
python -m unittest discover -s tests -t .
python -m src.blocking.generate_candidates --help
python -m src.matching.predict --help
python -m src.pipeline.predict_hybrid --help
python resources/utils/validate_submission.py --help
```
