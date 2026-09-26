# Amazon ML Entity Resolution

Solution workspace for the **Amazon ML Challenge 2026 — Business Entity Resolution** problem.

Given business records (name, address, country) from three independent sources, the system will
determine which Source 2 and Source 3 records correspond to each Source 1 business. Source 1 is the
deduplicated reference source; each Source 1 entity may match zero, one, or many records.
Submissions are scored with macro-averaged F0.5.

See [docs/problem_statement.md](docs/problem_statement.md) for the full official problem statement.

## Project Pipeline

```
Data Preparation
  → Candidate Generation / Blocking
  → Feature Engineering & Matching
  → Evaluation
  → Final Predictions
```

## Repository Structure

```
data/               local challenge data (raw data is git-ignored)
src/
  preprocessing/    dataset preparation / normalization
  blocking/         candidate generation / blocking
  matching/         feature engineering / ML matching
  evaluation/       validation / F0.5 / error analysis
  pipeline/         end-to-end integration
notebooks/          exploration
tests/              tests
output/             generated submission files (git-ignored)
resources/utils/    official organizer utilities (validate_submission.py)
docs/               problem statement and documentation template
```

## Data

Raw challenge data is intentionally **not** stored in this repository.
See [data/README.md](data/README.md) for how to set it up locally.

## Setup

Python 3.13 with `pip install -r requirements.txt`.

## Validate a submission

The official problem statement refers to `utils/validate_submission.py`; in this repo it lives under
`resources/utils/`:

```
python resources/utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir data/raw/test
```

## Status

- Candidate generation / blocking (`src/blocking`): implemented. See
  [docs/blocking_handoff.md](docs/blocking_handoff.md) for how to generate and consume
  candidates, and [docs/blocking_data_analysis.md](docs/blocking_data_analysis.md) for the analysis
  and measured results.
- Matching (`src/matching`): implemented. See [docs/p3_matching.md](docs/p3_matching.md).
- Evaluation and final pipeline: not implemented yet.

Run the tests with `python -m unittest discover -s tests -t .`
