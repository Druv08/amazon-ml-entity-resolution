# Final internal holdout (untouched)

> **DO NOT use this holdout for feature selection, K selection, threshold tuning, model selection or any
> repeated development evaluation.** Score it **once**, after the model and all settings are frozen. Looking at
> its macro-F0.5 earlier turns it into another development set.

The development samples (P2's blocking validation sample and P3's matcher sample) have been used repeatedly
for features, K and thresholds, so their scores are optimistic. This holdout gives one unbiased final estimate.

## Selection

`src/evaluation/holdout.py`, function `select_holdout(s1)`:

- **Hash:** a training S1 is selected when `hash_fraction(s1_id, "final-holdout-v1") < 0.0135`.
  `hash_fraction` is the md5-based function in `src/blocking/sampling.py`. It is deterministic and does not depend
  on file order.
- **Salt:** `final-holdout-v1`, which is new and used nowhere else in the repository.
- **Exclusions:** every S1 in an earlier development sample, applied explicitly (not left to chance):

  | Sample | Definition | Tuned on it |
  |---|---|---|
  | P2 blocking validation | salt `blocking-validation-v1`, rate 0.0025 (5,509 S1) | blocker configuration |
  | P3 random development | salt `p3-matcher-v1`, rate 0.009 | features, K, threshold |
  | P3 dense city sample | every S1 in Bhopal (India) / Tucson (US) | P3 training |

## Result on the training data

`python -m src.evaluation.holdout` writes the IDs to `output/holdout/final_holdout_s1.parquet` (git-ignored) and
prints the overlap proof. It reads only the S1 file, never ground truth or model output.

| | S1 |
|---|---|
| holdout | **29,314** of 2,206,821 |
| US / India | 17,558 / 11,756 |
| overlap with P2 blocking validation | **0** |
| overlap with P3 random development | **0** |
| overlap with P3 city sample | **0** |
| overlap with the saved P3 training list (`output/candidates_p3/sample_s1.parquet`, 29,169 S1) | **0** |

`tests/test_holdout.py` checks determinism, order independence, the expected size, unique IDs, both countries,
and zero overlap. The overlap check runs on the real training file when it is available.

## Using it (once, at the end)

1. Freeze the model, K and threshold using the development samples only.
2. Generate candidates for the holdout S1s with the frozen K, score them with the frozen model, and compute
   macro-F0.5 against `train_ground_truth.tsv`.
3. Report that number as the final internal estimate. Do not iterate on it.

Status: the holdout has been defined, but **no model has been evaluated on it.**
