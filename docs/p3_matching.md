# P3: Matching Model (input for Documentation_template.md, sections 2.1, 4, 5)

Code: `src/matching/matcher.py` (features, model, decision rule, metric), `src/matching/sample_candidates.py` (P2's blocker on the training sample), `src/matching/train.py` (train and validate), `src/matching/predict.py` (test inference to `matching_results.tsv`).

## Interfaces

**From P2 (blocking):** `src.blocking.handoff.CandidateStore` frames (`iter_frames(with_records=True)`, plus `with_labels=True` on train). `matcher.from_store()` turns a frame into the matcher's `pairs` (`s1_id`, `cand_id`, `block_score` = P2's `score`, `rank`, `name_score`) and `records` (`entity_id`, `business_name`, `business_address`, `country`). The store keeps `"NULL"`/`"nan"` as text. `from_store` maps them to missing values.
- **Train:** `python -m src.matching.sample_candidates --top-k K` runs P2's engine with P2's default config, but only on the P3 training sample. It writes a `CandidateStore("train", out_dir="output/candidates_p3/kK")`. P2's blocker scores every S1 independently (output does not depend on shard size, blocking_data_analysis.md §10.6), so these are exactly the pairs the full `generate_candidates --split train` run gives the same S1. Pair recall on the sample at K=100 is 97.01%; P2 measured 97.05%.
- **Test:** `python -m src.blocking.generate_candidates --split test --top-k K` → `output/candidates/test` and the official `output/candidate_pairs.tsv`. **K must equal the model's K.** `predict.py` refuses to run otherwise, because the rank, gap and cross-entity features depend on the length of the candidate list.

**From P1 (cleaning):** records with cleaned `business_name` / `business_address` can replace the ones `from_store` builds. `country` stays the raw label (France is unseen in train, and nothing one-hots it).

**To P4 (evaluation and submission):**
- `output/candidates_p3/kK/oof.parquet`: `s1_id`, `cand_id`, `rank`, `label`, `prob` (out-of-fold, GroupKFold by S1).
- `src.matching.matcher.macro_f05(pred, truth)`, where both arguments are `{s1_id: set(ids)}` and `truth` covers every S1 (empty set = singleton). **Import it, don't reimplement it**, so every number in the report comes from the same metric.
- `output/matching_results.tsv` from `src.matching.predict`. Every match is in P2's candidate list, and each S2/S3 record is used at most once. `output/candidate_pairs.tsv` is P2's file.

**Retrain rule:** the features (rank, gap, cross-entity, P2 scores), the TF-IDF vocabulary and the threshold all depend on the candidate distribution (including K) and on the text. Whenever P2's blocking, K or P1's cleaning changes, re-run `sample_candidates` + `train` and use the new `matcher.pkl` (model, threshold, TF-IDF, K). Never reuse a `matcher.pkl` across pipeline versions.

## 2.1 EDA findings that shaped the matcher

- Train has 2.2M S1 records and about 10.3M S2+S3 records. 5.6% of S1 are singletons; the mean is about 3.5 matches per S1, and the maximum is 11.
- Each S2/S3 record belongs to at most one S1. About 26% of S2/S3 records match no S1 at all, so they only act as distractors.
- Names:
  - native-script names: Hindi, Kannada, Telugu, Punjabi (9–11% of S2 records)
  - accents, and prefixes such as `***`, `The`, `Sri`, `Dr`
  - `formerly X`, `DBA: X`, brackets, domain-style names (`nelsonsmetrohealth.com`)
  - OCR-like typos (`Appare1s`) and word-order shuffles
- Addresses:
  - state names in native script, and state abbreviations (`MH` / `Maharashtra`)
  - reordered components, 3% missing or `NULL`
  - house numbers that differ even between true matches (`5425` vs `1425`)
- France appears only in test, so no feature one-hots `country`.

## 4. Matching model

**Unit:** one (S1, candidate) pair from P2's blocking, labelled 1 if the candidate is in that S1's ground-truth list. Training uses only blocked candidates, so the training distribution matches inference. True pairs the blocker missed (5.1% at K=20) cannot be predicted. They are left out of pair-level metrics but count as misses in macro-F0.5.

**Features (47):**
- Name and address, each: Levenshtein ratio, Jaro-Winkler, token-sort, token-set and partial ratio, token Jaccard, character 2–4-gram TF-IDF cosine (vectorizers fitted once on the training texts and stored in `matcher.pkl`, so a pair scores the same in training and in any test shard), length difference, missing flag, non-Latin script share.
- Core name, after removing legal forms and honorifics (including French SARL/SAS): token-sort ratio, Jaccard, and ratio and partial ratio of the joined name (catches domain-style names).
- Address numbers (leading zeros stripped): Jaccard, "both have numbers but none shared" flag, first-number equality.
- P2's blocker outputs: `score` (as `block_score`), `rank`, `name_score`.
- Context: same country, S2 vs S3.
- Rank within the S1's candidates: gap to the best score and rank position, for name TF-IDF, name token-sort, core token-sort, address TF-IDF, block score and name score. This lets the model reject the weaker of two lookalike candidates.
- Cross-entity, for name TF-IDF and block score: this S1's score minus the best score any *other* S1 gives the same candidate (0 if no other S1 has it). It targets a different business at the same address. `predict.py` computes it in two passes over P2's shards (pass 1 keeps each candidate's top-2 scores over all test S1), so a competing S1 in another shard still counts, as it does in training.

**Model:** sklearn `HistGradientBoostingClassifier` (BSD-3; gradient-boosted trees, same family as LightGBM). Settings: 500 iterations, learning rate 0.05, 31 leaves, early stopping.

**Decision rule:**
1. **Exclusivity:** each S2/S3 record may go only to the S1 that scores it highest (it belongs to at most one S1 in the ground truth).
2. **Threshold:** a pair counts as a match if its probability is at least **0.65** (K=20). The threshold was chosen by maximising macro-F0.5 on out-of-fold predictions after exclusivity (grid 0.05–0.98). The optimum is flat: 0.60–0.75 are all within 0.0007 of the best.

**Validation:** 5-fold GroupKFold by S1 entity, so no S1 appears in both train and validation folds. The metric is the official macro-F0.5 over all S1, including singletons and S1 whose true matches the blocker missed. The sample has 29,169 train S1:
- 19,819 hash-random S1 (`md5("p3-matcher-v1:" + id) < 0.009`). Every headline number below is on these.
- every S1 in Bhopal (India) and Tucson (US), 9,350 in total. Keeping a city together scores competing S1 together, so exclusivity and the cross-entity features can be measured ("cities" below).
- P2's blocking-validation hash sample is excluded, because P2 tuned its blocker on it.

## 5. Results (P2's final blocker, 76bc657)

### K comparison

The sample S1 are the same for every K. Each K is a separate P2 blocker run: the name and non-Latin quota slots scale with K, so K=30 is not the first 30 of the K=100 list. The matcher is retrained and its threshold re-tuned for each K. Singletons = share of singletons correctly left empty.

| K | Pair recall | Ceiling | **macro-F0.5** | Threshold | India | US | Singletons | Test pairs | Test inference* |
|---|---|---|---|---|---|---|---|---|---|
| 5 | 67.32% | 0.9059 | 0.8719 | 0.59 | 0.8574 | 0.8815 | 0.892 | 8.7M | ~0.5 h |
| 10 | 91.20% | 0.9718 | 0.9328 | 0.69 | 0.9136 | 0.9457 | 0.907 | 17.3M | ~1 h |
| 15 | 94.09% | 0.9790 | 0.9393 | 0.70 | 0.9206 | 0.9519 | 0.908 | 26.0M | ~1.4 h |
| **20** | **94.87%** | **0.9815** | **0.9407** | **0.65** | **0.9224** | **0.9529** | **0.895** | **34.7M** | **~1.9 h** |
| 30 | 95.70% | 0.9847 | 0.9385 | 0.64 | 0.9201 | 0.9508 | 0.875 | 52.0M | ~2.9 h |
| 50 | 96.44% | 0.9874 | 0.9370 | 0.70 | 0.9195 | 0.9487 | 0.876 | 86.6M | ~4.8 h |
| 100 | 97.01% | 0.9897 | 0.9328 | 0.69 | 0.9140 | 0.9453 | 0.859 | 173.3M | ~9.6 h |

\*Matcher only, single process, at the ~200 s per 1M pairs measured on a 3,000-S1 test smoke run (P2's blocker time is extra).

Paired differences on the same 19,819 S1 (bootstrap, 2,000 resamples):

| | Δ macro-F0.5 | 95% CI |
|---|---|---|
| K=20 − K=15 | +0.0014 | [+0.0004, +0.0023] |
| K=20 − K=30 | +0.0022 | [+0.0012, +0.0032] |
| K=20 − K=50 | +0.0037 | [+0.0024, +0.0050] |
| K=20 − K=100 | +0.0079 | [+0.0066, +0.0093] |

**Recommended K: 20, threshold 0.65.** This supersedes the provisional K=30 / 0.64 from the stand-in blocker.
- **Above K=20, extra candidates cost more than they add.** K=100 recovers 2.1 points of pair recall, but it has 5.8× the negatives, and false positives on the sample rise from 1,895 to 2,328 (+23%).
- **Singletons get worse with larger K:** 89.5% are correctly left empty at K=20, 85.9% at K=100. Each wrong singleton scores 0.
- **Below K=15, lost recall dominates.** At K=5 the quota slots use 3 of the 5 places.
- K=20 is also a 5× smaller candidate file than K=100, and the organisers rank smaller candidate sets higher.

### P2 features (`rank`, `score`, `name_score`)

| Features | K=20 | K=100 |
|---|---|---|
| no P2 features | 0.9298 | 0.9226 |
| + `score` (with its gap/rank/cross-entity) | 0.9342 | 0.9276 |
| **+ `rank`, `name_score` (with its gap/rank): final** | **0.9407** | **0.9328** |

All three help, most in India: 0.9048 → 0.9224 at K=20.

### Final model (K=20, threshold 0.65, 19,819 random S1)

| | macro-F0.5 | Ceiling |
|---|---|---|
| All | **0.9407** | 0.9815 |
| India (7,945 S1) | 0.9224 | 0.9703 |
| US (11,874 S1) | 0.9529 | 0.9891 |
| Singletons (1,060): correctly empty | 89.5% | |
| Matched S1 (18,759) | 0.9433 | |

- Pair AUC is 0.9986.
- Cities sample: 0.9504 with exclusivity and 0.9501 without. With P2's candidates, exclusivity barely changes the score. At test time every S1 is present, so it can act more often than in a random sample.

**Hard negatives** (K=20; FP = accepted negative after exclusivity):

| Negative pairs | Pairs | Accepted | Share of all FPs |
|---|---|---|---|
| all | 487,129 | 0.39% | 100% |
| blocker rank 1–3 | 21,631 | 4.09% | 47% |
| lookalike name (token-sort ≥ 0.9) | 32,955 | 1.85% | 32% |
| record that is another S1's true match | 345,614 | 0.18% | 33% |

- The blocker's own top-3 wrong candidates are the hardest group. They cause about half of all false positives, at every K from 10 to 100 (44–52%).
- A third of FPs are records that truly belong to a different S1. These are the same brand at another branch, or a different business at the same address. At test time exclusivity can drop them when the true owner scores higher.
- The remaining gap to the ceiling is 0.041. India's per-S1 gap is larger (0.048, vs 0.036 in the US), but the US has more S1, so both countries lose about the same total.

### Error analysis (K=20 OOF, development sample)

Reproduce: `python -m src.evaluation.matcher_errors --cands output/candidates_p3/k20`. It uses the production rule
(exclusivity plus threshold 0.65) and writes aggregate tables to `output/error_analysis/` (git-ignored, no raw
records).

Over 583,363 pairs and 29,169 S1:

| Outcome | Pairs |
|---|---|
| TP | 90,440 |
| FP | 1,895 |
| FN1: not among the K candidates | 5,200 |
| FN2: below threshold | 5,789 |
| FN3: lost to exclusivity | 5 |

152 of the 1,598 singleton S1 get at least one match.

| Finding | Evidence |
|---|---|
| Rank 1–3 FPs are mostly *same address, different business* | 885 FPs (47%). Acceptance among negatives: 7.2% at rank 1, 4.6% at rank 2, 2.9% at rank 3 (0.39% overall). These FPs have high address similarity (addr_tsort 0.84–0.89) but lower name similarity (0.63–0.67). 46% of all FPs share the first house number. |
| House-number disagreement costs recall | 39% of true pairs with fully conflicting numbers are missed, and 21% of those whose first numbers differ (6% overall). These pairs still have high name (0.78) and address (0.82) similarity, which points to typos in numbers. |
| Name-only candidates are ambiguous | Candidates without an address produce 327 FPs (near-identical names, often another branch of a chain). 40% of true name-only candidates are missed (1,200 FN2). |
| Indian-script candidates | 12% of FPs and 10% of FN2. Name similarity for them is about 0.12–0.16, because the matcher cannot compare scripts. FN2 rate for these true pairs is 8.5% (6.0% overall). |
| Claimed by another S1 | 33% of FPs, but the owner S1 is in the development sample for only 2.1% of them. Development data cannot teach cross-S1 ownership features; exclusivity handles it once every S1 is scored. |
| Blocking loss | 5,200 true pairs (5.1%) are not among the K=20 candidates; 63% of them are India. |

### Targeted features (adopted: groups A + C + E)

Each feature group targets one error class above (`src/matching/extra_features.py`). All of them are computed
identically for train and test from raw data only, with no external data.

| Group | Features | Motivation |
|---|---|---|
| A: address numbers | `num_shared_n`, `num_only_s1_n`, `num_only_cand_n`, `num_best_ratio` (typo-tolerant best number match), `num_prefix`, `addr_word_jacc` | missed true pairs with conflicting / typo'd house numbers |
| C: transliteration | `ph_jacc` and `ph_bigram_hit` (P2's offline phonetic keys), `latin_tsort` (after P2 transliteration), `script_mismatch` | Indian-script candidates |
| D: blocker evidence | `block_addr_score` = block_score − name_score | tested, **not adopted** |
| E: name rarity | `s1_core_freq`, `cand_core_freq` (core-name frequency in the split), `core_idf_jacc`, `core_rarest_shared_idf` | chains / branches, and same-address neighbours that share only generic words |

About group E:
- Its statistics come from the scored split's own three source files (`NameStats`), so training uses the train
  files and inference uses the test files.
- The IDFs do not depend on split size. The core-name counts are raw counts; the test split is about 22% smaller.

Setup: K=20 development sample, same 5 GroupKFold folds, unweighted model, threshold retuned per variant, deltas
paired against base with the same seed.
Reproduce: `python -m src.matching.feature_experiments --variants base A C D E` and
`--variants base A+C+E C+E A+E A+C --seeds 0 1`.

| Variant | Features | macro-F0.5 | Δ vs base | India | US | Singletons empty | AUC | FPs | Rank 1–3 FPs | Lookalike FPs | Claimed FPs | Threshold |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base | 47 | 0.9407 | – | 0.9224 | 0.9529 | 0.895 | 0.9986 | 1,895 | 885 | 611 | 629 | 0.65 |
| +A | 53 | 0.9428 | +0.0021 ± 0.0006 | 0.9250 | 0.9547 | 0.905 | 0.9987 | 1,598 | 772 | 514 | 495 | 0.70 |
| +C | 51 | 0.9443 | +0.0036 ± 0.0005 | 0.9282 | 0.9550 | 0.915 | 0.9988 | 1,689 | 727 | 601 | 569 | 0.65 |
| +D | 48 | 0.9400 | −0.0007 ± 0.0004 | 0.9219 | 0.9521 | 0.892 | 0.9986 | 1,893 | 884 | 622 | 606 | 0.65 |
| +E | 51 | 0.9447 | +0.0041 ± 0.0006 | 0.9277 | 0.9562 | 0.912 | 0.9989 | 1,424 | 728 | 388 | 328 | 0.68 |
| +A+C | 57 | 0.9469 | +0.0062 ± 0.0006 | 0.9317 | 0.9571 | 0.928 | 0.9990 | 1,332 | 611 | 493 | 434 | 0.71 |
| +A+E | 57 | 0.9472 | +0.0065 ± 0.0007 | 0.9295 | 0.9591 | 0.933 | 0.9990 | 1,040 | 559 | 295 | 219 | 0.76 |
| +C+E | 55 | 0.9480 | +0.0073 ± 0.0007 | 0.9328 | 0.9581 | 0.923 | 0.9990 | 1,336 | 637 | 414 | 350 | 0.67 |
| **+A+C+E** | **61** | **0.9494** | **+0.0087 ± 0.0007** | **0.9335** | **0.9601** | **0.929** | **0.9991** | **1,216** | **612** | **353** | **289** | 0.70 |
| base, seed 1 | 47 | 0.9393 | – | 0.9211 | 0.9515 | 0.883 | 0.9986 | 1,907 | 904 | 607 | 593 | 0.65 |
| +A+C+E, seed 1 | 61 | 0.9501 | +0.0108 ± 0.0007 | 0.9346 | 0.9605 | 0.932 | 0.9991 | 1,227 | 610 | 366 | 289 | 0.69 |

The gain is about 9–15 paired standard errors and holds under a second seed, far above the ~0.001 noise floor.
India, US and singletons all improve, and FPs fall by 36%.

Production: `src.matching.train` defaults to `--groups ACE` and stores `feature_groups` in `matcher.pkl`.
The retrained model reproduces the table exactly: macro-F0.5 0.9494, India 0.9335, US 0.9601, singletons 0.929,
AUC 0.9991, threshold 0.70.

Inference (`predict.py`) computes group E's statistics from the test split's own source files, adding about 2 min
and about 0.9 GB once, independent of K. The real 5k-S1 smoke run peaks at 4.6 GB and is byte-identical to the
in-memory reference. Group D is not used. Ownership features (family B) were
not built, because the error analysis showed the development sample cannot train them.

### K=20 vs K=50 with the improved features

K=50 candidates were generated for the same 29,169 development S1 (`python -m src.matching.sample_candidates
--top-k 50`, 94 s). The A+C+E matcher was then retrained from scratch at K=50
(`python -m src.matching.train --cands output/candidates_p3/k50`, 621 s, versus 472 s at K=20). All three runs use
the production rule on the 19,819 random development S1.
Reproduce: `python -m src.evaluation.k_compare base20=output/candidates_p3/k20/oof_baseline47.parquet:0.65
k20=output/candidates_p3/k20 k50=output/candidates_p3/k50`.

| Configuration | Pair recall | Ceiling | macro-F0.5 | India | US | Singletons empty | Matched S1 | AUC | FPs | Rank 1–3 FPs | Threshold |
|---|---|---|---|---|---|---|---|---|---|---|---|
| old K=20 baseline (47 features) | 0.9493 | 0.9815 | 0.9407 | 0.9224 | 0.9529 | 0.895 | 0.9433 | 0.9986 | 1,895 | 885 | 0.65 |
| improved K=20 (61 features) | 0.9493 | 0.9815 | **0.9494** | 0.9335 | 0.9601 | 0.929 | 0.9506 | 0.9991 | 1,216 | 612 | 0.70 |
| improved K=50 (61 features) | 0.9654 | 0.9874 | 0.9492 | 0.9343 | 0.9591 | 0.918 | 0.9509 | 0.9994 | 1,435 | 664 | 0.72 |

- **K=50 ties K=20:** −0.0002, inside the ~0.001 noise.
- **The larger ceiling (+0.006) is not converted into score.** K=50 accepts 1,572 of the 2,815 new rank 21–50 true
  pairs (55.8%) with only 140 FPs among them. But it loses 1,242 true positives and adds 79 FPs among ranks ≤ 20,
  because the model and threshold (0.72) are stricter with 2.5× more negatives per S1.
- **Inference cost scales with K.** Full test inference is estimated at about 5.5 h for K=20 (34.7M pairs) and
  13.7 h for K=50 (86.6M pairs), single process. This is extrapolated from the 5k-S1 smoke run at ~0.57 ms per pair
  plus ~2.5 min fixed, not measured.

### Hybrid candidate set: exact K=20 plus deep K=50 candidates

`src/pipeline/hybrid.py` keeps every candidate of the K=20 run, in K=20 order, and fills up to 50 with K=50-only
candidates in K=50 rank order. P2 scales its reserved slots with K, so K=20 is not a prefix of K=50. The set is
deterministic, has no duplicates, and can never lose a K=20 candidate.
Reproduce: `python -m src.pipeline.hybrid --base output/candidates_p3/k20 --deep output/candidates_p3/k50`.

Overlap on the 29,169 development S1:

| | Candidates | Per S1 | True |
|---|---|---|---|
| K20 only | 277 | 0.01 | 0 |
| both | 583,086 | 19.99 | 96,234 |
| K50 only | 875,317 | 30.01 | 1,591 (India 952, US 639) |

In practice K=50 contains the K=20 list. The deep pool adds 30 candidates per S1 at a 0.18% positive rate.

Blocking on the random development S1:

| Set | Pair recall | Ceiling | Per S1 | All true retained | India recall | US recall |
|---|---|---|---|---|---|---|
| K20 | 0.9493 | 0.9815 | 20.0 | 85.3% | 0.9223 | 0.9676 |
| K50 | 0.9654 | 0.9874 | 50.0 | 89.8% | 0.9448 | 0.9794 |
| HYBRID50 | 0.9654 | 0.9874 | 50.0 | 89.8% | 0.9448 | 0.9794 |

The hybrid adds 1,112 true pairs beyond K=20 (random S1) and keeps all K=20 true pairs. If the hybrid is adopted,
`candidate_pairs.tsv` must list the hybrid set.

### S1-level no-match gate (negative result, not adopted)

The gate is a second classifier on S1-level signals, built only from the S1's own candidates: the top-1 and top-2
probabilities and their gap, how many candidates clear 0.3/0.5/0.7/0.9, and the best block score, name and
address evidence.
- **Training:** out of fold, with the pair model's S1 folds.
- **Use:** applied after exclusivity plus threshold, suppressing all matches of S1 it calls "no match".
- **Tuning:** its threshold is tuned on the out-of-fold gate probabilities.

Reproduce: `python -m src.matching.s1_gate --cands output/candidates_p3/k20 --seeds 0 1`.

| Configuration | macro-F0.5 | Δ | Singletons empty | Matched-S1 F0.5 | India | US | S1 suppressed (true singletons / matched) |
|---|---|---|---|---|---|---|---|
| pair model (A+C+E), threshold 0.70 | 0.9494 | – | 0.929 | 0.9506 | 0.9335 | 0.9601 | – |
| + gate, seed 0 (gate threshold 0.68) | 0.9495 | +0.0001 ± 0.0001 | 0.930 | 0.9506 | 0.9335 | 0.9602 | 1 (1 / 0) |
| + gate, seed 1 (gate threshold 0.68) | 0.9495 | +0.0001 ± 0.0001 | 0.931 | 0.9505 | 0.9335 | 0.9602 | 3 (2 / 1) |

Why it cannot help:
- After the targeted features, only 101 of the 27,375 S1 that output a match are true singletons (0.37%).
- The gate ranks them well (AUC 0.968), but its precision is at most about 0.37.
- Suppressing a matched S1 costs about 0.95 while suppressing a singleton gains 1.0, so the break-even precision is
  about 0.49.

The remaining singleton errors look like matched S1 at the entity level too.

### Hard-negative weighting (negative result, not adopted)

The experiment up-weights the training negatives the blocker ranks 1–3 (21,631 pairs, about 47% of FPs). Everything else is held fixed: the same 47 features, the same 5 GroupKFold folds, and a threshold retuned per configuration on its own OOF predictions. Scores are on the 19,819 random development S1; the final holdout was not used.

Reproduce: `python -m src.matching.hard_negatives --cands output/candidates_p3/k20 --weights 1 1.5 2 3`

| Configuration | macro-F0.5 | Δ vs control (paired, ±SE) | India | US | singletons empty | AUC | FPs | rank 1–3 FPs | lookalike FPs | claimed FPs | threshold |
|---|---|---|---|---|---|---|---|---|---|---|---|
| no weights (`train.py`) | 0.9407 | – | 0.9224 | 0.9529 | 0.895 | 0.9986 | – | – | – | – | 0.65 |
| unit weights (control) | 0.9396 | 0 | 0.9216 | 0.9517 | 0.908 | 0.9986 | 1,557 | 760 | 499 | 457 | 0.71 |
| unit weights, seed 1 | 0.9406 | +0.0009 ± 0.0004 | 0.9237 | 0.9519 | 0.912 | 0.9986 | 1,560 | 761 | 510 | 464 | 0.70 |
| hard negatives ×1.5 | 0.9394 | −0.0002 ± 0.0004 | 0.9216 | 0.9513 | 0.904 | 0.9986 | 1,798 | 767 | 618 | 606 | 0.64 |
| hard negatives ×2.0 | 0.9397 | +0.0001 ± 0.0004 | 0.9212 | 0.9521 | 0.910 | 0.9986 | 1,859 | 700 | 644 | 664 | 0.61 |
| hard negatives ×3.0 | 0.9391 | −0.0006 ± 0.0005 | 0.9214 | 0.9509 | 0.913 | 0.9986 | 1,935 | 629 | 713 | 733 | 0.57 |

- **Unit weights differ from no weights (0.9396 vs 0.9407), and this is expected.** With more than 200k training
  rows, scikit-learn's histogram binning subsamples rows with `rng.choice(..., p=w/Σw)` when weights are passed and
  `p=None` otherwise. The bin edges therefore differ: statistically equivalent, but not bit-identical.
  - The three unweighted runs span **0.9396–0.9407**. That is the training-noise floor, about ±0.001.
  - A different seed alone moves the score by 2.25 paired SE, so the paired SE understates the real noise.
- **No weight improves macro-F0.5 beyond that noise.** Heavier weights do cut rank 1–3 FPs (760 → 629 at ×3).
  But the retuned threshold drops (0.71 → 0.57), and lookalike and claimed-by-another-S1 FPs grow. Total FPs rise
  1,557 → 1,935, singletons move by less than 0.5 pp, and there is no consistent India or US gain.
- **Decision:** production training stays unweighted (`train()` defaults to `sample_weight=None`). The experiment
  script is kept for later feature work on the same FP groups.

## Reproduce

```
python -m src.matching.sample_candidates --top-k 20       # P2's blocker on the 29,169-S1 training sample (~4 min, 2 workers)
python -m src.matching.train --cands output/candidates_p3/k20      # ~7 min; add --variants "" "^(rank|name_score)" for the ablation
python -m src.blocking.generate_candidates --split test --top-k 20   # P2: output/candidates/test + output/candidate_pairs.tsv
python -m src.matching.predict --model output/candidates_p3/k20/matcher.pkl   # streams shards; memory ~flat in #pairs
python resources/utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir data/raw/test
```

Previous results with the stand-in token blocker (K=30, macro-F0.5 0.8877) are in git history (`docs/p3_matching.md` at e5f18fb).
