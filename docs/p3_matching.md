# P3: Matching Model (input for Documentation_template.md, sections 2.1, 4, 5)

Code: `src/matching/matcher.py` (features, model, decision rule, metric), `src/matching/sample_candidates.py` (P2's blocker on the training sample), `src/matching/train.py` (train and validate), `src/matching/predict.py` (test inference to `matching_results.tsv`).

## Interfaces

**From P2 (blocking):** `src.blocking.handoff.CandidateStore` frames (`iter_frames(with_records=True)`, plus `with_labels=True` on train). `matcher.from_store()` turns a frame into the matcher's `pairs` (`s1_id`, `cand_id`, `block_score` = P2's `score`, `rank`, `name_score`) and `records` (`entity_id`, `business_name`, `business_address`, `country`). The store keeps `"NULL"`/`"nan"` as text. `from_store` maps them to missing values.
- **Train:** `python -m src.matching.sample_candidates --top-k K` runs P2's engine with P2's default config, but only on the P3 training sample. It writes a `CandidateStore("train", out_dir="output/candidates_p3/kK")`. P2's blocker scores every S1 independently (output does not depend on shard size, blocking_data_analysis.md §10.6), so these are exactly the pairs the full `generate_candidates --split train` run gives the same S1. Pair recall on the sample at K=100 is 97.01%; P2 measured 97.05%.
- **Test:** `python -m src.blocking.generate_candidates --split test --top-k K` → `output/candidates/test` and the official `output/candidate_pairs.tsv`. **K must equal the model's K.** `predict.py` refuses to run otherwise, because the rank, gap and cross-entity features depend on the length of the candidate list.
- **Test, adopted Hybrid50:** a K=20 and a K=50 run (`--no-tsv`, separate `--out-dir`), then `python -m src.pipeline.predict_hybrid`, which writes both official files (see "Production Hybrid50 inference").

**From P1 (cleaning):** records with cleaned `business_name` / `business_address` can replace the ones `from_store` builds. `country` stays the raw label (France is unseen in train, and nothing one-hots it).

**To P4 (evaluation and submission):**
- `output/candidates_p3/kK/oof.parquet`: `s1_id`, `cand_id`, `rank`, `label`, `prob` (out-of-fold, GroupKFold by S1).
- `src.matching.matcher.macro_f05(pred, truth)`, where both arguments are `{s1_id: set(ids)}` and `truth` covers every S1 (empty set = singleton). **Import it, don't reimplement it**, so every number in the report comes from the same metric.
- `output/matching_results.tsv` and `output/candidate_pairs.tsv` from `src.pipeline.predict_hybrid` (adopted Hybrid50), or `matching_results.tsv` from `src.matching.predict` with P2's own `candidate_pairs.tsv` (single K). Either way every match is in the candidate list, and each S2/S3 record is used at most once.

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

### Deep recovery: K=20 decisions plus confident deep matches (adopted rule: deep prob ≥ 0.85)

`src/pipeline/deep_recovery.py` works in three steps:
1. **Base:** the improved K=20 decisions (exclusivity plus threshold 0.70), which are never changed.
2. **Deep candidates:** the hybrid rows that are not in the K=20 list, scored with the K=50 matcher's OOF probability.
3. **Merge:** add deep candidates that pass the rule, with global exclusivity. The K=20 base keeps priority (a
   candidate it gave to another S1 is never added), and deep claimants are ranked by highest probability, then the
   smallest `s1_id`.

Reproduce: `python -m src.pipeline.deep_recovery`.

**The deep pool** has 875,040 rows with 1,591 true (0.18%); on the random S1 it is 594,540 rows with 1,112 true.
Deep true pairs have a median probability of 0.86, while 99% of false ones are below 0.012.

| Trait of deep candidates | True | False |
|---|---|---|
| median K=50 rank | 27 | 32 |
| India | 60% | 43% |
| candidate address missing | 25.5% | 10.2% |
| strong house-number conflict | 1.4% | 42% |
| name token-sort | 0.70 | 0.47 |
| phonetic Jaccard | 0.63 | 0.29 |
| S1 already has a K=20 match | 94.5% | 93.9% |

66,652 false deep candidates (and no true ones) are candidates the K=20 base already gave to another S1, so
base-priority exclusivity blocks them correctly.

Results on the random development S1:

| Method | macro-F0.5 | Δ vs 0.9494 (paired ±SE) | India | US | Singletons empty | Added TP | Added FP | S1 improved / harmed |
|---|---|---|---|---|---|---|---|---|
| K=20 base | 0.9494 | – | 0.9335 | 0.9601 | 0.929 | – | – | – |
| deep prob ≥ 0.70 | 0.9524 | +0.0030 ± 0.0003 | 0.9388 | 0.9614 | 0.926 | 623 | 62 | 545 / 56 |
| **deep prob ≥ 0.85 (adopted)** | **0.9525** | **+0.0031 ± 0.0003** | **0.9390** | **0.9615** | **0.927** | **572** | **39** | 503 / 34 |
| deep prob ≥ 0.95 | 0.9518 | +0.0024 ± 0.0003 | 0.9379 | 0.9611 | 0.927 | 451 | 29 | 397 / 25 |
| deep prob ≥ 0.99 | 0.9508 | +0.0014 ± 0.0002 | 0.9363 | 0.9605 | 0.927 | 290 | 26 | 263 / 22 |
| rule B: ≥ 0.85, no strong number conflict | 0.9525 | +0.0031 ± 0.0003 | 0.9390 | 0.9615 | 0.927 | 567 | 31 | – |
| rules C / D / E (name, margin, name+address requirements) | ≤ 0.9523 | ≤ +0.0029 | | | | | | |

- **The gain is flat** (+0.0030 ± 0.0001) for any deep threshold from 0.66 to 0.86, so it does not hinge on the
  exact threshold. It is about 11 paired SE, far above the ~0.001 noise.
- **Rule B ties rule A** (+0.00307 vs +0.00306), and the other rules lose true pairs, so the plain threshold is
  adopted.
- **No deep classifier was built (checkpoint 10):** the simple threshold already works, and only about 39 deep FPs
  are left to remove.
- **Singletons drop slightly** (0.929 → 0.927), because 2 random-sample singletons receive a deep match. Matched S1
  gain more than that.

Production implications (implemented in `src/pipeline/predict_hybrid.py`, see "Production Hybrid50 inference"):
- Test inference needs both the K=20 run (base decisions) and a K=50 run for the deep probabilities.
- `candidate_pairs.tsv` must list the HYBRID set (`src/pipeline/hybrid.py`), since every predicted match must be a
  candidate.

### K=100 (development sample only): not worth it

K=100 candidates were generated for the same development S1 (99 s) and the A+C+E matcher was retrained at K=100
(1,027 s).

K=100 alone scores 0.9465: below K=20 (0.9494), with India 0.9306, US 0.9571, singletons 0.922 and threshold 0.75.
This is the same pattern as K=50.

The fair test is incremental: K=100 candidates not already in hybrid50, scored with the K=100 OOF probability and
added on top of the adopted hybrid50 result. The K=20 base and the K=50 deep layer keep exclusivity priority.

| Set (random development S1) | Pair recall | Ceiling | Candidates / S1 | All true retained |
|---|---|---|---|---|
| K20 | 0.9493 | 0.9815 | 20 | 85.3% |
| hybrid50 | 0.9654 | 0.9874 | 50 | 89.8% |
| hybrid100 | 0.9714 | 0.9897 | 100 | 91.5% |

The incremental pool has 1,458,307 rows with 576 true (0.04%); the median probability of those true pairs is 0.46.

| Added on top of hybrid50 (0.9525) | macro-F0.5 | Δ vs hybrid50 (paired ±SE) | India | US | Singletons | Added TP / FP |
|---|---|---|---|---|---|---|
| K=100 layer, prob ≥ 0.70 | 0.9529 | +0.0004 ± 0.0002 | 0.9398 | 0.9617 | 0.926 | 181 / 64 |
| K=100 layer, prob ≥ 0.85 (best) | 0.9530 | +0.0005 ± 0.0001 | 0.9400 | 0.9617 | 0.926 | 158 / 43 |
| K=100 layer, prob ≥ 0.95 | 0.9530 | +0.0005 ± 0.0001 | 0.9400 | 0.9616 | 0.926 | 126 / 31 |

Decision: **not adopted.**
- The best gain (+0.0005) is below the ~0.001 training-noise floor.
- Singletons dip slightly.
- It needs a K=100 run on test, roughly twice the K=50 cost.

Hybrid50 stays the candidate set.

### Model ensemble experiment (checkpoint 14): no ensemble; the 63-leaf model wins alone

`src/matching/ensemble_experiment.py` compares a small, **predefined** set of HGB matchers (no search). Everything
else is held fixed: the same A+C+E features, the same GroupKFold(5) partitions, the same development sample and the
same Hybrid50 decision.

| Model | Configuration |
|---|---|
| M0 production | lr 0.05, 31 leaves, max_iter 500 |
| M1 conservative | lr 0.03, 31 leaves, max_iter 800 |
| M2 simpler | lr 0.05, 15 leaves |
| M3 richer | lr 0.05, 63 leaves |

Setup details:
- Every model has its own OOF at K=20 and K=50. The regenerated M0 OOF is bit-identical to the stored production
  OOF, so training is deterministic.
- The K=20 threshold is re-tuned per model on development, as `train.py` does; the deep rule stays prob ≥ 0.85.
- Ensembles combine all four models by arithmetic mean, median, or mean log-odds.

Reproduce: `python -m src.matching.ensemble_experiment oof`, then `... eval`.

Hybrid50 results on the random development S1. Δ is paired against the adopted Hybrid50 (0.9525), with the ±SE
and a 95% bootstrap CI over S1:

| Model / ensemble | macro-F0.5 | Δ (±SE) [95% CI] | India | US | Singletons | FP | FN | Rank 1–3 FP | Native-script FN | t20 | K=20 alone | Predict s / 1M rows (K=20) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| M0 (production) | 0.9525 | – | 0.9390 | 0.9615 | 0.927 | 950 | 6,575 | 455 | 903 | 0.70 | 0.9494 | 9.4 |
| M1 | 0.9531 | +0.0006 ± 0.0004 [−0.0002, +0.0013] | 0.9400 | 0.9618 | 0.933 | 861 | 6,657 | 421 | 899 | 0.72 | 0.9497 | 17.0 |
| M2 | 0.9480 | −0.0045 ± 0.0005 | 0.9332 | 0.9579 | 0.905 | 1,153 | 6,823 | 574 | 917 | 0.68 | 0.9451 | 9.8 |
| **M3** | **0.9558** | **+0.0033 ± 0.0005 [+0.0024, +0.0042]** | **0.9430** | **0.9644** | **0.937** | **887** | **6,137** | **408** | **824** | 0.70 | **0.9525** | 8.9 |
| mean of 4 | 0.9537 | +0.0013 ± 0.0003 [+0.0006, +0.0019] | 0.9405 | 0.9626 | 0.936 | 857 | 6,574 | 421 | 885 | 0.70 | 0.9504 | 45.2 |
| median of 4 | 0.9535 | +0.0010 ± 0.0003 [+0.0004, +0.0015] | 0.9403 | 0.9622 | 0.931 | 893 | 6,534 | 437 | 882 | 0.70 | 0.9501 | 45.2 |
| mean log-odds of 4 | 0.9537 | +0.0012 ± 0.0003 [+0.0005, +0.0019] | 0.9406 | 0.9624 | 0.933 | 893 | 6,503 | 436 | 872 | 0.70 | 0.9503 | 45.2 |

FN counts include 2,382 true pairs that are not in the candidate set, the same for every model. Training time for
5 folds: K=20 110 / 214 / 140 / 125 s (M0–M3), K=50 102 / 245 / 164 / 312 s.

Findings:
- **No ensemble is worth it.** Averaging adds +0.0010–0.0013 for 4× the predict cost, and every ensemble is worse
  than M3 alone. The weak M2 drags the average down.
- **M3 is a real improvement, not selection noise.** It gains +0.0033 (7 SE), on both the K=20 base alone and
  Hybrid50, with India +0.0040 and US +0.0029. It has fewer FPs and fewer FNs; 840 S1 improve and 385 are harmed.
  Picking the best of 7 options could explain about 2 SE (~0.001), not 7.
- **Capacity is the lever.** The trend is monotone in capacity: 15 leaves −0.0045, 31 leaves ±0, 63 leaves +0.0033.
- **Inference cost is unchanged at K=20** (8.9 vs 9.4 s per 1M rows). At K=50 it goes from 4.5 to 9.5 s per 1M rows,
  because early stopping ends later. Either way it is small next to feature building (~170 CPU-s per 1M pairs).

**Adopted:** `train.py` now trains the production matchers with M3 (`matcher.PRODUCTION_PARAMS`). The retrained
K=20 OOF is bit-identical to the experiment's M3 OOF, the threshold stays 0.70 and the deep rule stays ≥ 0.85.
`matcher.train()` keeps the 31-leaf defaults that the earlier experiments used. The M0 artifacts are kept as
`matcher_m0.pkl` / `oof_m0.parquet` (git-ignored). The strictly nested check below confirms the gain (+0.0032).

**K=50 alone becomes competitive with M3.** Retrained with M3, K=50 alone reaches 0.9561 (threshold 0.77,
singletons 0.952), statistically tied with Hybrid50 on M3 at 0.9558 (+0.0003 ± 0.0005, CI [−0.0006, +0.0013]).
With M0, K=50 alone was clearly below K=20. The extra capacity lets the model use the deeper lists. A single K=50
run is therefore a simpler, cheaper equal-score alternative to the two-run hybrid: one blocking run and one stream.
Hybrid50 stays the production path, because it is verified and the meta-model gain below was measured on it.

### Structured decisions on top of the pair probabilities (checkpoint 15)

`src/evaluation/structured_decoder.py` works on one row per Hybrid50 candidate:
- K=20 rows carry the K=20 OOF probability; deep rows carry the K=50 OOF probability.
- A decoder gives each row a score and a threshold, and keeps the production structure: base exclusivity, base
  priority, deep exclusivity.
- The vectorised decision engine reproduces the adopted 0.95248 exactly, and `tests/test_structured_decoder.py`
  checks it against `base_predictions` + `merge_deep` + per-S1 F0.5.
- **Cross-fitting:** S1 fall into the K=20 matcher's five GroupKFold folds. Every tuned parameter or model used for
  a fold is fitted on the other four folds, and results are pooled over the held-out folds.

**15A: per-S1 structure** (random S1, by number of true matches):

| True matches | S1 | top1 | top2 | top3 | S1 accepted | F0.5 |
|---|---|---|---|---|---|---|
| 0 | 1,060 | 0.195 | 0.065 | 0.021 | 0.09 | 0.927 |
| 1 | 1,030 | 0.912 | 0.143 | 0.036 | 0.96 | 0.886 |
| 2 | 3,406 | 0.980 | 0.856 | 0.139 | 1.86 | 0.940 |
| 3 | 4,777 | 0.993 | 0.960 | 0.808 | 2.77 | 0.957 |
| 4+ | 9,546 | 0.997 | 0.989 | 0.959 | 4.46 | 0.965 |

- **The count dimension is nearly solved already.** The number of accepted candidates tracks the true count.
- **Oracle bound:** told each S1's true count and taking its top-N winners by probability, the decoder reaches only
  **0.9533 (+0.0008)**.
- **So the remaining errors are about *which* candidates, not *how many*.**

**15B: adaptive thresholds** (cross-fitted coordinate descent over a 0.30–0.98 grid; small predefined variants):

| Decoder | macro-F0.5 | Δ vs 0.9525 (±SE) [95% CI] | India | US | Singletons | Rule (per-fold parameters) |
|---|---|---|---|---|---|---|
| uniform t20, deep 0.85 | 0.9524 | −0.0001 ± 0.0002 [−0.0004, +0.0002] | 0.9389 | 0.9614 | 0.927 | t20 0.68–0.72 |
| uniform t20 + tuned deep | 0.9522 | −0.0003 ± 0.0002 | 0.9385 | 0.9614 | 0.926 | deep 0.63–0.77 |
| rank 1–3 / rank 4–20 / deep | 0.9522 | −0.0003 ± 0.0002 | 0.9386 | 0.9613 | 0.926 | 0.68–0.72 / 0.69–0.72 / 0.63–0.77 |
| first vs additional match / deep | 0.9519 | −0.0006 ± 0.0003 [−0.0011, −0.0000] | 0.9374 | 0.9616 | 0.933 | |

- **The adopted threshold carries almost no tuning optimism:** a cross-fitted uniform threshold loses only 0.0001.
- **Extra threshold parameters only fit noise.** No native-script or address-contradiction thresholds were tried,
  because the error analysis showed no FP/FN asymmetry to justify them.

**15D: match-count decoder.** An HGB classifier predicts each S1's count bucket (0/1/2/3/4+) from S1 aggregates of
the first-stage probabilities: top-6 probabilities, gaps, counts above 0.1–0.9, probability sum, and the country.
It is cross-fitted by S1.
- Its count accuracy is 0.835, against 0.814 for the count the adopted rule implies.
- **Top-N:** accept the predicted number of best winners. **0.9387 (−0.0138).**
- **Cap:** the adopted decisions, capped at N. **0.9516 (−0.0009 ± 0.0004).**
- **Cap+fill:** the cap, then filled up to N from winners with prob ≥ 0.5. **0.9490 (−0.0035).**

All three are negative, as the +0.0008 oracle bound predicted.

**15C: second-stage meta-model.** This does not replace the matcher. An HGB pair model is trained on the rows with
first-stage probability ≥ 0.02; these hold 128,214 rows and 99.6% of true pairs, and the rest are never accepted.
Its inputs are:
- the first-stage OOF probability, the origin (base/deep) and the blocker rank;
- S1 context: top-3 probabilities, counts above 0.3/0.5/0.7/0.9, probability sum, best base/deep probability,
  position of the row in the S1, and the gap to the S1's best;
- optionally, the 61 A+C+E pair features.

The cross-fitting is nested. For each outer fold, four inner models give clean scores to tune (t_base, t_deep) on
the four training folds, and the outer model scores the held-out fold. Decisions keep the production structure,
with winners by meta score.

| Decoder | macro-F0.5 | Δ vs 0.9525 (±SE) [95% CI] | India | US | Singletons | FP | FN | Rule |
|---|---|---|---|---|---|---|---|---|
| current Hybrid50 | 0.9525 | – | 0.9390 | 0.9615 | 0.927 | 950 | 6,575 | t20 0.70, deep 0.85 |
| meta: prob + S1 context (17 features) | 0.9522 | −0.0003 ± 0.0005 [−0.0012, +0.0006] | 0.9384 | 0.9614 | 0.946 | 1,120 | 6,108 | t_base 0.64–0.70, t_deep 0.52–0.66 |
| **meta: + pair features (77 features)** | **0.9547** | **+0.0022 ± 0.0005 [+0.0012, +0.0031]** | **0.9407** | **0.9641** | **0.949** | 988 | 5,935 | t_base 0.65–0.71, t_deep 0.60–0.71 |

The pair-feature meta-model has a subtle stacking caveat. The first-stage OOF probabilities of its training rows came
from models that had seen the held-out fold's labels; this is standard stacking, but not strictly clean. The
strictly nested re-run is below.

**Strictly nested re-run** (`python -m src.evaluation.structured_decoder strict_meta --first M3`, 23 min). For every
outer fold k:
- the first-stage K=20 and K=50 matchers are retrained without fold k: inner 4-fold OOF for the training S1, and a
  model on all four training folds for fold k;
- the baseline threshold t20, the meta thresholds and the meta-model are all fitted on the training folds.

No model or threshold that touches fold k has seen a fold-k label. The baseline in the same framework is the first
stage alone, with t20 per fold and deep 0.85.

| System (first stage M3, strict) | macro-F0.5 | Δ vs 0.9525 (±SE) [95% CI] | India | US | Singletons | FP | FN | Rank 1–3 FP |
|---|---|---|---|---|---|---|---|---|
| first stage only (Hybrid50 decision) | 0.9557 | +0.0032 ± 0.0005 [+0.0023, +0.0041] | 0.9427 | 0.9644 | 0.938 | 867 | 6,199 | 404 |
| **first stage + meta-model decoder** | **0.9574** | **+0.0049 ± 0.0006 [+0.0038, +0.0059]** | **0.9446** | **0.9659** | **0.956** | 884 | 5,783 | 377 |

- **Meta over the M3 first stage:** +0.0017 ± 0.0004 [+0.0008, +0.0025], with India +0.0019 and US +0.0015. It is
  positive in every outer fold (+0.0005, +0.0020, +0.0016, +0.0022, +0.0020); 689 S1 improve and 361 are harmed.
- **The per-fold thresholds are stable:** t20 0.67–0.74; meta thresholds 0.67–0.75 for base and 0.65–0.77 for deep.
- **The M3 gain survives strict evaluation:** +0.0032 strict vs +0.0033 non-strict.
- **The same strict check with the M0 first stage** (`--first M0`, 19 min):
  - M0 alone: 0.9527 (+0.0002 vs the adopted 0.9525, CI [−0.0001, +0.0006]), so the adopted estimate is unbiased.
  - M0 + meta: 0.9545 (+0.0020 [+0.0010, +0.0030]); over its own first stage +0.0018 ± 0.0005 [+0.0007, +0.0028],
    with India +0.0023 and US +0.0014.
  - Positive in 4 of 5 folds.
  - The meta gain therefore replicates on both first stages.

**Decision:**
- **The meta-model decoder on the M3 first stage is the best development system (0.9574).** It clears the ≥ 0.9540
  bar and gains ≥ +0.0015 over its own first stage, with a CI above zero and both countries improving.
- **It is not in production yet.** It needs a joint pass over the paired K=20/K=50 shards: first-stage
  probabilities of all the S1's rows for the context features, then the meta score, then exclusivity.

### Targeted blocker rescue (checkpoint 16): negative

`src/evaluation/blocker_rescue.py`. The true pairs that never reach the matcher (**Hybrid50 FN1**) number 2,382 on the
random S1, 3.5% of true pairs. Their traits, compared with true pairs the blocker does find:

| FN1 trait | FN1 share | Reachable share |
|---|---|---|
| India | 64.6% | 39.6% |
| missing address (either side) | 31.4% | 3.4% |
| no shared name token | 40.4% | 13.4% |
| address-only evidence (no name token, address token-sort ≥ 0.6) | 17.3% | 12.3% |
| native-script candidate | 17.7% | 6.9% |
| house-number conflict | 11.2% | 4.9% |
| transliteration variant (Latin, phonetic match) | 7.9% | 4.3% |
| alias / rebrand (different name, same address) | 5.0% | 3.9% |
| short S1 name (≤ 1 token) | 1.0% | 0.6% |
| S1 with weak evidence (K=20 top prob < 0.5) | 6.9% | 0.2% |

Most FN1 are extra matches of S1 that already have a confident match, so targeting weak S1 cannot reach them.

**Two rescue channels** reuse P2's encoded token fields and the engine's vocabulary rules, at no more than +5/+10
candidates per S1:
- **trigram:** the `g` field, i.e. trigrams of the transliterated, space-free name, with df ≤ 2000, as the old
  fallback used;
- **address:** the `a`/`ab` fields with df ≤ 20000 and P2's weights 1.5/1.0.

| System (random S1) | Pair recall | Ceiling | Added / S1 (mean, p95, max) | Extra true / false | macro-F0.5 (rescue classifier, pre-declared T=0.85) |
|---|---|---|---|---|---|
| Hybrid50 | 0.9654 | 0.9874 | – | – | 0.9525 |
| + trigram, +5 | 0.9659 | 0.9875 | 1.5 / 5 / 5 | 31 / 30,400 | – |
| + trigram, +10 | 0.9660 | 0.9876 | 3.1 / 10 / 10 | 42 / 60,792 | 0.9521 (−0.0004 ± 0.0001; +17 TP / +44 FP) |
| + address, +5 | 0.9671 | 0.9878 | 5.0 / 5 / 5 | 113 / 98,844 | – |
| + address, +10 | 0.9674 | 0.9879 | 10.0 / 10 / 10 | 138 / 197,769 | 0.9512 (−0.0013 ± 0.0002; +84 TP / +145 FP) |

The targeted variants recover almost nothing:
- **weak-evidence S1 only:** 2 (trigram) or 3 (address) true pairs;
- **missing-address or non-Latin candidates only:** 11 (trigram) or 5 (address) true pairs.

Recovery by group: trigram +10 finds India 25 / US 17 / native-script 0; address +10 finds India 33 / US 105 /
native-script 5.

How the rescue classifier works:
- It is a cross-fitted HGB on the rescue pairs, using the matcher's A+C+E features.
- The channel score stands in for the block/name score, and the rescue rank for P2's rank.
- Rescued pairs get the lowest priority: base > deep > rescue.

Retrieval took 80 s (trigram) and 233 s (address) for 29k S1.

**Decision: not adopted.** Even perfect decisions on the rescued pairs would add at most +0.0005 (ceiling), and
the actual effect is negative. The missed pairs mostly have *no* usable name or address evidence (missing
addresses, no shared tokens), so no local token channel can reach them cheaply.

### Competition / ownership (checkpoint 17)

`python -m src.evaluation.structured_decoder competition` uses test-available signals only: the number of claiming
S1, the best competing claimant's probability and the winner's margin over it. Ground-truth ownership is used only
to describe FPs. The city S1 (every S1 of two cities is sampled) are the competition-rich context.

| Accepted pairs | Random S1 | City S1 |
|---|---|---|
| mean claimants per accepted candidate | 1.11 | 4.64 |
| accepted candidates with a competing claimant | 9.2% | 64.1% |
| FPs whose candidate is another S1's true match | 281 of 950 (30%) | 45 of 318 (14%) |

| Ownership rule | Random S1 F0.5 (Δ) | City S1 F0.5 (Δ ± SE) |
|---|---|---|
| adopted exclusivity | 0.95248 | 0.95994 |
| no exclusivity | 0.95248 (0) | 0.95983 (−0.0001 ± 0.0001) |
| exclusivity + winner margin ≥ 0.05 / 0.1 / 0.2 over the best competitor | 0.95247–0.95249 | +0.0003 / +0.0004 / +0.0003 (± 0.0002) |

- **No margin rule is adopted:** the gains are under 2 SE and there is no effect on the target S1.
- **Oracle bound (analysis only):** removing every random-S1 FP whose candidate is another S1's true match would give
  0.9562 (+0.0037).
- **How much of that full scale delivers:** where owners compete (city S1), exclusivity removes only ~15% of such
  FPs (53 → 45). The full test, where every owner is present, should gain somewhat over the development estimate,
  probably well under +0.001. It cannot be measured without full-split blocking.

### Oracle gap: where the remaining macro-F0.5 goes

`python -m src.evaluation.oracle_gap` (40 s from cached artifacts) takes the strictly nested held-out decisions of
the 0.9574 system and places every lost bit of per-S1 F0.5. Random development S1:

| Oracle ladder | macro-F0.5 |
|---|---|
| O0 current (M3 + meta, strictly nested) | 0.9574 |
| O1 perfect per-S1 cut on the current score order | 0.9831 |
| O1′ perfect cut on the M3 probability order | 0.9828 |
| O2 perfect selection among Hybrid50 candidates (ceiling) | 0.9874 |
| O3 perfect selection among Hybrid50 + K=100 candidates | 0.9897 |
| O4 ground truth | 1.0 |

Per S1 the loss telescopes: total 0.0426 = **cut 0.0257** (O1 − O0) + **blocking 0.0126** (1 − O2) +
**ranking 0.0043** (O2 − O1). The same split by group:

| Group | S1 | O0 | Blocking | Ranking | Cut |
|---|---|---|---|---|---|
| India | 7,945 | 0.9446 | 0.0207 | 0.0040 | 0.0307 |
| US | 11,874 | 0.9659 | 0.0073 | 0.0045 | 0.0224 |
| 0 true matches | 1,060 | 0.9557 | – | – | 0.0443 |
| 1 | 1,030 | 0.8684 | 0.0418 | 0.0094 | 0.0805 |
| 2 | 3,406 | 0.9434 | 0.0162 | 0.0055 | 0.0349 |
| 3 | 4,777 | 0.9624 | 0.0112 | 0.0044 | 0.0221 |
| 4+ | 9,546 | 0.9696 | 0.0103 | 0.0037 | 0.0163 |

Every error falls in exactly one class. "Recoverable" is the macro-F0.5 gain if only that class were fixed:

| Error source | Pairs | S1 affected | Loss of affected S1 (macro share) | Max recoverable |
|---|---|---|---|---|
| FN, true candidate ranked above every false one but below the threshold | 2,558 | 2,244 | 0.0155 | **+0.0143** |
| FN, missing from the candidates (blocking) | 2,382 | 1,920 | 0.0160 | +0.0141 |
| FP, every present true ranked above it (threshold too low for this S1) | 581 | 552 | 0.0071 | +0.0067 |
| FN, outranked by a false candidate of the S1 | 517 | 476 | 0.0047 | +0.0033 |
| FP, singleton S1 | 59 | 47 | 0.0024 | +0.0024 |
| FP, outranks a true candidate of the S1 | 242 | 237 | 0.0029 | +0.0024 |
| FN, below the meta floor (first stage < 0.02) | 317 | 304 | 0.0024 | +0.0019 |
| FN, lost to exclusivity | 9 | 9 | 0.0001 | +0.0000 |
| (S1-level) multi-match S1 with partial recovery | 3,204 | 4,342 | 0.0225 | +0.0123 |

Rank 1–20 rows carry most cut errors; deep rows carry most below-floor FNs. Among the missing pairs, 421 have a
native-script candidate and 749 a missing candidate address.

**The cut is not the real lever.** An expected-F0.5 set decoder (`src/evaluation/set_decoder.py`) picks each S1's
cut to maximise its expected F0.5, computed exactly with Poisson-binomial dynamic programming on calibrated
probabilities. It gains only +0.0002 ± 0.0004 over the threshold rule, so the threshold is already the best cut the
scores allow.

O1's +0.026 exists only because the oracle uses labels to separate candidates that have the *same* score. The
bottleneck is therefore **discrimination inside the uncertain band**. Inspecting that band shows generated decoys
against corrupted true duplicates (see "Token-alignment features" below). Similarity to the S1's confident matches
does not separate them either (AUC 0.48–0.60).

### Token-alignment ("decoy") features: the targeted fix

The uncertain band holds two kinds of near-identical records, and overall string similarity scores them alike:
- **Decoys** (invented illustrations): one distinctive name token replaced ("Harlow" → "Brenwick") or morphed at
  its end ("Tavell" → "Tavelli", "Quorin" → "Quorinex"); an injected house-number prefix ("H.no 12 …"); a truncated or
  extended number.
- **True duplicates:** typos and transpositions, OCR confusions (0/o, 1/l, c/e), added generic words (LLC,
  Services, Center, Shri), token shuffles, joined or domain forms ("#tavellbakery", "tavellbakery.com").

`src/matching/decoy_features.py` (13 features, computed from the two records only, so identical on test) aligns
each distinctive name token and records the type of difference:
- exact, OCR-equivalent, joined form, interior one-edit typo, end morph, or none;
- extra unmatched distinctive tokens in the candidate, and a "replaced token" flag;
- added generic words;
- an injected "H.no" prefix;
- number extension or truncation, and number equality.

Features for all 1.46M development pairs are built once in 131 s and cached (`decoy_experiment.all_row_decoy_features`).

Controlled comparison (`src/evaluation/decoy_experiment.py`, cross-fitted, the same folds and nested thresholds as
`train_meta`). Δ is against M3 Hybrid50 without the meta-model (0.9558):

| Method | macro-F0.5 | Δ | India | US | Singletons | FP | TP |
|---|---|---|---|---|---|---|---|
| M3 + meta (production features) | 0.9574 | +0.0016 | 0.9450 | 0.9657 | 0.959 | 882 | 62,999 |
| **M3 + meta + decoy** | **0.9624** | **+0.0066** | **0.9500** | **0.9707** | **0.963** | 747 | 63,643 |
| M3 + decoy in the first stage, no meta | 0.9626 | +0.0068 | 0.9495 | 0.9713 | 0.958 | 598 | 63,210 |
| M3 + decoy in the first stage + meta + decoy | 0.9630 | +0.0072 | 0.9494 | 0.9722 | 0.969 | 721 | 63,649 |

Findings:
- **The decoy features add +0.0050 ± 0.0004** to the meta decoder (95% CI [+0.0041, +0.0058]), with India
  +0.0050 and US +0.0050. 1,048 S1 improve and 334 are harmed.
- **Adding them to the first stage as well changes little:** +0.0006 ± 0.0005, CI crossing 0.
- **The M3 + meta + decoy bundle** (`train_meta --decoy`) reproduces 0.9624 exactly, with the matrix and decision
  checks passing and thresholds base 0.73, deep 0.65.

**Strictly nested confirmation** (`strict_meta --first M3 --decoy --first-decoy`, 31 min). The first stage, with
the decoy features, is retrained inside every outer fold. Nothing that scores a fold has seen its labels.

| System (strict) | macro-F0.5 | Δ vs 0.9574 | India | US | Singletons | FP | FN |
|---|---|---|---|---|---|---|---|
| M3 + decoy first stage (Hybrid50, t20 per fold) | 0.9627 | +0.0053 | 0.9498 | 0.9713 | 0.956 | 618 | 5,579 |
| **M3 + decoy first stage + meta decoder** | **0.9643** | **+0.0069** | **0.9512** | **0.9731** | **0.977** | 671 | 5,113 |

The meta decoder adds +0.0016 ± 0.0004 [+0.0008, +0.0024] over the first stage, positive in all 5 folds.

**Adopted:**
- The features are a first-stage `build_features` group "T" (`extra_features.PRODUCTION_GROUPS` = A, C, E, T), so
  they also reach the meta-model through the first-stage features.
- The production matchers were retrained through `train.py`, which now also writes the fingerprinted feature cache
  it computes (`X_ACET.parquet`).
- The retrained K=20 and K=50 OOF are **bit-identical** to the experiment's M3 + decoy OOF. The production feature
  path therefore computes exactly the experimental features.

**Retrained matchers:**
- **K=20:** alone 0.9593 (was 0.9525), threshold 0.76.
- **K=50:** alone 0.9626 (was 0.9561), threshold 0.74.

**New `train_meta` bundle:**
- `meta_extra` is empty, because the first-stage features already hold group T.
- Thresholds base 0.74, deep 0.63; built in 61 s from caches.
- Cross-fitted estimate **0.9630**; the first stage alone scores 0.9628.
- The strictly nested estimate of this architecture is **0.9643**. Paired against the previous 0.9574 system on the
  same S1 (both strict): **+0.0069 ± 0.0006, 95% CI [+0.0058, +0.0081]**, with India +0.0066 and US +0.0072;
  1,377 S1 improve and 585 are harmed.

**Oracle ladder after the fix** (`oracle_gap` on the new strict rows):
- O0 0.9643, O1 0.9844, O2 0.9874, O3 0.9897.
- Loss 0.0357 = cut 0.0201 (was 0.0257) + blocking 0.0126 + ranking 0.0029 (was 0.0043).
- **Missing candidates (blocking) are now the single largest recoverable class:** +0.0140, ahead of "correctly
  ranked but below the threshold" at +0.0117.

### Stronger GBDT family: one controlled XGBoost test

`python -m src.evaluation.gbdt_experiment` trains XGBoost 3.4 (already installed, no new dependency). It uses one
predefined configuration: depth 8, 700 rounds, learning rate 0.05, subsample and colsample 0.8,
`min_child_weight` 2, hist. Everything else matches M3 exactly: the cached matrices (61 + 13 token-alignment
features), the GroupKFold of `train.py`, the Hybrid50 decision (t20 per fold, deep 0.85), and the cross-fitted
meta + decoy decoder on top.

| First stage (with the decoy features) | Hybrid50, first stage only | + meta + decoy | India | US | Singletons |
|---|---|---|---|---|---|
| HGB M3 | 0.9626 | 0.9630 | 0.9494 | 0.9722 | 0.969 |
| **XGBoost** | **0.9640** | **0.9647** | **0.9526** | **0.9727** | **0.973** |

Paired XGBoost − HGB:
- **First stage only:** +0.0014 ± 0.0004, CI [+0.0007, +0.0021].
- **With meta + decoy:** **+0.0016 ± 0.0004, CI [+0.0008, +0.0025]**, with India +0.0032 and US +0.0006.

XGBoost 5-fold OOF takes 152 s at K=20 and 228 s at K=50, faster than M3.

**Strictly nested check** (`strict_meta --first XGB --decoy --first-decoy`, 16 min):
- **First stage alone:** 0.9636 for XGBoost vs 0.9627 for HGB, +0.0009 [+0.0001, +0.0017].
- **With the meta decoder:** 0.9650 vs 0.9643, **+0.0007 ± 0.0004 [−0.0002, +0.0015]**; India +0.0020, US −0.0002.

The cross-fitted +0.0016 shrinks to +0.0007 under strict evaluation: below the +0.0015 bar, with a CI touching zero.
**XGBoost is not adopted;** production stays HGB (M3 + group T + meta decoder).

### Candidate ceiling: targeted rescue cannot pass 0.990

`python -m src.evaluation.candidate_ceiling` measures ceilings with no matcher involved. It uses four channels
from P2's own encoded fields and vocabulary rules: name trigrams, rare address tokens, rare name tokens and
phonetic keys. Each channel gives every development S1 up to 20 candidates that are not in its Hybrid50 list.

| Rescue (random dev S1) | Pair recall | Ceiling | Added / S1 (mean, p95) | Extra true / false |
|---|---|---|---|---|
| none (Hybrid50) | 0.9654 | 0.9874 | – | – |
| best single channel (address) +5 / +10 / +20 | 0.9671 / 0.9674 / 0.9678 | 0.9878 / 0.9879 / 0.9880 | 5 / 10 / 20 | 113 / 137 / 162 true |
| all four channels +5 each | 0.9683 | 0.9883 | 15, 20 | 197 / 297,111 |
| all four channels +10 each | 0.9691 | 0.9886 | 30, 40 | 256 / 589,825 |
| all four channels +20 each | 0.9703 | **0.9889** | 59, 80 | 338 / 1,172,688 |
| (reference) Hybrid50 + K=100, perfect selection | 0.9714 | 0.9897 | 100 | – |

- **Coverage:** only 338 of the 2,382 missing true pairs are reached by any channel at depth 20.
- **The unreached pairs:** 66% India, 20% native-script candidates, 32% missing address, and 39% share no name
  token at all. They carry no local token evidence to retrieve them with.
- **No targeted rescue reaches a 0.990 ceiling, let alone 0.993 or 0.995.** Even doubling the candidate lists gets
  to 0.9889.
- **What this means for 0.99:** a local macro-F0.5 of 0.99 is therefore impossible with this candidate architecture,
  even with a perfect matcher.

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

### Inference performance (streaming, identical output)

**Profile** of the single-process path on 5k real test S1 (100k pairs, A+C+E model):

| Cost | Time (profiled) | Notes |
|---|---|---|
| group E name statistics | ~112 s unprofiled | once per run, independent of K |
| `build_features` | 45 s | per pair; biggest parts: transliteration group C 17 s, TF-IDF 12 s, group A 4.5 s, group E 2.9 s |
| `predict_proba` | 2.3 s | |
| RapidFuzz | 0.5 s | |
| global winner reduction | ~0 s | |
| shard reads | ~1.8 s per shard | a bug, now fixed (see below) |

**The shard-read bug.** P2's `CandidateStore.iter_frames` converted each whole multi-million-row S2/S3 column to
NumPy for every shard, then picked the shard's rows. It now takes the rows first (`Series.take`), with identical
values.

**Parallel scoring.** `stream_predict(..., workers=N)` / `predict.py --workers N` scores shards in N spawn-safe
processes:
- **Worker inputs:** each worker gets only the shard's compact pairs/records, the model, and (group E) the name
  statistics, via a temp file. Workers never load the source tables.
- **Bounded memory:** at most 2N shards are in flight.
- **Deterministic reduction:** the parent consumes results strictly in shard order, so global top-2, exclusivity,
  tie-breaks and output are unchanged.
- **Tested:** synthetic multi-shard equivalence (with and without A+C+E) in `tests/test_inference.py`.

Measured on real test S1 (K=20). Every output is byte-identical to the single-worker streaming output:

| Configuration | 5k S1 (100k pairs) | 50k S1 (1.0M pairs) | Speedup (50k) | Peak RAM, process tree (50k) | Pairs/s (50k) |
|---|---|---|---|---|---|
| before (legacy shard read), 1 worker | 192 s | 483.1 s | 1.00× | 4.7 GB | 2,070 |
| shard-read fix, 1 worker | 138.9 s | 305.4 s | 1.58× | 4.5 GB | 3,275 |
| **shard-read fix, 2 workers (CLI default)** | 133.1 s | **220.6 s** | **2.19×** | **6.7 GB** | 4,533 |
| shard-read fix, 4 workers | 131.8 s | 214.0 s | 2.26× | 8.8 GB | 4,673 |

- **Fixed cost:** about 132 s of each run is fixed (name statistics plus source tables), which is why 5k barely
  changes.
- **Per-pair cost:** 351 s (before) → 173 s (fix) → 89 s (2 workers) → 82 s (4 workers) per 1M pairs. A fourth worker
  adds 3% for about +2.1 GB, because the parent's sequential shard reads and joins become the limit.
- **Full-test estimates** (extrapolated from the 50k measurements plus the fixed cost, not measured): K=20
  (34.7M pairs) about 0.9 h with 2 workers, versus about 3.4 h before; K=50 (86.6M pairs) about 2.1 h with 2
  workers.

### Production Hybrid50 inference

`src/pipeline/predict_hybrid.py` is the test-time implementation of the adopted system. It needs two P2 runs over the
same S1 and shard plan (only `--top-k` differs):
1. **Base.** The improved K=20 matcher streams the exact K=20 candidates (`stream.stream_winners`): global
   exclusivity, then its threshold (0.70). These assignments are frozen.
2. **Deep.** The K=50 matcher streams the K=50 candidates. Every row feeds the cross-entity top-2 and the within-S1
   features, but only the hybrid's deep rows are scored and compete: K=50 candidates that are not in the S1's exact
   K=20 list, up to the cap of 50. A deep candidate goes to its global deep winner (highest probability, then the
   smallest `s1_id`, then the earliest row) when that probability is ≥ 0.85 and the base did not already assign
   the candidate.
3. **Outputs,** one row per S1 in `test_source1` order, written in one pass:
   - `candidate_pairs.tsv` lists the hybrid set: every exact K=20 candidate in K=20 order, then K=50-only
     candidates in K=50 order, up to 50.
   - `matching_results.tsv` holds the base plus deep matches, sorted. Every match is checked to be in its S1's
     candidate list, and the run fails otherwise.

The two runs share one copy of the source tables (`CandidateStore(..., tables=...)`) and one set of group E name
statistics. Paired shards are processed one per country at a time, so memory stays bounded as in single-K
inference.

The first-stage matchers are whatever `train.py` saved. Since checkpoint 14 that is the 63-leaf M3 configuration;
the equivalence and benchmark numbers below cover both the earlier M0 models and M3.

**Equivalence.** `src/pipeline/hybrid_equivalence.py` re-derives both files on real test smoke runs through an
independent path:
- every pair scored in memory, with the global cross-entity top-2 over the whole run;
- then the offline functions `hybrid_candidates`, `base_predictions` and `merge_deep`.

| Smoke set (first N test S1) | Matchers | Pairs K=20 / K=50 | Base matches | Deep matches | `matching_results.tsv` | `candidate_pairs.tsv` | Production runtime (2 workers) |
|---|---|---|---|---|---|---|---|
| 5k | M0 | 100k / 250k | 16,073 | 219 | identical | identical | 152 s |
| 50k | M0 | 1.0M / 2.5M | 160,471 | 2,324 | identical | identical | 479 s |
| 5k | M3 (production) | 100k / 250k | 16,120 | 274 | identical | identical | 153 s |
| 50k | M3 (production) | 1.0M / 2.5M | 160,843 | 2,671 | (benchmark run) | | **478 s, peak RAM 6.55 GB** (process tree) |

**Runtime breakdown (50k):**
- about 95 s fixed: source tables and group E name statistics;
- about 115 s for the K=20 stage (1.0M pairs);
- about 260 s for the K=50 stage: 2.5M pairs through pass 1 and features, 1.5M deep rows scored;
- about 10 s for outputs.

**Full-test estimate** (extrapolated from these numbers, not measured): about 3.7 h with 2 workers. RAM should stay
near 7 GB, because the smoke runs already hold the full test source tables and per-candidate arrays; only the
number of shards grows.

**Checks on the smoke outputs:**
- Official validator (`--check-ids`, against a copy of `test_source1.tsv` cut to the smoke S1): **PASS**, with no
  subset warning.
- The base part equals the previous single-K=20 stream output exactly; the deep matches come on top.

**Tests:** `tests/test_predict_hybrid.py` covers:
- base priority, deep threshold (inclusive, float32), deep exclusivity and ties across shards;
- random sharded equivalence with `merge_deep`;
- K=20-only preservation and K=50-only order/cap;
- shard pairing and file-order merging across countries;
- an end-to-end run on two real P2 runs (K=3 base, K=6 deep, cap 5) that must be byte-identical to the offline
  reference, and identical for 1 and 2 workers;
- official S1 order, singleton rows, no duplicate candidates, every match among its candidates, exclusivity.

Three mutations are all caught by these tests: no base priority, all K=50 rows treated as deep, and deep threshold
ignored.

### Production M3 + meta decoder (the 0.9574 system)

`python -m src.pipeline.predict_hybrid` now defaults to `--decoder meta`, the strictly nested 0.9574 system.

**Definitions and bundle:**
- `src/pipeline/meta_decoder.py` is the single definition of the meta-model's inputs and decision. It builds the
  per-row S1 context over all of the S1's Hybrid50 rows and the meta matrix, and applies the production decision:
  base winners by meta score ≥ `t_base`, then deep winners ≥ `t_deep` on candidates the base did not take.
- `src/pipeline/train_meta.py` builds the production bundle `output/candidates_p3/hybrid_meta.pkl` (git-ignored)
  from cached artifacts only, in 111 s: 25 s to load the caches and build the matrix, 86 s for the meta CV and the
  final fit. The bundle holds:
  - both M3 first-stage matchers;
  - the meta-model, its feature list, the meta floor and version;
  - the thresholds (base 0.69, deep 0.67), tuned on 5-fold meta OOF over all development S1;
  - the hybrid configuration (20/50/50);
  - the first-stage SHA-256s and the feature-cache fingerprints.

**Checks built into training (both hold):**
- The production meta matrix is identical to the development experiment's matrix.
- The production decision function is identical to the experiment's decision engine on the development OOF.
- The cross-fitted development estimate is **0.9574** (+0.0016 ± 0.0004 over M3 alone, CI [+0.0007, +0.0025]),
  equal to the strictly nested 0.9574.

**Inference (`predict_hybrid_meta`):**
- It runs pass 1 (cross-entity top-2) for both runs.
- One joint pass then walks the paired K=20/K=50 shards. Spawn-safe workers return the first-stage probabilities
  and the feature rows of rows ≥ 0.02.
- The parent computes each S1's context and the meta score, and keeps float64 global winners per stage.
- Outputs go through the same writer and subset check as before.

**Equivalence:** `hybrid_equivalence` (`--decoder meta`) re-derives both files by scoring every pair in memory and
applying the meta decoder over the whole run at once (vectorised exclusivity).

| Smoke set | Base / deep matches | Meta rows | `matching_results.tsv` | `candidate_pairs.tsv` | Validator | Production runtime (2 workers) |
|---|---|---|---|---|---|---|
| 5k (0.9574 bundle) | 16,162 / 358 | 25,063 | identical | identical | PASS | 200 s (CPU shared with a training job) |
| 50k (0.9574 bundle) | 161,281 / 3,360 | 246,310 | not re-derived (see note) | | | 448 s (CPU shared) |
| **5k (final bundle, group T)** | 16,004 / 331 | 22,997 | identical | identical | PASS | 154 s |
| **50k (final bundle, group T)** | 159,530 / 3,348 | 226,794 | identical | identical | PASS | 555 s |

With group T, 50k production takes 555 s, versus 448 s without it: every row's token alignment is now computed. The
offline reference took 1,302 s.

The first 50k offline reference hit a quadratic `np.isin` on string candidate ids inside `meta_decoder.decide`
(NumPy loops in Python for object arrays). The ids are now factorized to integers, and `decide` handles 2.5M rows
in 2 s. Production streaming never used that function; it uses the `Winners` state.

**Adopted since: the token-alignment features** (see "Token-alignment features"). They went in two steps:
1. **Meta-model only** (the bundle's `meta_extra`, computed by the workers for rows ≥ 0.02 from the shard records):
   90 meta features, thresholds base 0.73 / deep 0.65, cross-fitted estimate 0.9624. On 5k smoke it was
   byte-identical to the offline reference (16,096 base + 340 deep matches) with validator PASS.
2. **Final: the first-stage feature group T** (both matchers retrained, bit-identical to the experiment), so the
   meta-model sees the features through the first stage.
   - Thresholds base 0.74 / deep 0.63; cross-fitted estimate **0.9630**; strictly nested **0.9643**.
   - `train_meta` leaves `meta_extra` empty when the first stage has group T, and adds it otherwise.
   - `--no-decoy` rebuilds the older bundles. The 0.9574 bundle is kept as `hybrid_meta_nodecoy.pkl`, and the M3
     matchers without group T as `matcher_m3.pkl`.

**Tests:** `tests/test_meta_decoder.py` covers:
- context features against a loop reference, with ties;
- the meta-floor row check;
- the vectorised decision against streaming float64 winners across shards;
- an end-to-end run on two real P2 runs that must equal the offline reference, and be identical for 1 and 2
  workers;
- official-file properties and the bundle version check.

**Feature caches** (`src/matching/feature_cache.py`): experiments load the 61-feature development matrices in
1.3 s (K=20) and 1.7 s (K=50) instead of featurizing, which took 334–560 s. A cache is reused only if its
fingerprint matches:
- feature version and groups;
- the candidate run's `run.json`;
- the S1-sample hash;
- the size and first-MiB hash of each training source file.

The two existing caches were adopted after their rows and columns were verified.

## Reproduce

```
python -m src.matching.sample_candidates --top-k 20       # P2's blocker on the 29,169-S1 training sample (~4 min, 2 workers)
python -m src.matching.train --cands output/candidates_p3/k20      # ~7 min; add --variants "" "^(rank|name_score)" for the ablation
python -m src.blocking.generate_candidates --split test --top-k 20   # P2: output/candidates/test + output/candidate_pairs.tsv
python -m src.matching.predict --model output/candidates_p3/k20/matcher.pkl   # single K: streams shards (2 workers); memory ~flat in #pairs
# adopted Hybrid50 (needs output/candidates_p3/k50 trained with: sample_candidates --top-k 50 + train --cands .../k50):
python -m src.blocking.generate_candidates --split test --top-k 20 --out-dir output/candidates_k20 --no-tsv
python -m src.blocking.generate_candidates --split test --top-k 50 --out-dir output/candidates_k50 --no-tsv
python -m src.pipeline.train_meta       # production bundle from cached development artifacts (~3 min)
python -m src.pipeline.predict_hybrid   # meta decoder -> output/matching_results.tsv + output/candidate_pairs.tsv (+ hybrid_summary.json)
python -m src.pipeline.hybrid_equivalence --base-cands output/p3_smoke/k20_5000 --deep-cands output/p3_smoke/k50_5000   # smoke check
python resources/utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir data/raw/test
# development experiments (checkpoints 14-17; development sample only, never the final holdout)
python -m src.matching.ensemble_experiment oof && python -m src.matching.ensemble_experiment eval
python -m src.evaluation.structured_decoder analyze   # also: adaptive | meta | count | competition
python -m src.evaluation.structured_decoder strict_meta --first M3
python -m src.evaluation.blocker_rescue analyze && python -m src.evaluation.blocker_rescue recall --channel address   # or trigram; then: score
python -m src.evaluation.structured_decoder strict_meta --first M3 --decoy   # strictly nested check (writes strict_rows_*.parquet)
python -m src.evaluation.oracle_gap && python -m src.evaluation.set_decoder   # oracle ladder, error sources, expected-F0.5 cut
python -m src.evaluation.decoy_experiment [--first-stage]   # token-alignment features, controlled comparison
python -m src.evaluation.candidate_ceiling && python -m src.evaluation.gbdt_experiment
```

Previous results with the stand-in token blocker (K=30, macro-F0.5 0.8877) are in git history (`docs/p3_matching.md` at e5f18fb).
