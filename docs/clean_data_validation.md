# P1 clean dataset — validation against raw data

Reproduce: `python -m src.preprocessing.validate_clean` (read-only). It writes the full JSON report to
`output/clean_validation/report.json`, which is git-ignored.

Inputs:
- `data/clean/`: P1's package, copied unchanged from the delivered folder (checksums verified).
- `data/raw/`: the official challenge files, untouched.

Token-level figures use a deterministic 1-in-50 row subsample; every other figure covers all rows.

## 1. Files and structure

| File | Rows (excl. header) | Encoding |
|---|---|---|
| `source1_clean.csv` | 2,206,821 | UTF-8 |
| `source2_clean.csv` | 5,034,616 | UTF-8 |
| `source3_clean.csv` | 5,285,603 | UTF-8 |
| `source{1,2,3}_clean_SAMPLE.csv` | 2,000 each | S1/S2 UTF-8; **S3 has 99 cp1252 lines** |

- **Format:** comma-separated, double-quoted. Every file has the same 11 columns:
  `entity_id, business_name, business_name_normalized, name_tokens, business_address, address_normalized,
  country, country_normalized, state_normalized, postal_code_normalized, name_script`.
- **There is no train/test split column, and there is no cleaned ground-truth file.** Raw `train_ground_truth.tsv`
  stays valid because the IDs are unchanged.

## 2. Coverage, IDs and ordering

| Check | S1 | S2 | S3 |
|---|---|---|---|
| clean rows = raw **train** rows | ✅ 2,206,821 | ✅ 5,034,616 | ✅ 5,285,603 |
| raw train IDs missing from clean | 0 | 0 | 0 |
| raw **test** IDs present in clean | **0 of 1,732,544** | **0 of 4,887,273** | **0 of 5,082,316** |
| clean IDs not in raw | 0 | 0 | 0 |
| duplicate / empty / wrongly prefixed IDs | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| row order identical to raw train file | ✅ | ✅ | ✅ |
| raw train/test ID overlap | 0 | 0 | 0 |

**The clean files cover the training split only.** Every train entity is present exactly once, in the raw file
order, so clean row *i* is raw train row *i*. **No test data was cleaned.**

## 3. Raw values preserved?

| Column | S1 | S2 | S3 | What differs |
|---|---|---|---|---|
| `business_name` (verbatim) | 0 diffs | 2 | 13 | names that are literally `NA` became empty (read as a missing value) |
| `business_address` (verbatim) | 4 | 6 | 0 | addresses containing literal `"` characters were CSV-unescaped |
| `country` (verbatim) | 0 | 0 | 0 | – |
| `country_normalized` | US→US, India→IN | same | same | France never appears (test only) |

The raw text columns are therefore usable as the raw text: 25 rows out of 12.5M differ, and all 25 are
explained above.

## 4. How names and addresses were cleaned

**Names** (`business_name_normalized`):
- Lowercased and punctuation removed; `&` → `and`.
- Legal and abbreviation words expanded: `inc`→`incorporated`, `ltd`→`limited`, `pvt`→`private`,
  `corp`→`corporation`, `co`→`company`, `tech`→`technology`, `labs`→`laboratories`, `intl`, `mfg`, `bros`, …
- **Latin accents are kept** (`é`, `á`, …).

`name_tokens` is the sorted unique set of normalised tokens minus the stop-words `and`, `of`, `the`, `for`. That
is consistent in every sampled row.

| | S1 | S2 | S3 |
|---|---|---|---|
| normalised name ≠ lowercased raw name | 35.1% | 62.6% | 60.0% |
| token set changed vs a plain lowercase/accent-fold of raw (sample) | 24.5% | 39.2% | 35.5% |

**Addresses** (`address_normalized`):
- Lowercased and punctuation removed.
- Street words expanded: `st`→`street`, `rd`→`road`, `dr`→`drive`, `ave`→`avenue`, `ln`→`lane`, `ct`→`court`,
  `fl`→`floor`, `apt`→`apartment`, `bldg`→`building`.
- Single letters expanded to directions: `n/s/e/w`→`north/south/east/west`.
- Placeholders `<NULL>` and `N/A` removed (about 44k rows each in S2/S3 raw; none survive).
- Leading zeros removed from numbers.

| | S1 | S2 | S3 |
|---|---|---|---|
| token set changed (sample) | 13.2% | 48.5% | 46.3% |
| rows where a direction word appears that is not in the raw address | 90,215 (4.1%) | 240,261 (4.8%) | 222,482 (4.2%) |
| rows whose digit string changed (always only by removed zeros) | 26,292 (1.2%) | 277,051 (5.5%) | 274,185 (5.2%) |
| raw address empty (stays empty) | 0 | 168,967 | 175,916 |

**Structured fields:**

| | S1 US / India | S2 US / India | S3 US / India |
|---|---|---|---|
| `state_normalized` filled | 98.1% / 100% | 94.5% / 76.5% | 96.4% / 76.7% |
| `postal_code_normalized` filled | 0% / 0% | 0% / 0% | 0% / 0% |

## 5. Problems found

1. **No cleaned test data.** Any feature built on the clean columns cannot be produced for test inference until
   P1 delivers cleaned `test_source{1,2,3}`, or the cleaning code so we can apply it ourselves.
   **This is the main blocker for adopting clean data in the final pipeline.**
2. **Indic-script text is damaged in the normalised columns.** Vowel signs (combining marks) are stripped, so a
   word like `लिमिटेड` becomes the separate consonants `ल म ट ड`. This affects 99.9% of names containing an
   Indic script in the sample (9,487 of 9,492 in S2, 5,520 of 5,527 in S3). Those are about 474k S2 and 279k S3
   train names, and also Indic state names inside addresses. The raw columns are intact.
3. **Accented Latin names are tagged `name_script = non_latin`.** That's 290,263 S2 and 328,213 S3 names (all of
   them with accents and no Indic character), and the accents are not folded. `non_latin` therefore mixes real
   Indic names with Latin names that contain injected accent typos.
4. **Over-eager expansions:**
   - single letters in unit/block numbers become directions (`S-01` → `south 1`, `Block E` → `block east`).
     In S2 this happens to all 5,323 `Block E` and all 451 `S-0n` addresses.
   - `St` meaning *Saint* sometimes becomes `street` (438 / 616 / 640 rows show `street <saint-city>`)
5. **Small losses:**
   - 15 business names `NA` emptied
   - 10 addresses with literal quotes altered
   - leading zeros dropped (minor, and arguably desirable)
   - `postal_code_normalized` is always empty
6. **`source3_clean_SAMPLE.csv` is not a faithful sample:** 99 lines are cp1252-encoded, and only 1,763 of 2,000
   rows are identical to the full file. Use the full files.

## 6. Readiness

- **Training-side experiments can start now.** Clean rows align 1:1 with raw train rows and IDs.
- **Recommended use:**
  - keep raw text as the primary input
  - offer clean text as an *optional extra view*, e.g. expanded legal/street words and `state_normalized`
  - do not use `*_normalized` for Indic-script names, or the `name_script` flag as an Indic indicator
- **Before any clean-data feature can go into the submission,** P1 must deliver cleaned test files, or the
  cleaning code, and fix problems 2–4.
- **P2 blocking already handles these cases on raw text:** its own normaliser keeps Indic vowel signs,
  transliterates, and folds accents.

## 7. Loading raw, clean or both

`src/preprocessing/clean_data.py` gives experiments one switch. No P2/P3 algorithm uses it yet.

```python
from src.preprocessing.clean_data import load_source, clean_available

df = load_source("train", 2, mode="raw")        # name/address = raw text
df = load_source("train", 2, mode="clean")      # name/address = P1 normalised text
df = load_source("train", 2, mode="raw+clean")  # name/address = raw text, plus clean_* columns
```

- **Raw text in every mode:** `raw_name` and `raw_address` are always present.
- **Clean columns:** `clean_name`, `clean_address`, `clean_name_tokens`, `clean_country`, `clean_state`,
  `clean_postal_code` and `clean_name_script` are added whenever clean data is requested.
- **Alignment check:** every load verifies that clean rows align 1:1 with the raw rows.
- **Missing splits:** requesting clean data for a split without P1 files (currently `test`) raises
  `CleanDataUnavailable`, never a silent fallback.
- **Speed:** loading takes about 0.5 s (S1) to 1.5 s (S3) for the train split.
