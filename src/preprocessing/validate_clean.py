"""Validate P1's cleaned sources (data/clean/) against the raw challenge files.

Usage (from the repository root):

    python -m src.preprocessing.validate_clean                 # writes output/clean_validation/report.json

For each source (S1/S2/S3) it checks, separately for the train and test rows:
  * files, columns, row counts, duplicate / empty / wrongly prefixed IDs;
  * whether every raw entity is in the clean file (and nothing extra), and whether
    the clean row order equals raw train order followed by raw test order;
  * whether the raw ``business_name`` / ``business_address`` / ``country`` values are
    preserved verbatim, and how ``country_normalized`` maps the raw country;
  * how names/addresses were normalised (token changes, expansions) and where
    information may be lost (emptied values, dropped digits, damaged Indic text).

Whole-file checks run vectorised with pyarrow; token-level statistics use a
deterministic 1-in-``--sample-every`` row subsample. Read-only on all inputs.
"""

import argparse
import csv
import io
import json
import os
import re
import time
import unicodedata
from collections import Counter

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv

RAW_COLUMNS = ["entity_id", "business_name", "business_address", "country"]
CLEAN_COLUMNS = ["entity_id", "business_name", "business_name_normalized", "name_tokens", "business_address",
                 "address_normalized", "country", "country_normalized", "state_normalized",
                 "postal_code_normalized", "name_script"]
INDIC = r"[\x{0900}-\x{0DFF}]"
DIRECTIONS = r"\b(north|south|east|west)\b"


def read_table(path, delimiter, columns):
    quote = False if delimiter == "\t" else '"'
    table = pacsv.read_csv(
        path,
        read_options=pacsv.ReadOptions(block_size=1 << 26),
        parse_options=pacsv.ParseOptions(delimiter=delimiter, quote_char=quote),
        convert_options=pacsv.ConvertOptions(column_types={c: pa.string() for c in columns},
                                             strings_can_be_null=False),
    )
    if table.column_names != columns:
        raise ValueError(f"{path}: columns {table.column_names}, expected {columns}")
    return table


def count(mask):
    return int(pc.sum(pc.cast(mask, pa.int64())).as_py() or 0)


def fold_tokens(text):
    """Lowercase, strip Latin accents, split on non-alphanumerics (keeps Indic marks)."""
    out, prev_latin = [], False
    for ch in unicodedata.normalize("NFKD", text):
        if unicodedata.combining(ch) and prev_latin:
            continue
        prev_latin = ord(ch) < 0x250
        out.append(ch)
    s = "".join(c if (c.isalnum() or unicodedata.category(c)[0] == "M") else " " for c in "".join(out).lower())
    return s.split()


def combining_marks(text):
    return sum(1 for c in text if unicodedata.category(c)[0] == "M")


# ---------------------------------------------------------------- per split
def compare_split(raw, clean, sample_every):
    """Compare aligned raw/clean tables (same row order)."""
    r = {}
    n = raw.num_rows
    r["rows"] = n
    for col in ("business_name", "business_address", "country"):
        diff = pc.not_equal(raw[col], clean[col])
        r[f"{col}_verbatim_mismatches"] = count(diff)
    raw_name, raw_addr = raw["business_name"], raw["business_address"]
    norm_name, norm_addr = clean["business_name_normalized"], clean["address_normalized"]

    # country mapping and structured fields
    pairs = pc.value_counts(pc.binary_join_element_wise(raw["country"], clean["country_normalized"], "->"))
    r["country_mapping"] = {p["values"].as_py(): p["counts"].as_py() for p in pairs}
    by_country = {}
    for ctry in pc.unique(raw["country"]).to_pylist():
        m = pc.equal(raw["country"], ctry)
        k = count(m)
        by_country[ctry] = {
            "rows": k,
            "state_filled_pct": round(100 * count(pc.and_(m, pc.not_equal(clean["state_normalized"], ""))) / k, 2),
            "postal_filled_pct": round(100 * count(pc.and_(m, pc.not_equal(clean["postal_code_normalized"], ""))) / k, 2),
        }
    r["by_country"] = by_country
    scripts = pc.value_counts(clean["name_script"])
    r["name_script"] = {p["values"].as_py(): p["counts"].as_py() for p in scripts}
    indic = pc.match_substring_regex(raw_name, INDIC)
    r["raw_name_has_indic"] = count(indic)
    r["indic_but_script_not_non_latin"] = count(pc.and_(indic, pc.not_equal(clean["name_script"], "non_latin")))
    r["script_non_latin_but_no_indic"] = count(pc.and_(pc.invert(indic), pc.equal(clean["name_script"], "non_latin")))

    # emptiness / loss (whole split)
    r["name_raw_nonempty_norm_empty"] = count(pc.and_(pc.not_equal(pc.utf8_trim_whitespace(raw_name), ""),
                                                   pc.equal(norm_name, "")))
    r["address_raw_empty"] = count(pc.equal(pc.utf8_trim_whitespace(raw_addr), ""))
    r["address_raw_nonempty_norm_empty"] = count(pc.and_(pc.not_equal(pc.utf8_trim_whitespace(raw_addr), ""),
                                                      pc.equal(norm_addr, "")))
    r["name_norm_equals_lower_raw"] = count(pc.equal(norm_name, pc.utf8_lower(raw_name)))
    r["address_norm_equals_lower_raw"] = count(pc.equal(norm_addr, pc.utf8_lower(raw_addr)))
    raw_digits = pc.replace_substring_regex(raw_addr, r"[^0-9]", "")
    norm_digits = pc.replace_substring_regex(norm_addr, r"[^0-9]", "")
    r["address_digit_string_changed"] = count(pc.not_equal(raw_digits, norm_digits))
    r["address_digits_changed_only_by_zeros"] = count(pc.and_(
        pc.not_equal(raw_digits, norm_digits),
        pc.equal(pc.replace_substring(raw_digits, "0", ""), pc.replace_substring(norm_digits, "0", ""))))
    r["address_direction_word_introduced"] = count(pc.and_(
        pc.match_substring_regex(norm_addr, DIRECTIONS),
        pc.invert(pc.match_substring_regex(pc.utf8_lower(raw_addr), DIRECTIONS))))
    r["address_raw_has_NULL_placeholder"] = count(pc.match_substring(raw_addr, "<NULL>"))
    r["address_raw_has_NA_placeholder"] = count(pc.match_substring(raw_addr, "N/A"))
    r["address_norm_has_null_token"] = count(pc.match_substring_regex(norm_addr, r"\bnull\b"))
    r["address_norm_has_n_a_tokens"] = count(pc.match_substring_regex(norm_addr, r"\bn a\b"))

    # token-level statistics on a deterministic subsample
    idx = pa.array(range(0, n, sample_every))
    s_raw_name = raw_name.take(idx).to_pylist()
    s_norm_name = norm_name.take(idx).to_pylist()
    s_tokens = clean["name_tokens"].take(idx).to_pylist()
    s_raw_addr = raw_addr.take(idx).to_pylist()
    s_norm_addr = norm_addr.take(idx).to_pylist()
    s_script = clean["name_script"].take(idx).to_pylist()
    name_added, name_removed, addr_added, addr_removed = Counter(), Counter(), Counter(), Counter()
    tok_ok = name_tok_changed = addr_tok_changed = marks_lost = indic_rows = 0
    for rn, nn, tk, ra, na, sc in zip(s_raw_name, s_norm_name, s_tokens, s_raw_addr, s_norm_addr, s_script):
        tok_ok += tk == "|".join(sorted(set(nn.split())))
        a, b = set(fold_tokens(rn)), set(nn.split())
        if a != b:
            name_tok_changed += 1
            name_added.update(b - a)
            name_removed.update(a - b)
        a, b = set(fold_tokens(ra)), set(na.split())
        if a != b:
            addr_tok_changed += 1
            addr_added.update(b - a)
            addr_removed.update(a - b)
        if re.search(r"[ऀ-෿]", rn):
            indic_rows += 1
            marks_lost += combining_marks(rn) > 0 and combining_marks(nn) == 0
    k = len(s_raw_name)
    r["sample"] = {
        "rows": k,
        "name_tokens_consistent_pct": round(100 * tok_ok / max(1, k), 2),
        "name_token_set_changed_pct": round(100 * name_tok_changed / max(1, k), 2),
        "address_token_set_changed_pct": round(100 * addr_tok_changed / max(1, k), 2),
        "name_tokens_added_top": name_added.most_common(15),
        "name_tokens_removed_top": name_removed.most_common(15),
        "address_tokens_added_top": addr_added.most_common(15),
        "address_tokens_removed_top": addr_removed.most_common(15),
        "indic_name_rows": indic_rows,
        "indic_name_rows_with_all_vowel_signs_stripped": marks_lost,
    }
    return r


# ---------------------------------------------------------------- per source
def validate_source(s, raw_dir, clean_dir, sample_every):
    t0 = time.time()
    tr = read_table(os.path.join(raw_dir, "train", f"train_source{s}.tsv"), "\t", RAW_COLUMNS)
    te = read_table(os.path.join(raw_dir, "test", f"test_source{s}.tsv"), "\t", RAW_COLUMNS)
    cl = read_table(os.path.join(clean_dir, f"source{s}_clean.csv"), ",", CLEAN_COLUMNS)
    ids = cl["entity_id"]
    res = {
        "clean_file": f"source{s}_clean.csv",
        "clean_rows": cl.num_rows, "raw_train_rows": tr.num_rows, "raw_test_rows": te.num_rows,
        "clean_columns": cl.column_names,
        "clean_duplicate_ids": cl.num_rows - pc.count_distinct(ids).as_py(),
        "clean_empty_ids": count(pc.equal(ids, "")),
        "clean_wrong_prefix_ids": count(pc.invert(pc.starts_with(ids, f"S{s}-"))),
        "raw_train_test_id_overlap": count(pc.is_in(tr["entity_id"], value_set=te["entity_id"])),
    }
    all_raw = pa.concat_arrays([tr["entity_id"].combine_chunks(), te["entity_id"].combine_chunks()])
    res["clean_ids_not_in_raw"] = cl.num_rows - count(pc.is_in(ids, value_set=all_raw))
    clean_ids = ids.combine_chunks()
    res["order_matches_raw_train"] = bool(cl.num_rows == tr.num_rows
                                          and pc.all(pc.equal(clean_ids, tr["entity_id"].combine_chunks())).as_py())
    res["order_is_train_then_test"] = bool(cl.num_rows == len(all_raw)
                                           and pc.all(pc.equal(clean_ids, all_raw)).as_py())
    for split, raw in (("train", tr), ("test", te)):
        present = pc.is_in(raw["entity_id"], value_set=ids)
        n_present = count(present)
        res[f"{split}_ids_in_clean"] = n_present
        res[f"{split}_ids_missing_from_clean"] = raw.num_rows - n_present
        if n_present == 0:
            res[split] = "not present in the clean files"
            continue
        if split == "train" and res["order_matches_raw_train"]:
            aligned_raw, aligned_clean = raw, cl
        elif res["order_is_train_then_test"]:
            aligned_raw, aligned_clean = raw, cl.slice(tr.num_rows if split == "test" else 0, raw.num_rows)
        else:  # align by id on the rows that exist in both
            aligned_raw = raw.filter(present)
            aligned_clean = cl.take(pc.index_in(aligned_raw["entity_id"], value_set=ids))
        res[split] = compare_split(aligned_raw, aligned_clean, sample_every)
    res["sample_file"] = check_sample(os.path.join(clean_dir, f"source{s}_clean_SAMPLE.csv"), cl)
    res["seconds"] = round(time.time() - t0, 1)
    return res


def check_sample(path, full):
    """The *_SAMPLE.csv file: encoding, and whether its rows are exact rows of the full file."""
    rows, bad_lines = [], 0
    with open(path, "rb") as fh:
        text = []
        for line in fh:
            try:
                text.append(line.decode("utf-8"))
            except UnicodeDecodeError:
                bad_lines += 1
                text.append(line.decode("cp1252", errors="replace"))
    reader = csv.reader(io.StringIO("".join(text)))
    header = next(reader)
    rows = list(reader)
    ids = pa.array([r[0] for r in rows])
    pos = pc.index_in(ids, value_set=full["entity_id"])
    found = [p for p in pos.to_pylist() if p is not None]
    equal = 0
    if found:
        full_rows = full.take(pa.array(found)).to_pylist()
        by_id = {r["entity_id"]: r for r in full_rows}
        for r in rows:
            f = by_id.get(r[0])
            equal += f is not None and [f[c] for c in CLEAN_COLUMNS] == r
    return {"rows": len(rows), "columns_match": header == CLEAN_COLUMNS, "non_utf8_lines": bad_lines,
            "ids_in_full_file": len(found), "rows_identical_to_full_file": equal}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--clean-dir", default="data/clean")
    ap.add_argument("--sample-every", type=int, default=50)
    ap.add_argument("--out", default="output/clean_validation/report.json")
    args = ap.parse_args(argv)
    report = {"files": sorted(os.listdir(args.clean_dir))}
    for s in (1, 2, 3):
        report[f"S{s}"] = validate_source(s, args.raw_dir, args.clean_dir, args.sample_every)
        print(f"S{s} done in {report[f'S{s}']['seconds']}s", flush=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
