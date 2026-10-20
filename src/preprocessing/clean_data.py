"""Load a source file with raw text, P1's cleaned text, or both.

    from src.preprocessing.clean_data import load_source

    df = load_source("train", 2, mode="raw+clean")

Every mode keeps the raw text (``raw_name``, ``raw_address``). ``name`` / ``address``
hold the text a consumer should use for the chosen mode:

    mode        name / address          extra columns
    raw         raw text                -
    clean       P1 normalised text      clean_* columns
    raw+clean   raw text                clean_* columns

Clean columns: ``clean_name``, ``clean_address``, ``clean_name_tokens``,
``clean_country``, ``clean_state``, ``clean_postal_code``, ``clean_name_script``.

P1's files cover the train split only, row-aligned with the raw train files (see
docs/clean_data_validation.md). The alignment is verified on every load, and asking
for clean data of a split that has none raises ``CleanDataUnavailable``.
"""

import os

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv

MODES = ("raw", "clean", "raw+clean")
RAW_COLUMNS = ["entity_id", "business_name", "business_address", "country"]
CLEAN_COLUMNS = ["entity_id", "business_name", "business_name_normalized", "name_tokens", "business_address",
                 "address_normalized", "country", "country_normalized", "state_normalized",
                 "postal_code_normalized", "name_script"]
CLEAN_RENAME = {
    "business_name_normalized": "clean_name",
    "address_normalized": "clean_address",
    "name_tokens": "clean_name_tokens",
    "country_normalized": "clean_country",
    "state_normalized": "clean_state",
    "postal_code_normalized": "clean_postal_code",
    "name_script": "clean_name_script",
}


class CleanDataUnavailable(FileNotFoundError):
    pass


def read_table(path, delimiter, columns):
    """Read a raw TSV (``delimiter='\\t'``, no quoting) or a clean CSV as all-string Arrow columns."""
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


def raw_path(split, source, raw_dir="data/raw"):
    return os.path.join(raw_dir, split, f"{split}_source{source}.tsv")


def clean_path(split, source, clean_dir="data/clean"):
    """Path of P1's clean file for ``split`` (only train exists so far)."""
    if split == "train":
        return os.path.join(clean_dir, f"source{source}_clean.csv")
    return os.path.join(clean_dir, f"{split}_source{source}_clean.csv")  # expected name for future test files


def clean_available(split, source, clean_dir="data/clean"):
    return os.path.exists(clean_path(split, source, clean_dir))


def load_table(split, source, mode="raw", raw_dir="data/raw", clean_dir="data/clean"):
    """Arrow table in the layout described in the module docstring."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    raw = read_table(raw_path(split, source, raw_dir), "\t", RAW_COLUMNS)
    cols = {
        "entity_id": raw["entity_id"],
        "country": raw["country"],
        "raw_name": raw["business_name"],
        "raw_address": raw["business_address"],
    }
    if mode != "raw":
        path = clean_path(split, source, clean_dir)
        if not os.path.exists(path):
            raise CleanDataUnavailable(f"no P1 clean data for {split} source{source} ({path}); use mode='raw'")
        clean = read_table(path, ",", CLEAN_COLUMNS)
        if clean.num_rows != raw.num_rows or not pc.all(
                pc.equal(clean["entity_id"].combine_chunks(), raw["entity_id"].combine_chunks())).as_py():
            raise ValueError(f"{path} is not row-aligned with {raw_path(split, source, raw_dir)}")
        for src, dst in CLEAN_RENAME.items():
            cols[dst] = clean[src]
    if mode == "clean":
        cols["name"], cols["address"] = cols["clean_name"], cols["clean_address"]
    else:
        cols["name"], cols["address"] = cols["raw_name"], cols["raw_address"]
    return pa.table(cols)


def load_source(split, source, mode="raw", raw_dir="data/raw", clean_dir="data/clean"):
    """pandas DataFrame (pyarrow-backed strings) in the layout described in the module docstring."""
    table = load_table(split, source, mode, raw_dir, clean_dir)
    return table.to_pandas(types_mapper={pa.string(): pd.StringDtype("pyarrow")}.get)
