"""Tests for the clean-data validator and loader (synthetic data only)."""

import os
import shutil
import tempfile
import unittest

RAW_HEADER = "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
CLEAN_HEADER = ("entity_id,business_name,business_name_normalized,name_tokens,business_address,"
                "address_normalized,country,country_normalized,state_normalized,postal_code_normalized,"
                "name_script\n")


def write(path, text, encoding="utf-8"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding=encoding, newline="\n") as f:
        f.write(text)


def make_dirs(root):
    """Raw train/test for S1-S3 and a clean copy that covers the train split only."""
    raw, clean = os.path.join(root, "raw"), os.path.join(root, "clean")
    for s in (1, 2, 3):
        train = [(f"S{s}-1", "Acme Tools Inc", "12 Oak St, Dayton, OH", "US"),
                 (f"S{s}-2", "NA", "5 Main Rd, Pune", "India"),
                 (f"S{s}-3", "Nétwork Labs", "Block S-01, Pune", "India")]
        test = [(f"S{s}-9", "Other Co", "1 Rue X", "France")]
        write(os.path.join(raw, "train", f"train_source{s}.tsv"),
              RAW_HEADER + "".join("\t".join(r) + "\n" for r in train))
        write(os.path.join(raw, "test", f"test_source{s}.tsv"),
              RAW_HEADER + "".join("\t".join(r) + "\n" for r in test))
        rows = [
            f'S{s}-1,Acme Tools Inc,acme tools incorporated,acme|incorporated|tools,"12 Oak St, Dayton, OH",'
            f'12 oak street dayton oh,US,US,OH,,latin\n',
            f'S{s}-2,,,,"5 Main Rd, Pune",5 main road pune,India,IN,,,latin\n',  # "NA" name emptied
            f'S{s}-3,Nétwork Labs,nétwork laboratories,laboratories|nétwork,"Block S-01, Pune",'
            f'block south 1 pune,India,IN,,,non_latin\n',
        ]
        write(os.path.join(clean, f"source{s}_clean.csv"), CLEAN_HEADER + "".join(rows))
        write(os.path.join(clean, f"source{s}_clean_SAMPLE.csv"), CLEAN_HEADER + rows[0])
    return raw, clean


class ValidateCleanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.raw, cls.clean = make_dirs(cls.tmp)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_train_only_clean_file(self):
        from src.preprocessing.validate_clean import validate_source
        r = validate_source(1, self.raw, self.clean, sample_every=1)
        self.assertEqual(r["clean_rows"], 3)
        self.assertEqual(r["clean_duplicate_ids"], 0)
        self.assertTrue(r["order_matches_raw_train"])
        self.assertEqual(r["train_ids_missing_from_clean"], 0)
        self.assertEqual(r["test_ids_missing_from_clean"], 1)
        self.assertEqual(r["test"], "not present in the clean files")
        t = r["train"]
        self.assertEqual(t["business_name_verbatim_mismatches"], 1)  # "NA" -> ""
        self.assertEqual(t["name_raw_nonempty_norm_empty"], 1)
        self.assertEqual(t["country_mapping"], {"US->US": 1, "India->IN": 2})
        self.assertEqual(t["script_non_latin_but_no_indic"], 1)  # accented Latin tagged non_latin
        self.assertEqual(t["address_direction_word_introduced"], 1)  # "S-01" -> "south 1"
        self.assertEqual(t["address_digits_changed_only_by_zeros"], 1)
        self.assertEqual(r["sample_file"]["rows_identical_to_full_file"], 1)

    def test_sample_with_cp1252_line(self):
        from src.preprocessing.clean_data import CLEAN_COLUMNS, read_table
        from src.preprocessing.validate_clean import check_sample
        full = read_table(os.path.join(self.clean, "source2_clean.csv"), ",", CLEAN_COLUMNS)
        path = os.path.join(self.tmp, "bad_sample.csv")
        with open(path, "wb") as f:
            f.write(CLEAN_HEADER.encode("utf-8"))
            f.write('S2-3,Nétwork Labs,nétwork laboratories,laboratories|nétwork,"Block S-01, Pune",'
                    'block south 1 pune,India,IN,,,non_latin\n'.encode("cp1252"))
        r = check_sample(path, full)
        self.assertEqual(r["non_utf8_lines"], 1)
        self.assertEqual(r["ids_in_full_file"], 1)
        self.assertEqual(r["rows_identical_to_full_file"], 1)


class LoaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.raw, cls.clean = make_dirs(cls.tmp)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def load(self, split, source, mode):
        from src.preprocessing.clean_data import load_source
        return load_source(split, source, mode, raw_dir=self.raw, clean_dir=self.clean)

    def test_raw_mode(self):
        df = self.load("train", 1, "raw")
        self.assertEqual(list(df.columns), ["entity_id", "country", "raw_name", "raw_address", "name", "address"])
        self.assertEqual(df["name"].tolist(), ["Acme Tools Inc", "NA", "Nétwork Labs"])
        self.assertEqual(df["name"].tolist(), df["raw_name"].tolist())

    def test_clean_mode_uses_normalised_text_and_keeps_raw(self):
        df = self.load("train", 2, "clean")
        self.assertEqual(df["name"].tolist(), ["acme tools incorporated", "", "nétwork laboratories"])
        self.assertEqual(df["address"].tolist()[0], "12 oak street dayton oh")
        self.assertEqual(df["raw_name"].tolist()[1], "NA")  # raw text survives P1's emptied value
        for col in ("clean_name", "clean_address", "clean_name_tokens", "clean_country", "clean_state",
                    "clean_postal_code", "clean_name_script"):
            self.assertIn(col, df.columns)

    def test_raw_plus_clean_mode(self):
        df = self.load("train", 3, "raw+clean")
        self.assertEqual(df["name"].tolist(), df["raw_name"].tolist())
        self.assertEqual(df["clean_name"].tolist()[0], "acme tools incorporated")
        self.assertEqual(df["clean_country"].tolist(), ["US", "IN", "IN"])

    def test_test_split_has_no_clean_data(self):
        from src.preprocessing.clean_data import CleanDataUnavailable, clean_available
        self.assertFalse(clean_available("test", 1, self.clean))
        self.assertTrue(clean_available("train", 1, self.clean))
        with self.assertRaises(CleanDataUnavailable):
            self.load("test", 1, "clean")
        self.assertEqual(self.load("test", 1, "raw")["name"].tolist(), ["Other Co"])

    def test_misaligned_clean_file_is_rejected(self):
        from src.preprocessing.clean_data import load_source
        bad = os.path.join(self.tmp, "bad_clean")
        os.makedirs(bad, exist_ok=True)
        with open(os.path.join(self.clean, "source1_clean.csv"), encoding="utf-8") as f:
            lines = f.readlines()
        write(os.path.join(bad, "source1_clean.csv"), lines[0] + lines[2] + lines[1] + lines[3])  # swapped rows
        with self.assertRaises(ValueError):
            load_source("train", 1, "clean", raw_dir=self.raw, clean_dir=bad)

    def test_unknown_mode(self):
        with self.assertRaises(ValueError):
            self.load("train", 1, "cleaned")


if __name__ == "__main__":
    unittest.main()
