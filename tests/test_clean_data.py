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
        from src.preprocessing.validate_clean import check_sample, read_table, CLEAN_COLUMNS
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


if __name__ == "__main__":
    unittest.main()
