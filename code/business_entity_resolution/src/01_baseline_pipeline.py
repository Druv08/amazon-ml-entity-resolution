import os
import re
import polars as pl
from rapidfuzz import fuzz
from tqdm import tqdm

DATA_DIR = "dataset/test"
OUT_DIR = "output"
os.makedirs(OUT_DIR, exist_ok=True)

CANDIDATE_OUT = os.path.join(OUT_DIR, "candidate_pairs.tsv")
MATCHING_OUT = os.path.join(OUT_DIR, "matching_results.tsv")

def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())

def run_pipeline():
    print("Loading test datasets via Polars...")
    s1_df = pl.read_csv(os.path.join(DATA_DIR, "test_source1.tsv"), separator="\t")
    s2_df = pl.read_csv(os.path.join(DATA_DIR, "test_source2.tsv"), separator="\t")
    s3_df = pl.read_csv(os.path.join(DATA_DIR, "test_source3.tsv"), separator="\t")

    print(f"Loaded: S1 ({len(s1_df)}), S2 ({len(s2_df)}), S3 ({len(s3_df)})")

    # Combine S2 and S3 operational records
    s_ops = pl.concat([s2_df, s3_df])
    del s2_df, s3_df  # Free memory immediately

    # Open both output files for streaming
    with open(CANDIDATE_OUT, "w", encoding="utf-8") as f_cand, \
         open(MATCHING_OUT, "w", encoding="utf-8") as f_match:

        # Write required headers
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")
        f_match.write("source1_entity_id\tmatched_entity_ids\n")

        # Process country by country (Strict Country Blocking)
        countries = s1_df["country"].unique().to_list()
        print(f"Countries found in test set: {countries}")

        for country in countries:
            print(f"\n--- Processing country: {country} ---")
            
            s1_country = s1_df.filter(pl.col("country") == country)
            ops_country = s_ops.filter(pl.col("country") == country)

            if len(s1_country) == 0:
                continue

            print(f"Building fast index for {len(ops_country)} operational records...")
            token_index = {}
            ops_records = ops_country.to_dicts()

            for idx, row in enumerate(ops_records):
                clean_name = normalize_text(row.get("business_name", ""))
                tokens = clean_name.split()
                if tokens:
                    first_token = tokens[0]
                    if first_token not in token_index:
                        token_index[first_token] = []
                    token_index[first_token].append(idx)

            print(f"Evaluating matches for {len(s1_country)} Source 1 records...")
            s1_records = s1_country.to_dicts()

            for s1_row in tqdm(s1_records, desc=f"Matching {country}"):
                s1_id = s1_row["entity_id"]
                s1_name_clean = normalize_text(s1_row.get("business_name", ""))
                s1_addr_clean = normalize_text(s1_row.get("business_address", ""))
                
                s1_tokens = s1_name_clean.split()
                candidate_indices = set()

                for tok in s1_tokens[:2]:
                    if tok in token_index:
                        candidate_indices.update(token_index[tok])

                candidate_ids = []
                matched_ids = []

                # Cap candidates per entity to keep memory and runtime bounded
                for c_idx in list(candidate_indices)[:50]:
                    cand = ops_records[c_idx]
                    cand_id = cand["entity_id"]
                    candidate_ids.append(cand_id)

                    cand_name_clean = normalize_text(cand.get("business_name", ""))
                    cand_addr_clean = normalize_text(cand.get("business_address", ""))

                    name_score = fuzz.token_set_ratio(s1_name_clean, cand_name_clean)
                    addr_score = fuzz.token_set_ratio(s1_addr_clean, cand_addr_clean)

                    # High threshold to prevent false merges on singletons
                    if name_score >= 88 and addr_score >= 65:
                        matched_ids.append(cand_id)

                cand_str = ",".join(candidate_ids)
                match_str = ",".join(matched_ids)

                f_cand.write(f"{s1_id}\t{cand_str}\n")
                f_match.write(f"{s1_id}\t{match_str}\n")

    print("\nProcessing complete! Output files generated in 'output/' directory.")

if __name__ == "__main__":
    run_pipeline()