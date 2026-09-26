import os
import re
import polars as pl
from tqdm import tqdm
from rapidfuzz import fuzz

try:
    from indic_transliteration import sanscript
    from indic_transliteration.sanscript import SchemeMap, SCHEMES, transliterate
    HAS_INDIC = True
except ImportError:
    HAS_INDIC = False

DATA_DIR = "dataset/test"
OUT_DIR = "output"
os.makedirs(OUT_DIR, exist_ok=True)

CANDIDATE_OUT = os.path.join(OUT_DIR, "candidate_pairs.tsv")
MATCHING_OUT = os.path.join(OUT_DIR, "matching_results.tsv")

# Indic Unicode detection ranges (Devanagari, Bengali, Gurmukhi, Gujarati, Oriya, Tamil, Telugu, Kannada, Malayalam)
INDIC_REGEX = re.compile(r'[\u0900-\u0D7F]')

def transliterate_indic(text: str) -> str:
    """Converts native Indic script strings into Latin/ITRANS representations."""
    if not text or not HAS_INDIC:
        return text
    if INDIC_REGEX.search(text):
        try:
            # Fallback transliteration from Devanagari/Tamil to ITRANS Latin
            return transliterate(text, sanscript.DEVANAGARI, sanscript.ITRANS)
        except Exception:
            return text
    return text

def normalize_text(text: str) -> str:
    """Lowercases, handles Indic transliteration, and cleans punctuation."""
    if text is None:
        return "[EMPTY]"
    text = str(text).strip()
    if not text:
        return "[EMPTY]"
    text = transliterate_indic(text)
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    cleaned = " ".join(text.split())
    return cleaned if cleaned else "[EMPTY]"

def run_tier1():
    print("Loading test datasets via Polars streaming...")
    s1_df = pl.read_csv(os.path.join(DATA_DIR, "test_source1.tsv"), separator="\t")
    s2_df = pl.read_csv(os.path.join(DATA_DIR, "test_source2.tsv"), separator="\t")
    s3_df = pl.read_csv(os.path.join(DATA_DIR, "test_source3.tsv"), separator="\t")

    print(f"Loaded: S1 ({len(s1_df)}), S2 ({len(s2_df)}), S3 ({len(s3_df)})")

    # Merge operational records S2 and S3 into one pool
    s_ops = pl.concat([s2_df, s3_df])
    del s2_df, s3_df  # Free memory immediately

    # Open output streams
    with open(CANDIDATE_OUT, "w", encoding="utf-8") as f_cand, \
         open(MATCHING_OUT, "w", encoding="utf-8") as f_match:

        # Write exact required headers
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")
        f_match.write("source1_entity_id\tmatched_entity_ids\n")

        # Dynamic country partitioning (Never hardcode country names; handles France zero-shot)
        countries = s1_df["country"].unique().to_list()
        print(f"Countries detected dynamically: {countries}")

        for country in countries:
            print(f"\n==========================================")
            print(f"Processing partition: country = {country}")
            print(f"==========================================")

            s1_country = s1_df.filter(pl.col("country") == country)
            ops_country = s_ops.filter(pl.col("country") == country)

            if len(s1_country) == 0:
                continue

            print(f"Indexing {len(ops_country)} operational records...")
            token_index = {}
            ops_records = ops_country.to_dicts()

            for idx, row in enumerate(ops_records):
                clean_name = normalize_text(row.get("business_name"))
                tokens = [t for t in clean_name.split() if t != "[EMPTY]"]
                if tokens:
                    # Index on the first two informative tokens
                    for tok in tokens[:2]:
                        if len(tok) > 2:  # Avoid single letters
                            if tok not in token_index:
                                token_index[tok] = []
                            token_index[tok].append(idx)

            print(f"Generating candidates for {len(s1_country)} Source 1 records...")
            s1_records = s1_country.to_dicts()

            for s1_row in tqdm(s1_records, desc=f"Tier-1 [{country}]"):
                s1_id = s1_row["entity_id"]
                s1_name = normalize_text(s1_row.get("business_name"))
                s1_addr = normalize_text(s1_row.get("business_address"))

                s1_tokens = [t for t in s1_name.split() if t != "[EMPTY]"]
                candidate_pool = set()

                for tok in s1_tokens[:2]:
                    if tok in token_index:
                        candidate_pool.update(token_index[tok])

                # Dynamic scoring and ranking
                scored_candidates = []
                for c_idx in candidate_pool:
                    cand = ops_records[c_idx]
                    cand_name = normalize_text(cand.get("business_name"))
                    cand_addr = normalize_text(cand.get("business_address"))

                    name_sim = fuzz.token_set_ratio(s1_name, cand_name)

                    # Address-less fallback path
                    if s1_addr == "[EMPTY]" or cand_addr == "[EMPTY]":
                        addr_sim = 50.0  # Neutral baseline when address is missing
                    else:
                        addr_sim = fuzz.token_set_ratio(s1_addr, cand_addr)

                    # Blended score
                    blended_score = 0.7 * name_sim + 0.3 * addr_sim

                    # Baseline similarity cutoff
                    if blended_score >= 60.0:
                        scored_candidates.append((cand["entity_id"], blended_score, name_sim, addr_sim))

                # Sort by score descending
                scored_candidates.sort(key=lambda x: x[1], reverse=True)

                # Early Singleton Gating:
                # If there are no candidates or top score fails initial baseline, mark as empty singleton
                if not scored_candidates or scored_candidates[0][1] < 68.0:
                    f_cand.write(f"{s1_id}\t\n")
                    f_match.write(f"{s1_id}\t\n")
                    continue

                # Hard Cap: Retain maximum K = 30 candidates
                top_candidates = scored_candidates[:30]
                cand_ids = [c[0] for c in top_candidates]

                # High-precision matching threshold (ensures matches are a strict subset of candidates)
                matched_ids = []
                for cid, b_score, n_sim, a_sim in top_candidates:
                    # Stricter criterion for confident match acceptance
                    if n_sim >= 88.0 and (a_sim >= 65.0 or a_sim == 50.0):
                        matched_ids.append(cid)

                # Strict subset verification
                assert set(matched_ids).issubset(set(cand_ids)), f"Subset invariant failed on {s1_id}"

                # Write tab-separated lines
                f_cand.write(f"{s1_id}\t{','.join(cand_ids)}\n")
                f_match.write(f"{s1_id}\t{','.join(matched_ids)}\n")

    print("\nTier-1 Candidate Generation & Baseline Matching Complete!")
    print(f"Outputs written to: {CANDIDATE_OUT} and {MATCHING_OUT}")

if __name__ == "__main__":
    run_tier1()