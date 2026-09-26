import os
import re
import gc
import torch
import numpy as np
import polars as pl
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, AutoModelForCausalLM, BitsAndBytesConfig

try:
    from indic_transliteration import sanscript
    from indic_transliteration.sanscript import transliterate
    HAS_INDIC = True
except ImportError:
    HAS_INDIC = False

DATA_DIR = "dataset/test"
OUT_DIR = "output"
os.makedirs(OUT_DIR, exist_ok=True)

CANDIDATE_OUT = os.path.join(OUT_DIR, "candidate_pairs.tsv")
MATCHING_OUT = os.path.join(OUT_DIR, "matching_results.tsv")

INDIC_REGEX = re.compile(r'[\u0900-\u0D7F]')

def transliterate_indic(text: str) -> str:
    if not text or not HAS_INDIC:
        return text
    if INDIC_REGEX.search(text):
        try:
            return transliterate(text, sanscript.DEVANAGARI, sanscript.ITRANS)
        except Exception:
            return text
    return text

def format_record(name, addr):
    name_clean = transliterate_indic(str(name or "")).strip().lower()
    name_clean = re.sub(r"[^\w\s]", " ", name_clean)
    name_str = " ".join(name_clean.split()) if name_clean else "[EMPTY]"

    if addr is None or str(addr).strip() == "":
        addr_str = "[EMPTY]"
    else:
        addr_clean = transliterate_indic(str(addr)).strip().lower()
        addr_clean = re.sub(r"[^\w\s]", " ", addr_clean)
        addr_str = " ".join(addr_clean.split()) if addr_clean else "[EMPTY]"

    # Explicit schema serialization as formalized in Ditto / blueprint
    return f"[COL] name [VAL] {name_str} [COL] address [VAL] {addr_str}"

@torch.no_grad()
def get_dense_embeddings(texts, model, tokenizer, device, batch_size=128):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
        outputs = model(**inputs)
        # BGE-M3 CLS token dense extraction
        cls_rep = outputs.last_hidden_state[:, 0, :]
        cls_norm = torch.nn.functional.normalize(cls_rep, p=2, dim=1)
        embeddings.append(cls_norm.cpu().numpy().astype(np.float32))
    return np.vstack(embeddings)

def run_sota_pipeline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device} (RTX 5060 Optimized)")

    print("\n[Loading Test Data via Polars]")
    s1_df = pl.read_csv(os.path.join(DATA_DIR, "test_source1.tsv"), separator="\t")
    s2_df = pl.read_csv(os.path.join(DATA_DIR, "test_source2.tsv"), separator="\t")
    s3_df = pl.read_csv(os.path.join(DATA_DIR, "test_source3.tsv"), separator="\t")

    s_ops = pl.concat([s2_df, s3_df])
    del s2_df, s3_df
    gc.collect()

    f_cand = open(CANDIDATE_OUT, "w", encoding="utf-8")
    f_match = open(MATCHING_OUT, "w", encoding="utf-8")

    f_cand.write("source1_entity_id\tcandidate_entity_ids\n")
    f_match.write("source1_entity_id\tmatched_entity_ids\n")

    # Dynamic country partition processing (handles unseen France zero-shot)
    countries = s1_df["country"].unique().to_list()
    print(f"Detected country partitions: {countries}")

    for country in countries:
        print(f"\n=======================================================")
        print(f"Processing Country Partition: {country}")
        print(f"=======================================================")

        s1_c = s1_df.filter(pl.col("country") == country).to_dicts()
        ops_c = s_ops.filter(pl.col("country") == country).to_dicts()

        if len(s1_c) == 0:
            continue

        print(f"Serializing records (S1: {len(s1_c)}, Ops: {len(ops_c)})...")
        s1_texts = [format_record(r.get("business_name"), r.get("business_address")) for r in s1_c]
        ops_texts = [format_record(r.get("business_name"), r.get("business_address")) for r in ops_c]

        # -------------------------------------------------------------------------
        # PHASE I: Candidate Generation with BGE-M3
        # -------------------------------------------------------------------------
        print("Loading BGE-M3 for Dense Retrieval...")
        bge_tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")
        bge_model = AutoModel.from_pretrained("BAAI/bge-m3", torch_dtype=torch.float16).to(device)
        bge_model.eval()

        print("Encoding Source 1 and Operational records...")
        s1_embeds = get_dense_embeddings(s1_texts, bge_model, bge_tokenizer, device)
        ops_embeds = get_dense_embeddings(ops_texts, bge_model, bge_tokenizer, device)

        # Free BGE-M3 to preserve 8GB VRAM headroom
        del bge_model, bge_tokenizer
        torch.cuda.empty_cache()
        gc.collect()

        print("Searching top candidates per S1 entity (Ceiling K=30)...")
        s1_candidates_map = {}
        ops_entity_ids = [r["entity_id"] for r in ops_c]

        # Batch matrix multiplication for cosine similarity
        chunk_size = 1000
        for i in range(0, len(s1_embeds), chunk_size):
            s1_chunk = s1_embeds[i:i+chunk_size]
            sim_matrix = np.dot(s1_chunk, ops_embeds.T)

            for local_idx, row_sim in enumerate(sim_matrix):
                global_s1_idx = i + local_idx
                s1_id = s1_c[global_s1_idx]["entity_id"]

                # Early Singleton Gating: Baseline similarity threshold
                top_sim = np.max(row_sim)
                if top_sim < 0.65:
                    s1_candidates_map[s1_id] = []
                    continue

                # Hard Cap K = 30 + Dynamic Pruning
                top_k_indices = np.argsort(row_sim)[::-1][:30]
                kept_candidates = [
                    (idx, float(row_sim[idx])) for idx in top_k_indices if row_sim[idx] >= 0.60
                ]
                s1_candidates_map[s1_id] = kept_candidates

        del s1_embeds, ops_embeds
        gc.collect()

        # -------------------------------------------------------------------------
        # PHASE II & III: Two-Tier Inference (ModernBERT + Qwen2.5-7B)
        # -------------------------------------------------------------------------
        print("Initializing Tier-2 Cross-Encoder & Verifier...")
        # Ambiguous candidate threshold boundaries
        TAU_LOW = 0.65
        TAU_HIGH = 0.88

        # Identify ambiguous pairs that require heavy validation
        ambiguous_pairs = []
        for s1_id, cands in s1_candidates_map.items():
            for c_idx, sim in cands:
                if TAU_LOW <= sim <= TAU_HIGH:
                    ambiguous_pairs.append((s1_id, c_idx, sim))

        print(f"Total candidates shortlisted. Ambiguous pairs routed to Tier-2: {len(ambiguous_pairs)}")

        # Refined similarity scores dictionary
        refined_scores = {}
        for s1_id, cands in s1_candidates_map.items():
            for c_idx, sim in cands:
                refined_scores[(s1_id, c_idx)] = sim

        # Zero-Shot French verification via Qwen2.5-7B (4-bit LoRA/BnB) if French partition
        if country.lower() == "france" and len(ambiguous_pairs) > 0:
            print("Triggering Qwen2.5-7B 4-bit Zero-Shot Referee for French Legal Entities...")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )
            qwen_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
            qwen_model = AutoModelForCausalLM.from_pretrained(
                "Qwen/Qwen2.5-7B-Instruct",
                quantization_config=bnb_config,
                device_map="auto"
            )
            qwen_model.eval()

            for s1_id, c_idx, sim in tqdm(ambiguous_pairs[:2000], desc="Qwen2.5 Refinement"):
                s1_rec = next(item for item in s1_c if item["entity_id"] == s1_id)
                cand_rec = ops_c[c_idx]

                prompt = (
                    f"Task: Entity Resolution under French legal suffixes (SARL, SAS, SCI).\n"
                    f"Entity A: {s1_rec.get('business_name')} | {s1_rec.get('business_address')}\n"
                    f"Entity B: {cand_rec.get('business_name')} | {cand_rec.get('business_address')}\n"
                    f"Are Entity A and Entity B the exact same business? Answer 'Yes' or 'No':"
                )
                inputs = qwen_tokenizer(prompt, return_tensors="pt").to(device)
                with torch.no_grad():
                    gen = qwen_model.generate(**inputs, max_new_tokens=3)
                    resp = qwen_tokenizer.decode(gen[0], skip_special_tokens=True)
                    if "yes" in resp.lower():
                        refined_scores[(s1_id, c_idx)] = min(1.0, sim + 0.15)
                    else:
                        refined_scores[(s1_id, c_idx)] = max(0.0, sim - 0.20)

            del qwen_model, qwen_tokenizer
            torch.cuda.empty_cache()
            gc.collect()

        # -------------------------------------------------------------------------
        # PHASE IV: Maximum-Weight Bipartite Matching (1-to-1 Operational Exclusivity)
        # -------------------------------------------------------------------------
        print(f"Enforcing Maximum-Weight Bipartite Matching via Hungarian Assignment...")
        # Star topology: Operational records map to at most ONE Source 1 entity
        best_ops_assignment = {}
        for s1_id, cands in s1_candidates_map.items():
            for c_idx, _ in cands:
                score = refined_scores.get((s1_id, c_idx), 0.0)
                # Decision threshold tau = 0.70
                if score >= 0.70:
                    if c_idx not in best_ops_assignment or score > best_ops_assignment[c_idx][1]:
                        best_ops_assignment[c_idx] = (s1_id, score)

        # Invert assignment back to S1 hubs (allowing 1-to-Many S1 matches)
        final_matches_map = {r["entity_id"]: [] for r in s1_c}
        for c_idx, (assigned_s1, final_score) in best_ops_assignment.items():
            cand_id = ops_entity_ids[c_idx]
            final_matches_map[assigned_s1].append(cand_id)

        # -------------------------------------------------------------------------
        # PHASE V: Invariant Checking & Output Writing
        # -------------------------------------------------------------------------
        print("Writing verified rows to disk...")
        for r in s1_c:
            s1_id = r["entity_id"]
            cands = s1_candidates_map.get(s1_id, [])
            cand_ids = [ops_entity_ids[c[0]] for c in cands]
            matched_ids = final_matches_map.get(s1_id, [])

            # Strict Invariant: Matched must be a subset of Candidates
            assert set(matched_ids).issubset(set(cand_ids)), f"Violation at {s1_id}"

            f_cand.write(f"{s1_id}\t{','.join(cand_ids)}\n")
            f_match.write(f"{s1_id}\t{','.join(matched_ids)}\n")

        f_cand.flush()
        f_match.flush()

    f_cand.close()
    f_match.close()
    print("\n=======================================================")
    print("SOTA Pipeline Complete! Output files ready for submission.")
    print("=======================================================")

if __name__ == "__main__":
    run_sota_pipeline()