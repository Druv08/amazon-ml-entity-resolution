import os
import re
import gc
import joblib
import torch
import numpy as np
import polars as pl
from tqdm import tqdm
from rapidfuzz import fuzz, distance
from metaphone import doublemetaphone
from transformers import AutoTokenizer, AutoModel

try:
    from indic_transliteration import sanscript
    from indic_transliteration.sanscript import transliterate
    HAS_INDIC = True
except ImportError:
    HAS_INDIC = False

DATA_DIR = "dataset/test"
OUT_DIR = "output"
CACHE_DIR = "temp_cache"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

CORP_STOPWORDS = {
    'private', 'limited', 'ltd', 'pvt', 'inc', 'llc', 'corp', 'corporation',
    'services', 'solutions', 'enterprises', 'company', 'co', 'holdings', 'group',
    'public', 'sarl', 'sas', 'sci', 'ste', 'gmbh', 'sa', 'bv', 'nv'
}

US_STATES = {
    'al', 'ak', 'az', 'ar', 'ca', 'co', 'ct', 'de', 'fl', 'ga', 'hi', 'id', 'il',
    'in', 'ia', 'ks', 'ky', 'la', 'me', 'md', 'ma', 'mi', 'mn', 'ms', 'mo', 'mt',
    'ne', 'nv', 'nh', 'nj', 'nm', 'ny', 'nc', 'nd', 'oh', 'ok', 'or', 'pa', 'ri',
    'sc', 'sd', 'tn', 'tx', 'ut', 'vt', 'va', 'wa', 'wv', 'wi', 'wy'
}

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

def extract_meta(name: str, addr: str):
    n_clean = transliterate_indic(str(name or "")).lower()
    n_clean = n_clean.replace(".com", "").replace("@", " ")
    n_clean = re.sub(r"[^\w\s]", " ", n_clean)
    n_tokens = n_clean.split()
    n_core = [t for t in n_tokens if t not in CORP_STOPWORDS]
    core_name = " ".join(n_core) if n_core else " ".join(n_tokens)
    raw_name = "".join(n_core) if n_core else "".join(n_tokens)

    meta_keys = set()
    for t in (n_core if n_core else n_tokens)[:3]:
        dm = doublemetaphone(t)
        if dm[0]: meta_keys.add(dm[0])
        if dm[1]: meta_keys.add(dm[1])

    a_clean = transliterate_indic(str(addr or "")).lower()
    a_clean = re.sub(r"[^\w\s]", " ", a_clean)
    a_tokens = a_clean.split()
    nums = {str(int(t)) for t in a_tokens if t.isdigit()}
    states = {t for t in a_tokens if t in US_STATES}
    core_addr = " ".join([t for t in a_tokens if t not in nums and len(t) > 1])

    return (core_name, raw_name, meta_keys), (core_addr, nums, states)

def compute_features_fast(s1_meta, op_meta, dense_sc, rank_pos, margin_delta):
    (s1_cn, s1_rn, s1_meta_keys), (s1_ca, s1_nums, s1_st) = s1_meta
    (op_cn, op_rn, op_meta_keys), (op_ca, op_nums, op_st) = op_meta

    f_dense = float(dense_sc)
    f_rank = 1.0 / (float(rank_pos) + 1.0)
    f_margin = float(margin_delta)

    f_name_sort = fuzz.token_sort_ratio(s1_cn, op_cn) / 100.0
    f_name_set = fuzz.token_set_ratio(s1_cn, op_cn) / 100.0
    f_name_ratio = fuzz.ratio(s1_cn, op_cn) / 100.0
    f_name_partial = fuzz.partial_ratio(s1_cn, op_cn) / 100.0
    f_name_jw = distance.JaroWinkler.similarity(s1_cn, op_cn)

    f_exact_sub = 1.0 if (s1_rn and op_rn and (s1_rn in op_rn or op_rn in s1_rn)) else 0.0
    f_phonetic = 1.0 if (s1_meta_keys and op_meta_keys and bool(s1_meta_keys.intersection(op_meta_keys))) else 0.0

    f_addr_sort = fuzz.token_sort_ratio(s1_ca, op_ca) / 100.0 if (s1_ca and op_ca) else 0.0
    f_addr_set = fuzz.token_set_ratio(s1_ca, op_ca) / 100.0 if (s1_ca and op_ca) else 0.0
    f_addr_jw = distance.JaroWinkler.similarity(s1_ca, op_ca) if (s1_ca and op_ca) else 0.0

    f_num_match = 1.0 if (s1_nums and op_nums and bool(s1_nums.intersection(op_nums))) else 0.0
    f_num_diff = 1.0 if (s1_nums and op_nums and not bool(s1_nums.intersection(op_nums))) else 0.0
    num_jaccard = len(s1_nums.intersection(op_nums)) / max(len(s1_nums.union(op_nums)), 1)

    f_state_conflict = 1.0 if (s1_st and op_st and not bool(s1_st.intersection(op_st))) else 0.0
    f_state_match = 1.0 if (s1_st and op_st and bool(s1_st.intersection(op_st))) else 0.0

    len_diff = abs(len(s1_cn) - len(op_cn)) / max(len(s1_cn), len(op_cn), 1)
    tok_diff = abs(len(s1_cn.split()) - len(op_cn.split()))
    
    f_composite_lex_dense = f_dense * f_name_sort
    f_composite_addr_dense = f_dense * f_addr_sort

    return [
        f_dense, f_rank, f_margin,
        f_name_sort, f_name_set, f_name_ratio, f_name_partial, f_name_jw,
        f_exact_sub, f_phonetic,
        f_addr_sort, f_addr_set, f_addr_jw,
        f_num_match, f_num_diff, num_jaccard,
        f_state_conflict, f_state_match,
        len_diff, tok_diff,
        f_composite_lex_dense, f_composite_addr_dense
    ]

@torch.no_grad()
def encode_to_disk(texts, model, tokenizer, device, file_path, batch_size=512):
    n_samples = len(texts)
    dim = 1024
    expected_bytes = n_samples * dim * 2

    if os.path.exists(file_path) and os.path.getsize(file_path) == expected_bytes:
        print(f"  [Found existing cache] Reusing: {file_path} ({n_samples} vectors)")
        return np.memmap(file_path, dtype='float16', mode='r', shape=(n_samples, dim))

    print(f"  [Generating cache] Encoding {n_samples} items to {file_path}...")
    mmap = np.memmap(file_path, dtype='float16', mode='w+', shape=(n_samples, dim))
    for i in tqdm(range(0, n_samples, batch_size), desc="Encoding", leave=False):
        b = texts[i:i+batch_size]
        inp = tokenizer(b, padding=True, truncation=True, max_length=96, return_tensors="pt").to(device)
        out = model(**inp)
        cls_norm = torch.nn.functional.normalize(out.last_hidden_state[:, 0, :], p=2, dim=1).half()
        mmap[i:i+len(b)] = cls_norm.cpu().numpy()
    mmap.flush()
    return mmap

def run_test_inference(target_countries=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    print("[Loading Models]")
    models = [joblib.load(f"lgb_fold_{f}.pkl") for f in range(5)]

    print("[Loading Datasets]")
    s1_df = pl.read_csv(os.path.join(DATA_DIR, "test_source1.tsv"), separator="\t")
    s2_df = pl.read_csv(os.path.join(DATA_DIR, "test_source2.tsv"), separator="\t")
    s3_df = pl.read_csv(os.path.join(DATA_DIR, "test_source3.tsv"), separator="\t")
    s_ops = pl.concat([s2_df, s3_df])
    del s2_df, s3_df
    gc.collect()

    all_countries = s1_df["country"].unique().to_list()
    countries = [c for c in all_countries if c in target_countries] if target_countries else all_countries
    print(f"Target Partitions: {countries}")

    tag = "_".join(countries) if target_countries else "final"
    cand_path = os.path.join(OUT_DIR, f"candidate_pairs_{tag}.tsv" if target_countries else "candidate_pairs.tsv")
    match_path = os.path.join(OUT_DIR, f"matching_results_{tag}.tsv" if target_countries else "matching_results.tsv")

    f_cand = open(cand_path, "w", encoding="utf-8")
    f_match = open(match_path, "w", encoding="utf-8")
    f_cand.write("source1_entity_id\tcandidate_entity_ids\n")
    f_match.write("source1_entity_id\tmatched_entity_ids\n")

    tok = AutoTokenizer.from_pretrained("BAAI/bge-m3", local_files_only=True)
    mod = AutoModel.from_pretrained("BAAI/bge-m3", torch_dtype=torch.float16, local_files_only=True).to(device).eval()

    def process_df(df):
        texts, meta = [], []
        names = df["business_name"].to_list()
        addrs = df["business_address"].to_list()
        for n, a in zip(names, addrs):
            m = extract_meta(n, a)
            meta.append(m)
            texts.append(f"[COL] name [VAL] {m[0][0]} [COL] address [VAL] {str(a or '').lower()}")
        return texts, meta

    ANCHOR_THRESH = 0.84
    SISTER_THRESH = 0.76

    for country in countries:
        print(f"\n==================================================")
        print(f"Processing Partition: {country}")
        print(f"==================================================")

        s1_c_df = s1_df.filter(pl.col("country") == country)
        ops_c_df = s_ops.filter(pl.col("country") == country)
        n_s1 = len(s1_c_df)
        n_ops = len(ops_c_df)
        if n_s1 == 0 or n_ops == 0:
            continue

        s1_ids = s1_c_df["entity_id"].to_list()
        ops_ids = ops_c_df["entity_id"].to_list()

        s1_texts, s1_meta = process_df(s1_c_df)
        ops_texts, ops_meta = process_df(ops_c_df)
        del s1_c_df, ops_c_df
        gc.collect()

        s1_mmap_path = os.path.join(CACHE_DIR, f"s1_{country}.dat")
        ops_mmap_path = os.path.join(CACHE_DIR, f"ops_{country}.dat")

        ops_mmap = encode_to_disk(ops_texts, mod, tok, device, ops_mmap_path, batch_size=512)
        del ops_texts
        gc.collect()

        s1_mmap = encode_to_disk(s1_texts, mod, tok, device, s1_mmap_path, batch_size=512)
        del s1_texts
        gc.collect()

        # =========================================================================
        # INVERTED RETRIEVAL:
        # Load S1 in large VRAM blocks (up to 150k S1 entities per GPU slice).
        # Stream Ops in single sequential pass from disk to avoid SSD thrashing.
        # Top-15 tracked directly on GPU via torch.topk (no Python loops / argsorts).
        # =========================================================================
        topk_scores_all = np.full((n_s1, 15), -1.0, dtype=np.float32)
        topk_indices_all = np.full((n_s1, 15), -1, dtype=np.int64)

        s1_block_size = min(n_s1, 150_000)  # ~300MB VRAM per block
        ops_stream_batch = 32_768           # ~64MB VRAM per batch

        for s1_start in range(0, n_s1, s1_block_size):
            s1_end = min(s1_start + s1_block_size, n_s1)
            cur_s1_len = s1_end - s1_start
            print(f"\n[GPU] Loading S1 slice [{s1_start:,} : {s1_end:,}] to CUDA...")

            s1_tensor = torch.from_numpy(s1_mmap[s1_start:s1_end]).to(device=device, dtype=torch.float16)

            best_scores_gpu = torch.full((cur_s1_len, 15), -1.0, dtype=torch.float16, device=device)
            best_indices_gpu = torch.full((cur_s1_len, 15), -1, dtype=torch.int64, device=device)

            pbar = tqdm(total=n_ops, desc=f"Streaming Ops ({country})", leave=False)
            for ops_start in range(0, n_ops, ops_stream_batch):
                ops_end = min(ops_start + ops_stream_batch, n_ops)
                ops_chunk = torch.from_numpy(ops_mmap[ops_start:ops_end]).to(device=device, dtype=torch.float16)

                # Matrix multiplication on GPU: (cur_s1_len, 1024) @ (1024, ops_chunk_len)
                sims = torch.matmul(s1_tensor, ops_chunk.T)

                k_chunk = min(15, ops_end - ops_start)
                chunk_scores, chunk_indices = torch.topk(sims, k=k_chunk, dim=1, largest=True, sorted=False)
                chunk_indices = chunk_indices + ops_start

                combined_scores = torch.cat([best_scores_gpu, chunk_scores], dim=1)
                combined_indices = torch.cat([best_indices_gpu, chunk_indices], dim=1)

                best_scores_gpu, top_sel = torch.topk(combined_scores, k=15, dim=1, largest=True, sorted=True)
                best_indices_gpu = torch.gather(combined_indices, 1, top_sel)

                pbar.update(ops_end - ops_start)

            pbar.close()

            topk_scores_all[s1_start:s1_end] = best_scores_gpu.cpu().float().numpy()
            topk_indices_all[s1_start:s1_end] = best_indices_gpu.cpu().numpy()

            del s1_tensor, best_scores_gpu, best_indices_gpu
            torch.cuda.empty_cache()

        print(f"\n[Scoring & Reranking {country} Candidates with LightGBM Ensemble...]")
        # GBDT scoring in fast memory-friendly chunks
        eval_chunk_size = 5_000
        for i in tqdm(range(0, n_s1, eval_chunk_size), desc=f"GBDT Eval {country}"):
            c_end = min(i + eval_chunk_size, n_s1)
            c_len = c_end - i

            chunk_pairs = []
            pair_meta = []

            for local_idx in range(c_len):
                global_idx = i + local_idx
                sid = s1_ids[global_idx]
                row_vals = topk_scores_all[global_idx]
                row_inds = topk_indices_all[global_idx]

                top1_sc = row_vals[0]
                top2_sc = row_vals[1] if len(row_vals) > 1 else 0.0
                margin_delta = top1_sc - top2_sc

                for rank_pos, (score, c_idx) in enumerate(zip(row_vals, row_inds)):
                    if c_idx != -1 and score >= 0.62:
                        c_idx_int = int(c_idx)
                        feats = compute_features_fast(s1_meta[global_idx], ops_meta[c_idx_int], score, rank_pos, margin_delta)
                        chunk_pairs.append(feats)
                        pair_meta.append((local_idx, ops_ids[c_idx_int]))

            chunk_matches = {local_idx: [] for local_idx in range(c_len)}
            if chunk_pairs:
                X_chunk = np.array(chunk_pairs)
                p_sum = np.zeros(len(X_chunk), dtype=np.float32)
                for m in models:
                    p_sum += m.predict_proba(X_chunk)[:, 1]
                probs = p_sum / len(models)

                has_anchor_local = set()
                for (local_idx, op_id), prob in zip(pair_meta, probs):
                    if prob >= ANCHOR_THRESH:
                        has_anchor_local.add(local_idx)
                        chunk_matches[local_idx].append((op_id, prob))
                    elif local_idx in has_anchor_local and prob >= SISTER_THRESH:
                        chunk_matches[local_idx].append((op_id, prob))

            for local_idx in range(c_len):
                global_idx = i + local_idx
                sid = s1_ids[global_idx]
                row_inds = topk_indices_all[global_idx]
                row_vals = topk_scores_all[global_idx]

                cands = [ops_ids[int(c)] for score, c in zip(row_vals, row_inds) if c != -1 and score >= 0.55]
                matches = [m[0] for m in chunk_matches[local_idx] if m[0] in cands]

                f_cand.write(f"{sid}\t{','.join(cands)}\n")
                f_match.write(f"{sid}\t{','.join(matches)}\n")

            f_cand.flush()
            f_match.flush()

        del ops_mmap, s1_mmap, ops_ids, s1_ids, topk_scores_all, topk_indices_all
        gc.collect()

    f_cand.close()
    f_match.close()
    print(f"\nDone! Output written to {match_path} and {cand_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--countries", nargs="+", default=None)
    args = parser.parse_args()

    run_test_inference(target_countries=args.countries)