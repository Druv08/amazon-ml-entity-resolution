import os
import re
import gc
import joblib
import torch
import numpy as np
import polars as pl
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
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

def fill_feature_row(out_arr, row_idx, s1_meta, op_meta, dense_sc, rank_pos, margin_delta):
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

    out_arr[row_idx, 0] = f_dense
    out_arr[row_idx, 1] = f_rank
    out_arr[row_idx, 2] = f_margin
    out_arr[row_idx, 3] = f_name_sort
    out_arr[row_idx, 4] = f_name_set
    out_arr[row_idx, 5] = f_name_ratio
    out_arr[row_idx, 6] = f_name_partial
    out_arr[row_idx, 7] = f_name_jw
    out_arr[row_idx, 8] = f_exact_sub
    out_arr[row_idx, 9] = f_phonetic
    out_arr[row_idx, 10] = f_addr_sort
    out_arr[row_idx, 11] = f_addr_set
    out_arr[row_idx, 12] = f_addr_jw
    out_arr[row_idx, 13] = f_num_match
    out_arr[row_idx, 14] = f_num_diff
    out_arr[row_idx, 15] = num_jaccard
    out_arr[row_idx, 16] = f_state_conflict
    out_arr[row_idx, 17] = f_state_match
    out_arr[row_idx, 18] = len_diff
    out_arr[row_idx, 19] = float(tok_diff)
    out_arr[row_idx, 20] = f_composite_lex_dense
    out_arr[row_idx, 21] = f_composite_addr_dense

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

        # Inverted GPU Top-K Retrieval
        topk_scores_all = np.full((n_s1, 15), -1.0, dtype=np.float32)
        topk_indices_all = np.full((n_s1, 15), -1, dtype=np.int64)

        s1_block_size = min(n_s1, 450_000)
        ops_stream_batch = 32_768

        for s1_start in range(0, n_s1, s1_block_size):
            s1_end = min(s1_start + s1_block_size, n_s1)
            cur_s1_len = s1_end - s1_start
            print(f"\n[GPU Retrieval] Processing S1 slice [{s1_start:,} : {s1_end:,}] ({cur_s1_len:,} entities)...")

            s1_tensor = torch.from_numpy(np.array(s1_mmap[s1_start:s1_end])).to(device=device, dtype=torch.float16)

            best_scores_gpu = torch.full((cur_s1_len, 15), -1.0, dtype=torch.float16, device=device)
            best_indices_gpu = torch.full((cur_s1_len, 15), -1, dtype=torch.int64, device=device)

            pbar = tqdm(total=n_ops, desc=f"Ops Pass ({country})", leave=False)
            for ops_start in range(0, n_ops, ops_stream_batch):
                ops_end = min(ops_start + ops_stream_batch, n_ops)
                ops_chunk = torch.from_numpy(np.array(ops_mmap[ops_start:ops_end])).to(device=device, dtype=torch.float16)

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

        print(f"\n[GBDT Reranking] Multi-threaded vectorized evaluation for {country}...")

        eval_batch_s1 = 25_000
        n_workers = min(os.cpu_count() or 4, 8)

        for i in tqdm(range(0, n_s1, eval_batch_s1), desc=f"GBDT Eval {country}"):
            c_end = min(i + eval_batch_s1, n_s1)
            c_len = c_end - i

            candidate_items = []
            s1_cand_counts = [0] * c_len

            for local_idx in range(c_len):
                global_idx = i + local_idx
                row_vals = topk_scores_all[global_idx]
                row_inds = topk_indices_all[global_idx]

                top1_sc = row_vals[0]
                top2_sc = row_vals[1] if len(row_vals) > 1 else 0.0
                margin_delta = top1_sc - top2_sc

                for rank_pos, (score, c_idx) in enumerate(zip(row_vals, row_inds)):
                    if c_idx != -1 and score >= 0.62:
                        candidate_items.append((local_idx, rank_pos, int(c_idx), score, margin_delta))
                        s1_cand_counts[local_idx] += 1

            n_candidates = len(candidate_items)
            X_chunk = np.zeros((n_candidates, 22), dtype=np.float32)

            if n_candidates > 0:
                def worker_fill(range_tuple):
                    start_idx, end_idx = range_tuple
                    for c_i in range(start_idx, end_idx):
                        loc_i, r_pos, op_i, sc, m_delta = candidate_items[c_i]
                        g_idx = i + loc_i
                        fill_feature_row(X_chunk, c_i, s1_meta[g_idx], ops_meta[op_i], sc, r_pos, m_delta)

                chunk_step = (n_candidates + n_workers - 1) // n_workers
                ranges = [(k, min(k + chunk_step, n_candidates)) for k in range(0, n_candidates, chunk_step)]

                with ThreadPoolExecutor(max_workers=n_workers) as executor:
                    list(executor.map(worker_fill, ranges))

                p_sum = np.zeros(n_candidates, dtype=np.float32)
                for m in models:
                    p_sum += m.predict_proba(X_chunk)[:, 1]
                probs = p_sum / len(models)
            else:
                probs = np.array([], dtype=np.float32)

            cand_ptr = 0
            for local_idx in range(c_len):
                global_idx = i + local_idx
                sid = s1_ids[global_idx]
                row_inds = topk_indices_all[global_idx]
                row_vals = topk_scores_all[global_idx]

                k_count = s1_cand_counts[local_idx]
                matched_ops = []

                if k_count > 0:
                    ent_probs = probs[cand_ptr : cand_ptr + k_count]
                    ent_items = candidate_items[cand_ptr : cand_ptr + k_count]
                    cand_ptr += k_count

                    max_prob = np.max(ent_probs)
                    if max_prob >= ANCHOR_THRESH:
                        for (_, _, op_i, _, _), p in zip(ent_items, ent_probs):
                            if p >= SISTER_THRESH:
                                matched_ops.append(ops_ids[op_i])

                cands_55 = [ops_ids[int(c)] for score, c in zip(row_vals, row_inds) if c != -1 and score >= 0.55]
                
                f_cand.write(f"{sid}\t{','.join(cands_55)}\n")
                f_match.write(f"{sid}\t{','.join(matched_ops)}\n")

            f_cand.flush()
            f_match.flush()

        del ops_mmap, s1_mmap, ops_ids, s1_ids, topk_scores_all, topk_indices_all
        gc.collect()

    f_cand.close()
    f_match.close()
    print(f"\n[DONE] Pipeline complete! Output written to {match_path} and {cand_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--countries", nargs="+", default=None)
    args = parser.parse_args()

    run_test_inference(target_countries=args.countries)