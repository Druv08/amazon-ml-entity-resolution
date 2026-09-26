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
from sklearn.model_selection import GroupKFold
import lightgbm as lgb

try:
    from indic_transliteration import sanscript
    from indic_transliteration.sanscript import transliterate
    HAS_INDIC = True
except ImportError:
    HAS_INDIC = False

DATA_DIR = "dataset/train"
INDIC_REGEX = re.compile(r'[\u0900-\u0D7F]')

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
    for t in (n_core if n_core else n_tokens)[:4]:
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

def compute_features_22(s1_meta, op_meta, dense_sc, rank_pos, margin_delta):
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

def compute_official_macro_f05(ground_truth_dict, pred_dict):
    f05_scores = []
    for s1_id, y_true in ground_truth_dict.items():
        y_pred = pred_dict.get(s1_id, set())
        if len(y_true) == 0:
            f05_scores.append(1.0 if len(y_pred) == 0 else 0.0)
            continue

        tp = len(y_true & y_pred)
        fp = len(y_pred - y_true)
        fn = len(y_true - y_pred)

        if tp == 0:
            f05_scores.append(0.0)
            continue

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f05_scores.append((1.25 * precision * recall) / (0.25 * precision + recall))

    return float(np.mean(f05_scores))

@torch.no_grad()
def get_dense(texts, model, tokenizer, device, batch_size=256):
    res = []
    for i in range(0, len(texts), batch_size):
        b = texts[i:i+batch_size]
        inp = tokenizer(b, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
        out = model(**inp)
        rep = torch.nn.functional.normalize(out.last_hidden_state[:, 0, :], p=2, dim=1).half()
        res.append(rep.cpu().numpy())
    return np.vstack(res)

def run_rigorous_cv():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0)})")

    print("\n[Loading Datasets: 8,000 Entities]")
    s1_df = pl.read_csv(os.path.join(DATA_DIR, "train_source1.tsv"), separator="\t").head(8000)
    gt_df = pl.read_csv(os.path.join(DATA_DIR, "train_ground_truth.tsv"), separator="\t")
    s2_df = pl.read_csv(os.path.join(DATA_DIR, "train_source2.tsv"), separator="\t")
    s3_df = pl.read_csv(os.path.join(DATA_DIR, "train_source3.tsv"), separator="\t")
    s_ops_all = pl.concat([s2_df, s3_df])
    del s2_df, s3_df
    gc.collect()

    s1_ids_all = s1_df["entity_id"].to_list()
    gt_slice = gt_df.filter(pl.col("source1_entity_id").is_in(set(s1_ids_all))).to_dicts()
    gt_map_all = {
        r["source1_entity_id"]: set(r["matched_entity_ids"].split(",")) if r["matched_entity_ids"] else set()
        for r in gt_slice
    }

    all_true_ops = set().union(*gt_map_all.values())
    s_ops = pl.concat([
        s_ops_all.filter(pl.col("entity_id").is_in(all_true_ops)),
        s_ops_all.filter(~pl.col("entity_id").is_in(all_true_ops)).head(50000)
    ])
    del s_ops_all
    gc.collect()

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

    s1_texts, s1_meta = process_df(s1_df)
    ops_texts, ops_meta = process_df(s_ops)

    ops_emb = get_dense(ops_texts, mod, tok, device)
    s1_emb = get_dense(s1_texts, mod, tok, device)

    ops_t = torch.from_numpy(ops_emb).to(device)
    ops_ids = s_ops["entity_id"].to_list()

    print("\n[Extracting Candidate Pairs]")
    all_pairs_by_s1 = {}
    
    for i in tqdm(range(0, len(s1_ids_all), 500), desc="Candidate Extraction"):
        chunk = torch.from_numpy(s1_emb[i:i+500]).to(device)
        sims = torch.matmul(chunk, ops_t.T)
        vals, inds = torch.topk(sims, k=15, dim=1)
        vals, inds = vals.cpu().numpy(), inds.cpu().numpy()

        for l_idx in range(len(vals)):
            g_idx = i + l_idx
            sid = s1_ids_all[g_idx]
            top1_sc = vals[l_idx][0]
            top2_sc = vals[l_idx][1] if len(vals[l_idx]) > 1 else 0.0
            margin_delta = top1_sc - top2_sc

            cands_data = []
            for rank_pos, (sc, c_idx) in enumerate(zip(vals[l_idx], inds[l_idx])):
                if sc < 0.55:
                    continue
                c_idx_int = int(c_idx)
                op_id = ops_ids[c_idx_int]
                feats = compute_features_22(s1_meta[g_idx], ops_meta[c_idx_int], sc, rank_pos, margin_delta)
                label = 1 if op_id in gt_map_all.get(sid, set()) else 0
                cands_data.append((feats, label, op_id, c_idx_int))
            all_pairs_by_s1[sid] = cands_data

    del ops_t, s1_emb, ops_emb
    torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    print("RUNNING 5-FOLD STRICT GroupKFold (BATCH VECTORIZED)")
    print("=" * 60)

    gkf = GroupKFold(n_splits=5)
    s1_array = np.array(s1_ids_all)
    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(s1_array, groups=s1_array)):
        train_s1 = set(s1_array[train_idx])
        val_s1 = list(s1_array[val_idx])

        X_tr, y_tr = [], []
        for sid in train_s1:
            for feats, label, _, _ in all_pairs_by_s1[sid]:
                X_tr.append(feats)
                y_tr.append(label)

        clf = lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.04,
            num_leaves=63,
            max_depth=7,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42 + fold,
            n_jobs=-1
        )
        clf.fit(np.array(X_tr), np.array(y_tr))
        joblib.dump(clf, f"lgb_fold_{fold}.pkl")

        # Vectorized batch prediction across all validation pairs
        val_meta_list = []
        X_val = []
        for sid in val_s1:
            for feats, _, op_id, c_idx_int in all_pairs_by_s1[sid]:
                X_val.append(feats)
                val_meta_list.append((sid, op_id, c_idx_int))

        if len(X_val) > 0:
            val_probs = clf.predict_proba(np.array(X_val))[:, 1]
        else:
            val_probs = np.array([])

        val_gt = {sid: gt_map_all[sid] for sid in val_s1}
        best_f05 = 0.0
        best_t = 0.85

        for thresh in np.arange(0.80, 0.94, 0.02):
            val_preds = {sid: set() for sid in val_s1}
            best_ops = {}

            for (sid, op_id, c_idx_int), prob in zip(val_meta_list, val_probs):
                if prob >= thresh:
                    if c_idx_int not in best_ops or prob > best_ops[c_idx_int][1]:
                        best_ops[c_idx_int] = (sid, prob, op_id)

            for c_idx_int, (sid, _, op_id) in best_ops.items():
                val_preds[sid].add(op_id)

            score = compute_official_macro_f05(val_gt, val_preds)
            if score > best_f05:
                best_f05 = score
                best_t = thresh

        fold_scores.append(best_f05)
        print(f"  Fold {fold+1} Macro F0.5: {best_f05:.4f} (Optimal Threshold: {best_t:.2f})")

    mean_cv = np.mean(fold_scores)
    std_cv = np.std(fold_scores)
    print("-" * 60)
    print(f"STRICT UNLEAKED CV MACRO F0.5: {mean_cv:.4f} (+/- {std_cv:.4f})")
    print("=" * 60)

if __name__ == "__main__":
    run_rigorous_cv()