"""Stand-in blocking so P3 can train on real data. P2's blocking replaces it.

Weighted token overlap: name + address tokens, IDF-weighted, top-K per S1,
searched only within the same country label (open set, works for France).
Output: DataFrame s1_id, cand_id, block_score.
"""
import argparse
import time

import numpy as np
import pandas as pd

MAX_DF = 3000  # tokens in more records than this ("ltd", "road", "mumbai") are skipped
K = 30
COLS = ["entity_id", "business_name", "business_address", "country"]


def _clean(s):
    return (s.fillna("").str.normalize("NFKD").str.replace(r"[̀-ͯ]", "", regex=True)
            .str.lower().str.replace(r"[^\w]+", " ", regex=True))


def tokens(df, chunk=500_000):
    """-> (row int32, tok uint64), unique per row. Name/address tokens kept apart."""
    parts = []
    for lo in range(0, len(df), chunk):
        d = df.iloc[lo:lo + chunk]
        name = _clean(d["business_name"]).str.replace(r"\b(com|www)\b", " ", regex=True).str.split()
        addr = _clean(d["business_address"]).str.replace(r"\b0+(\d)", r"\1", regex=True).str.split()
        feats = {
            "n:": name,
            "b:": name.map(lambda w: [a + " " + b for a, b in zip(w, w[1:])]),  # word pairs: "big foods"
            "c:": name.map(lambda w: ["".join(w)]),  # joined name: matches "bigfoods.com"
            "a:": addr,
        }
        for pre, s in feats.items():
            t = s.explode().dropna()
            t = t[t.str.len() > 1]
            parts.append(pd.DataFrame({"row": t.index.to_numpy(np.int32),
                                       "tok": pd.util.hash_array((pre + t).to_numpy(object))}))
    return pd.concat(parts).drop_duplicates()


def iter_block(queries, universe, k=K, chunk=5000):
    """Yields (pairs, records) per chunk of queries; records = the S1 + candidate rows those pairs need."""
    q, u = queries.reset_index(drop=True), universe.reset_index(drop=True)
    ut = tokens(u).sort_values("tok")
    utok, urow = ut["tok"].to_numpy(), ut["row"].to_numpy()
    uniq, start, df = np.unique(utok, return_index=True, return_counts=True)
    idf = np.log(len(u) / df)
    for lo in range(0, len(q), chunk):
        qt = tokens(q.iloc[lo:lo + chunk])
        pos = np.clip(np.searchsorted(uniq, qt["tok"].to_numpy()), 0, len(uniq) - 1)
        ok = (uniq[pos] == qt["tok"].to_numpy()) & (df[pos] <= MAX_DF)
        pos, qrow = pos[ok], qt["row"].to_numpy()[ok]
        n = df[pos]
        # expand each matched query token into all universe rows holding it
        offs = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
        hits = pd.DataFrame({"q": np.repeat(qrow, n), "u": urow[np.repeat(start[pos], n) + offs],
                             "w": np.repeat(idf[pos], n)})
        s = hits.groupby(["q", "u"], sort=False)["w"].sum().reset_index()
        s = s.sort_values(["q", "w"], ascending=[True, False])
        s = s[s.groupby("q").cumcount() < k]
        pairs = pd.DataFrame({"s1_id": q["entity_id"].to_numpy()[s["q"]],
                              "cand_id": u["entity_id"].to_numpy()[s["u"]],
                              "block_score": s["w"].to_numpy(np.float32)})
        yield pairs, pd.concat([q.iloc[lo:lo + chunk], u.iloc[np.unique(s["u"].to_numpy())]], ignore_index=True)


def iter_by_country(s1, others, k=K, chunk=5000):
    for c, g in s1.groupby("country"):  # open set of labels: France gets its own universe
        uni = others[others["country"] == c]
        if len(uni):
            yield from iter_block(g, uni, k, chunk)


def block_by_country(s1, others, k=K):
    return pd.concat([p for p, _ in iter_by_country(s1, others, k)], ignore_index=True)


def read(path):
    return pd.read_csv(path, sep="\t", usecols=COLS, dtype=str)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100_000, help="sampled S1 entities")
    ap.add_argument("--cities", help="comma list: take ALL S1 in these cities instead of a random sample "
                                     "(keeps competing S1 entities together, needed to evaluate exclusivity)")
    ap.add_argument("--out", default="data/interim/train_pairs.parquet")
    a = ap.parse_args()
    t0 = time.time()
    s1 = read("data/raw/train/train_source1.tsv")
    if a.cities:
        s1 = s1[s1.business_address.str.lower().str.contains(rf"\b(?:{a.cities.replace(',', '|')})\b", na=False)]
    else:
        s1 = s1.sample(a.n, random_state=0)
    s1[["entity_id"]].to_parquet(a.out.replace(".parquet", "_s1.parquet"))  # full S1 list, incl. zero-candidate ones
    others = pd.concat([read("data/raw/train/train_source2.tsv"), read("data/raw/train/train_source3.tsv")],
                       ignore_index=True)
    print(f"loaded {time.time() - t0:.0f}s")
    pairs = block_by_country(s1, others)
    print(f"blocked {time.time() - t0:.0f}s, {len(pairs)} pairs")
    import os
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    pairs.to_parquet(a.out)

    gt = pd.read_csv("data/raw/train/train_ground_truth.tsv", sep="\t", dtype=str, keep_default_na=False)
    gt = gt[gt.source1_entity_id.isin(s1.entity_id)]
    true = gt.assign(cand_id=gt.matched_entity_ids.str.split(",")).explode("cand_id")
    true = true[true.cand_id != ""][["source1_entity_id", "cand_id"]].rename(columns={"source1_entity_id": "s1_id"})
    hit = true.merge(pairs, on=["s1_id", "cand_id"]).shape[0]
    print(f"pair recall {hit / len(true):.4f} | cands/S1 {len(pairs) / len(s1):.1f}")
