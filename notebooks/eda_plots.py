"""Dataset + model overview charts -> output/eda.png

python -m notebooks.eda_plots
(panels 4-6 and 9 use the NEWEST data/interim/*_oof.parquet, i.e. the latest src.matching.train run)
"""
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"  # categorical slots 1-4, fixed order
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
COUNTRY = {"US": BLUE, "India": ORANGE, "France": AQUA}  # color follows the country, not its rank
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                     "axes.grid": True, "grid.color": GRID, "axes.axisbelow": True, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold"})
R = "data/raw"


def profile(path):
    c, miss, nonlat, n = {}, 0, 0, 0
    for ch in pd.read_csv(path, sep="\t", dtype=str, usecols=["business_name", "business_address", "country"],
                          chunksize=1_000_000):
        for k, v in ch.country.value_counts().items():
            c[k] = c.get(k, 0) + v
        miss += ch.business_address.isna().sum() + (ch.business_address.str.upper() == "NULL").sum()
        nonlat += ch.business_name.fillna("").str.contains(r"[ऀ-෿]").sum()  # Indic scripts
        n += len(ch)
    return c, miss / n, nonlat / n, n


files = [(f"{split} S{i}", f"{R}/{split}/{split}_source{i}.tsv") for split in ("train", "test") for i in (1, 2, 3)]
prof = {name: profile(p) for name, p in files}
print("profiled sources")

fig, ax = plt.subplots(3, 3, figsize=(16, 14))
fig.suptitle("Business entity resolution: data & matcher overview", fontsize=13, fontweight="bold", x=0.01, ha="left")

# 1. records per source, stacked by country (France only in test)
a = ax[0, 0]
names = list(prof)
bottom = np.zeros(len(names))
for ctry, col in COUNTRY.items():
    v = np.array([prof[n][0].get(ctry, 0) for n in names]) / 1e6
    a.bar(names, v, bottom=bottom, color=col, label=ctry, width=0.6, edgecolor=SURFACE, linewidth=2)
    bottom += v
for i, t in enumerate(bottom):
    a.text(i, t + 0.08, f"{t:.1f}M", ha="center", color=INK2, fontsize=8)
a.set_title("Records per source (millions), by country", loc="left")
a.set_ylim(0, bottom.max() * 1.12)
a.legend(frameon=False, loc="upper left")
a.tick_params(axis="x", rotation=30)

# 2. matches per S1 entity
gt = pd.read_csv(f"{R}/train/train_ground_truth.tsv", sep="\t", dtype=str, keep_default_na=False)
k = (gt.matched_entity_ids.str.count(",") + (gt.matched_entity_ids != "")).value_counts().sort_index()
a = ax[0, 1]
a.bar(k.index, k.values / 1e3, color=BLUE, width=0.7)
a.bar([0], [k.get(0, 0) / 1e3], color=ORANGE, width=0.7, label=f"singletons ({k.get(0, 0) / k.sum():.1%})")
a.set_title("Train: true matches per S1 entity (thousands of S1)", loc="left")
a.set_xlabel("number of S2/S3 matches")
a.set_xticks(k.index)
a.legend(frameon=False, loc="upper right")

# 3. noise rates per source
a = ax[0, 2]
x = np.arange(len(names))
a.bar(x - 0.2, [prof[n][1] * 100 for n in names], 0.38, color=BLUE, label="missing / NULL address")
a.bar(x + 0.2, [prof[n][2] * 100 for n in names], 0.38, color=ORANGE, label="name in Indic script")
a.set_xticks(x, names, rotation=30)
a.set_title("Noise rate per source (% of records)", loc="left")
a.legend(frameon=False, loc="upper left")

oofs = glob.glob("data/interim/*_oof.parquet")
if oofs:
    from src.matching.matcher import exclusive, macro_f05, to_matches
    path = max(oofs, key=os.path.getmtime)
    print(f"model panels from {path}")
    oof = pd.read_parquet(path)
    sample = set(pd.read_parquet(path.replace("_oof.parquet", "_s1.parquet")).entity_id)
    g = gt[gt.source1_entity_id.isin(sample)]
    truth = {s: set(filter(None, m.split(","))) for s, m in zip(g.source1_entity_id, g.matched_entity_ids)}
    n_true = sum(map(len, truth.values()))
    prob = exclusive(oof, oof.prob.to_numpy())  # the final decision rule applies exclusivity
    tag = f"{os.path.basename(path)}, {len(sample):,} S1"

    # 4. blocking recall@K
    a = ax[1, 0]
    rank = oof.groupby("s1_id").block_score.rank(ascending=False, method="first")
    ks = np.arange(1, 31)
    rec = [oof.label[rank <= kk].sum() / n_true for kk in ks]
    a.plot(ks, np.array(rec) * 100, color=BLUE, lw=2)
    a.plot(ks[-1], rec[-1] * 100, "o", color=BLUE, ms=8, mec=SURFACE, mew=2)
    a.annotate(f"{rec[-1]:.1%} @ K=30", (ks[-1], rec[-1] * 100), xytext=(-70, -18), textcoords="offset points",
               color=INK2)
    a.set_title(f"Stand-in blocking: recall of true pairs vs K\n{tag}", loc="left")
    a.set_xlabel("candidates kept per S1 (K)")
    a.set_ylabel("% of true pairs in candidates")

    # 5. model score distribution by label
    a = ax[1, 1]
    bins = np.linspace(0, 1, 41)
    a.hist(oof.prob[oof.label == 0], bins, color=BLUE, label="non-match pairs", histtype="step", lw=2)
    a.hist(oof.prob[oof.label == 1], bins, color=ORANGE, label="true-match pairs", histtype="step", lw=2)
    a.set_yscale("log")
    a.set_title(f"Matcher out-of-fold probability (log count)\n{tag}", loc="left")
    a.set_xlabel("P(match)")
    a.legend(frameon=False)

    # 6. macro-F0.5 vs threshold (S1 with zero candidates count as empty predictions)
    ts = np.round(np.arange(0.2, 0.96, 0.02), 2)
    f = [macro_f05(to_matches(oof, prob, t, truth), truth) for t in ts]
    a = ax[1, 2]
    a.plot(ts, f, color=BLUE, lw=2)
    i = int(np.argmax(f))
    a.plot(ts[i], f[i], "o", color=BLUE, ms=8, mec=SURFACE, mew=2)
    a.annotate(f"best {f[i]:.3f} @ {ts[i]}", (ts[i], f[i]), xytext=(-40, 10), textcoords="offset points", color=INK2)
    a.set_title(f"macro-F0.5 vs threshold (with exclusivity)\n{tag}", loc="left")
    a.set_xlabel("probability threshold")

    # 9. macro-F0.5 per country at the best threshold
    ctry = pd.read_csv(f"{R}/train/train_source1.tsv", sep="\t", dtype=str, usecols=["entity_id", "country"])
    ctry = ctry[ctry.entity_id.isin(sample)].set_index("entity_id").country
    pred = to_matches(oof, prob, ts[i], truth)
    a = ax[2, 2]
    for c, ids in ctry.groupby(ctry).groups.items():
        sc = macro_f05({s: pred[s] for s in ids}, {s: truth[s] for s in ids})
        a.bar(c, sc, color=COUNTRY.get(c, YELLOW), width=0.6)
        a.text(c, sc + 0.01, f"{sc:.3f}\n{len(ids):,} S1", ha="center", color=INK2, fontsize=8)
    a.set_ylim(0, 1.1)
    a.set_title(f"Validation macro-F0.5 per country @ {ts[i]}\n{tag}", loc="left")
else:
    for a in list(ax[1]) + [ax[2, 2]]:
        a.text(0.5, 0.5, "run src.matching.train first", ha="center", color=INK2, transform=a.transAxes)

# 7-8: experiment results copied from docs/p3_matching.md (each point is a full retrain; too slow to redo here)
a = ax[2, 0]
K = [3, 5, 10, 15, 20, 30]
a.plot(K, [0.8063, 0.8633, 0.8994, 0.9161, 0.9267, 0.9370], color=ORANGE, lw=2, marker="o", label="blocking ceiling")
a.plot(K, [0.7806, 0.8307, 0.8607, 0.8745, 0.8836, 0.8921], color=BLUE, lw=2, marker="o", label="model macro-F0.5")
a.annotate("recommended K=30", (30, 0.8921), xytext=(-110, -30), textcoords="offset points", color=INK2,
           arrowprops={"arrowstyle": "-", "color": INK2})
a.set_title("Cost of a smaller K (top-K by block_score, 30k S1)", loc="left")
a.set_xlabel("candidates kept per S1 (K)")
a.set_xticks(K)
a.legend(frameon=False, loc="upper left")

a = ax[2, 1]
changes = [("TF-IDF fit once\n(full sample)", 0.8877, 0.8868), ("cross-entity gap\n(dense-city sample)", 0.8823, 0.8835),
           ("cross-entity gap\n(full sample)", 0.8868, 0.8877)]
for j, (lab, b, af) in enumerate(changes):
    a.plot([b, af], [j, j], color=GRID, lw=3, zorder=1)
    a.scatter([b], [j], color=INK2, s=50, zorder=2, label="before" if j == 0 else None)
    a.scatter([af], [j], color=BLUE, s=50, zorder=2, label="after" if j == 0 else None)
    a.annotate(f"{af - b:+.4f}", (max(b, af), j), xytext=(8, -3), textcoords="offset points", color=INK2)
a.set_yticks(range(len(changes)), [c[0] for c in changes])
a.set_ylim(-0.6, len(changes) - 0.4)
a.invert_yaxis()
a.set_xlim(0.880, 0.890)
a.set_title("Matcher changes: CV macro-F0.5 with exclusivity", loc="left")
a.legend(frameon=False, loc="lower right")

fig.tight_layout()
os.makedirs("output", exist_ok=True)
fig.savefig("output/eda.png", dpi=130)
print("saved output/eda.png")
