"""
Clustering and visualisation of the LLPS protein embedding space.

Pipeline:
1. ESM2 ``<cls>`` embeddings for positive (LLPS) and negative (background)
   sequences.
2. Aggregate IUPred3 features (mean, std, tail statistics).
3. Separate PCA (ESM2: 1280→20) + standard-scaling (IUPred3: 8 dims),
   then concatenation.
4. UMAP reduction to 2D.
5. K-Means clustering of the background set (k = 8).
6. Overlay of the LLPS density contour on the same 2D map.
7. Academic-style publication figure.
8. CSV tables with UMAP coordinates.

Expected runtime: ~90--120 minutes (ESM2 ~80 min, IUPred3 ~15 min, rest ~5 min).
"""

import gc
import os
import sys
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import umap
from matplotlib import rcParams
from scipy.stats import gaussian_kde
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from src.utils import PROJECT_ROOT, clean_sequence_df, get_iupred_scores  # noqa: E402

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
POS_CSV = os.path.join(PROJECT_ROOT, "all_species_llps_sequences.csv")
NEG_CSV = os.path.join(PROJECT_ROOT, "all_species_no_llps_merged.csv")
ESM2_DIR = os.path.join(PROJECT_ROOT, "esm2")
OUTPUT_FIG = os.path.join(PROJECT_ROOT, "outputs", "figures", "embedding_clusters.png")
OUTPUT_CSV_DIR = os.path.join(PROJECT_ROOT, "outputs", "predictions")
os.makedirs(os.path.dirname(OUTPUT_FIG), exist_ok=True)
os.makedirs(OUTPUT_CSV_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
N_CLUSTERS = 8
NEG_SUBSAMPLE = 20000  # use 20k random negatives for UMAP

print(f"Device: {DEVICE}  |  K-Means k = {N_CLUSTERS}")

# ---------------------------------------------------------------------------
# 1. Load and clean
# ---------------------------------------------------------------------------
print("\n[1] Loading data...")
pos_orig = pd.read_csv(POS_CSV)
neg_orig = pd.read_csv(NEG_CSV)
pos_df = clean_sequence_df(pos_orig.copy(), seq_col="sequence", label=1)
neg_df = clean_sequence_df(neg_orig.copy(), seq_col="sequence", label=0)
if len(neg_df) > NEG_SUBSAMPLE:
    neg_df = neg_df.sample(n=NEG_SUBSAMPLE, random_state=42).reset_index(drop=True)
print(f"  Positive (LLPS): {len(pos_df)} (original: {len(pos_orig)})")
print(f"  Negative (background): {len(neg_df)} (original: {len(neg_orig)}, subsampled)")

# ---------------------------------------------------------------------------
# 2. ESM2 CLS embeddings
# ---------------------------------------------------------------------------
print("\n[2] Extracting ESM2 CLS embeddings (fp16, max_len=512)...")
tokenizer = AutoTokenizer.from_pretrained(ESM2_DIR, local_files_only=True)
esm_model = (
    AutoModel.from_pretrained(ESM2_DIR, local_files_only=True, dtype=torch.float16)
    .to(DEVICE)
)
esm_model.eval()


def extract_esm(seqs, desc=""):
    out = []
    n = len(seqs)
    for i in range(0, n, BATCH_SIZE):
        batch = seqs[i : i + BATCH_SIZE]
        inputs = tokenizer(
            batch, truncation=True, max_length=512, padding=True, return_tensors="pt",
        )
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.inference_mode():
            h = esm_model(**inputs).last_hidden_state[:, 0, :].float()
        out.append(h.cpu().numpy())
        if (i // BATCH_SIZE) % 10 == 0:
            print(f"  ESM2 {desc}: {min(i + BATCH_SIZE, n)}/{n}", flush=True)
    return np.concatenate(out, axis=0)


neg_esm = extract_esm(neg_df["sequence"].tolist(), "Neg")
pos_esm = extract_esm(pos_df["sequence"].tolist(), "Pos")
print(f"  Neg: {neg_esm.shape}  Pos: {pos_esm.shape}")

# ---------------------------------------------------------------------------
# 3. IUPred3 aggregate features
# ---------------------------------------------------------------------------
print("\n[3] Computing IUPred3 aggregate features...")


def iupred_feats(seq):
    scores = get_iupred_scores(seq)
    if len(scores) == 0:
        return np.zeros(8, dtype=np.float32)
    return np.array(
        [
            scores.mean(),
            scores.std(),
            (scores > 0.5).mean(),
            (scores > 0.7).mean(),
            scores.max(),
            scores.min(),
            np.percentile(scores, 90),
            np.percentile(scores, 10),
        ],
        dtype=np.float32,
    )


def compute_iupred(seqs, desc=""):
    out = []
    for i, s in enumerate(seqs):
        out.append(iupred_feats(s))
        if (i + 1) % 5000 == 0:
            print(f"  IUPred {desc}: {i + 1}/{len(seqs)}", flush=True)
    return np.stack(out, axis=0)


neg_iup = compute_iupred(neg_df["sequence"].tolist(), "Neg")
pos_iup = compute_iupred(pos_df["sequence"].tolist(), "Pos")
print(f"  Neg: {neg_iup.shape}  Pos: {pos_iup.shape}")

# ---------------------------------------------------------------------------
# 4. Separate PCA + scaling
# ---------------------------------------------------------------------------
print("\n[4] Reducing dimensions and concatenating features...")

esm_scaler = StandardScaler()
neg_esm_scaled = esm_scaler.fit_transform(neg_esm)
pos_esm_scaled = esm_scaler.transform(pos_esm)
del neg_esm, pos_esm; gc.collect()

esm_pca = PCA(n_components=20, random_state=42)
neg_esm_pc = esm_pca.fit_transform(neg_esm_scaled)
pos_esm_pc = esm_pca.transform(pos_esm_scaled)
del neg_esm_scaled, pos_esm_scaled; gc.collect()
print(f"  ESM2 PCA explained variance ratio: {esm_pca.explained_variance_ratio_.sum():.3f}")

iup_scaler = StandardScaler()
neg_iup_scaled = iup_scaler.fit_transform(neg_iup)
pos_iup_scaled = iup_scaler.transform(pos_iup)
del neg_iup, pos_iup; gc.collect()

neg_feat = np.concatenate([neg_esm_pc, neg_iup_scaled], axis=1)
pos_feat = np.concatenate([pos_esm_pc, pos_iup_scaled], axis=1)
del neg_esm_pc, pos_esm_pc, neg_iup_scaled, pos_iup_scaled; gc.collect()
print(f"  Final feature dimension: {neg_feat.shape[1]} (20 ESM2-PC + 8 IUPred3)")

# ---------------------------------------------------------------------------
# 5. UMAP
# ---------------------------------------------------------------------------
print("\n[5] UMAP dimensionality reduction...")
umap_model = umap.UMAP(
    n_components=2, n_neighbors=30, min_dist=0.2,
    random_state=42, verbose=False,
)
neg_2d = umap_model.fit_transform(neg_feat)
pos_2d = umap_model.transform(pos_feat)
del neg_feat, pos_feat; gc.collect()
print(f"  Neg: {neg_2d.shape}  Pos: {pos_2d.shape}")

# ---------------------------------------------------------------------------
# 6. K-Means
# ---------------------------------------------------------------------------
print(f"\n[6] K-Means clustering (k = {N_CLUSTERS})...")
km = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
neg_labels = km.fit_predict(neg_2d)
cluster_sizes = np.bincount(neg_labels)
print(f"  Cluster sizes: {cluster_sizes.tolist()}")

# ---------------------------------------------------------------------------
# 7. LLPS density
# ---------------------------------------------------------------------------
print("\n[7] Computing LLPS density (KDE)...")
kde = gaussian_kde(pos_2d.T)
x_min, x_max = neg_2d[:, 0].min(), neg_2d[:, 0].max()
y_min, y_max = neg_2d[:, 1].min(), neg_2d[:, 1].max()
xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 100),
    np.linspace(y_min, y_max, 100),
)
zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)

# ---------------------------------------------------------------------------
# 8. Publication figure
# ---------------------------------------------------------------------------
print("\n[8] Generating figure...")
rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "axes.edgecolor": "black",
        "axes.linewidth": 1.0,
        "xtick.major.size": 4,
        "ytick.major.size": 4,
    }
)

fig, ax = plt.subplots(figsize=(10, 8))

gray_levels = np.linspace(0.85, 0.35, N_CLUSTERS)
for c in range(N_CLUSTERS):
    mask = neg_labels == c
    g = gray_levels[c]
    ax.scatter(
        neg_2d[mask, 0], neg_2d[mask, 1],
        c=[(g, g, g, 0.3)], s=2, alpha=0.35, linewidths=0,
        label=f"BG-C{c + 1} ({mask.sum()})",
    )

ax.contourf(xx, yy, zz, levels=8, cmap="Reds", alpha=0.12, zorder=2)
ax.contour(xx, yy, zz, levels=4, colors="#E74C3C", linewidths=0.8, alpha=0.5, zorder=3)

ax.scatter(
    pos_2d[:, 0], pos_2d[:, 1],
    c="#E74C3C", marker=".", s=4, alpha=0.3,
    label=f"LLPS (n={len(pos_2d)})", zorder=4, linewidths=0,
)

ax.set_xlabel("UMAP 1")
ax.set_ylabel("UMAP 2")
ax.set_title(
    f"Protein Embedding Space (ESM2 20PC + IUPred3 8 features)\n"
    f"{len(neg_2d)} non-LLPS (K-Means k = {N_CLUSTERS}) | "
    f"{len(pos_2d)} LLPS (red)",
    fontsize=11,
    pad=10,
)
ax.legend(
    fontsize=7, loc="upper right", markerscale=4, frameon=True,
    ncol=2, facecolor="white", edgecolor="gray", framealpha=0.9,
)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(OUTPUT_FIG, dpi=300, bbox_inches="tight")
plt.close()
print(f"  Figure saved to {OUTPUT_FIG}")

# ---------------------------------------------------------------------------
# 9. Output CSVs
# ---------------------------------------------------------------------------
print("\n[9] Saving CSV tables with UMAP coordinates...")

neg_out = neg_df.copy()
neg_out["UMAP1"] = neg_2d[:, 0]
neg_out["UMAP2"] = neg_2d[:, 1]
neg_out["BG_Cluster"] = neg_labels

pos_out = pos_df.copy()
pos_out["UMAP1"] = pos_2d[:, 0]
pos_out["UMAP2"] = pos_2d[:, 1]
pos_out["BG_Cluster"] = -1

neg_merged = neg_orig.merge(neg_out[["sequence", "UMAP1", "UMAP2"]], on="sequence", how="left")
pos_merged = pos_orig.merge(pos_out[["sequence", "UMAP1", "UMAP2"]], on="sequence", how="left")

combined = pd.concat([neg_merged, pos_merged], ignore_index=True)
combined_path = os.path.join(OUTPUT_CSV_DIR, "all_with_umap.csv")
neg_path = os.path.join(OUTPUT_CSV_DIR, "neg_with_umap.csv")
pos_path = os.path.join(OUTPUT_CSV_DIR, "pos_with_umap.csv")

combined.to_csv(combined_path, index=False)
neg_merged.to_csv(neg_path, index=False)
pos_merged.to_csv(pos_path, index=False)

print(f"  Combined ({len(combined)} rows): {combined_path}")
print(f"  Neg only  ({len(neg_merged)} rows): {neg_path}")
print(f"  Pos only  ({len(pos_merged)} rows): {pos_path}")
print("Done!")
