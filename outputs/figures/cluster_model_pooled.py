"""
Clustering using the model's internal pooled representation.

Uses the penultimate-layer output (512-dim pooled vector after multi-query
cross-attention pooling) from the trained ResidueLLPSClassifier as features
for UMAP visualisation and K-Means clustering.

Pipeline
--------
1. Load trained model + ESM2 extractor + IUPred3.
2. Batch forward pass through the model up to the pooling layer;
   capture the pooled vector via a forward hook.
3. UMAP 2D projection.
4. K-Means clustering of the background set (k = 8).
5. LLPS density contour overlay.
6. Publication-quality figure + CSV tables with UMAP coordinates.

Expected runtime: ~2.5–3 hours (ESM2 residue-level extraction is the bottleneck).
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
from sklearn.preprocessing import StandardScaler
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from src.model import ResidueLLPSClassifier
from src.utils import PROJECT_ROOT, ESM2Extractor, clean_sequence_df, get_iupred_scores

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
POS_CSV = os.path.join(PROJECT_ROOT, "all_species_llps_sequences.csv")
NEG_CSV = os.path.join(PROJECT_ROOT, "all_species_no_llps_merged.csv")
ESM2_DIR = os.path.join(PROJECT_ROOT, "esm2")
MODEL_PATH = os.path.join(PROJECT_ROOT, "outputs", "models", "best_model.pth")
OUTPUT_FIG = os.path.join(
    PROJECT_ROOT, "outputs", "figures", "model_pooled_clusters.png"
)
OUTPUT_CSV_DIR = os.path.join(PROJECT_ROOT, "outputs", "predictions")
os.makedirs(os.path.dirname(OUTPUT_FIG), exist_ok=True)
os.makedirs(OUTPUT_CSV_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
N_CLUSTERS = 8
N_NEG_SAMPLE = None

print(f"Device: {DEVICE}  |  Batch: {BATCH_SIZE}  |  K-Means k = {N_CLUSTERS}")
if N_NEG_SAMPLE:
    print(f"Subsampling negative set to {N_NEG_SAMPLE} sequences")

# ---------------------------------------------------------------------------
# 1. Load and clean sequence data
# ---------------------------------------------------------------------------
print("\n[1] Loading data...")
pos_orig = pd.read_csv(POS_CSV)
neg_orig = pd.read_csv(NEG_CSV)
pos_df = clean_sequence_df(pos_orig.copy(), seq_col="sequence", label=1)
neg_df = clean_sequence_df(neg_orig.copy(), seq_col="sequence", label=0)

if N_NEG_SAMPLE and len(neg_df) > N_NEG_SAMPLE:
    neg_df = neg_df.sample(n=N_NEG_SAMPLE, random_state=42)

print(f"  Positive (LLPS): {len(pos_df)} | Negative (background): {len(neg_df)}")

# ---------------------------------------------------------------------------
# 2. Load model and ESM2
# ---------------------------------------------------------------------------
print("\n[2] Loading trained model and ESM2 extractor...")
extractor = ESM2Extractor(ESM2_DIR, DEVICE)
model = ResidueLLPSClassifier(d_model=1280, hidden=512).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ---------------------------------------------------------------------------
# 3. Register forward hook to capture pooled vector
# ---------------------------------------------------------------------------
pooled_vectors = []


def capture_pooled(_module, _input, output):
    pooled_vectors.append(output.detach().cpu().numpy())


hook_handle = model.query_pool.register_forward_hook(capture_pooled)


# ---------------------------------------------------------------------------
# 4. Batch inference: extract pooled vectors for all sequences
# ---------------------------------------------------------------------------
def extract_pooled(sequences, desc=""):
    pooled_vectors.clear()
    n = len(sequences)
    for i in tqdm(range(0, n, BATCH_SIZE), desc=f"Extracting {desc}"):
        batch_seqs = sequences[i : i + BATCH_SIZE]

        batch_iupred = [get_iupred_scores(seq) for seq in batch_seqs]
        iupred_tensors = [
            torch.tensor(s, dtype=torch.float32) for s in batch_iupred
        ]
        lengths = torch.tensor([t.size(0) for t in iupred_tensors], dtype=torch.long)
        iupred_padded = pad_sequence(
            iupred_tensors, batch_first=True, padding_value=0.0
        )

        esm_embeds, _ = extractor.extract_residue_batch_tensors(
            batch_seqs, pad_to_length=iupred_padded.size(1)
        )
        esm_embeds = esm_embeds.to(DEVICE)
        iupred_padded = iupred_padded.to(DEVICE)
        lengths = lengths.to(DEVICE)

        with torch.inference_mode():
            _ = model(esm_embeds, iupred_padded, lengths)

        del esm_embeds, iupred_padded, lengths
        torch.cuda.empty_cache()

    return np.concatenate(pooled_vectors, axis=0)


print("\n[3] Extracting pooled vectors from trained model...")
neg_pooled = extract_pooled(neg_df["sequence"].tolist(), "Neg")
pos_pooled = extract_pooled(pos_df["sequence"].tolist(), "Pos")
print(f"  Neg pool: {neg_pooled.shape}  Pos pool: {pos_pooled.shape}")

hook_handle.remove()

# ---------------------------------------------------------------------------
# 5. UMAP reduction
# ---------------------------------------------------------------------------
print("\n[4] UMAP dimensionality reduction...")
scaler = StandardScaler()
neg_scaled = scaler.fit_transform(neg_pooled)
pos_scaled = scaler.transform(pos_pooled)

umap_model = umap.UMAP(
    n_components=2, n_neighbors=30, min_dist=0.2,
    random_state=42, verbose=False,
)
neg_2d = umap_model.fit_transform(neg_scaled)
pos_2d = umap_model.transform(pos_scaled)
del neg_scaled, pos_scaled, neg_pooled, pos_pooled
gc.collect()
print(f"  Neg: {neg_2d.shape}  Pos: {pos_2d.shape}")

# ---------------------------------------------------------------------------
# 6. K-Means clustering
# ---------------------------------------------------------------------------
print(f"\n[5] K-Means clustering (k = {N_CLUSTERS})...")
km = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
neg_labels = km.fit_predict(neg_2d)
cluster_sizes = np.bincount(neg_labels)
print(f"  Cluster sizes: {cluster_sizes.tolist()}")

# ---------------------------------------------------------------------------
# 7. LLPS density (KDE)
# ---------------------------------------------------------------------------
print("\n[6] Computing LLPS density (KDE)...")
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
print("\n[7] Generating figure...")
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
ax.contour(
    xx, yy, zz, levels=4,
    colors="#E74C3C", linewidths=0.8, alpha=0.5, zorder=3,
)
ax.scatter(
    pos_2d[:, 0], pos_2d[:, 1],
    c="#E74C3C", marker=".", s=4, alpha=0.3,
    label=f"LLPS (n={len(pos_2d)})", zorder=4, linewidths=0,
)

ax.set_xlabel("UMAP 1")
ax.set_ylabel("UMAP 2")
ax.set_title(
    f"Model Internal Representation (pooled 512-dim)\n"
    f"{len(neg_2d)} non-LLPS (K-Means k={N_CLUSTERS}) | "
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
print("\n[8] Saving CSV tables...")
neg_out = neg_df.copy()
neg_out["UMAP1"] = neg_2d[:, 0]
neg_out["UMAP2"] = neg_2d[:, 1]
neg_out["BG_Cluster"] = neg_labels

pos_out = pos_df.copy()
pos_out["UMAP1"] = pos_2d[:, 0]
pos_out["UMAP2"] = pos_2d[:, 1]
pos_out["BG_Cluster"] = -1

neg_out.to_csv(
    os.path.join(OUTPUT_CSV_DIR, "neg_model_pooled_umap.csv"), index=False
)
pos_out.to_csv(
    os.path.join(OUTPUT_CSV_DIR, "pos_model_pooled_umap.csv"), index=False
)
print(f"  neg_model_pooled_umap.csv ({len(neg_out)} rows)")
print(f"  pos_model_pooled_umap.csv ({len(pos_out)} rows)")
print("Done!")
