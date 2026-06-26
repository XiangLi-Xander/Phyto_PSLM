"""UMAP visualization of PhytoRNP pooled vectors, colored by RNA dependency."""
import os, sys, warnings, time
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import umap
from tqdm import tqdm

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from src.model import ResidueLLPSClassifier
from src.utils import ESM2Extractor, get_iupred_scores, PROJECT_ROOT

DEVICE = torch.device("cuda:0")
BATCH_SIZE = 32
ESM2_DIR = os.path.join(PROJECT_ROOT, "esm2")
LLPS_MODEL_PATH = os.path.join(PROJECT_ROOT, "outputs", "models", "best_model.pth")
OUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

# Load data
print("[1] Loading data...")
pred = pd.read_excel(os.path.join(PROJECT_ROOT, "outputs", "predictions", "rna_dependent_predictions_all3.xlsx"))
llps = pred[pred['LLPS_Score'] >= 0.5].copy()
llps['label'] = (llps['rna_dependent'] == 'Yes').astype(int)
llps['species_code'] = llps['species'].map({'Zea mays': 0, 'Oryza sativa': 1, 'Arabidopsis thaliana': 2})
print(f"  Total LLPS: {len(llps)} (dep={llps['label'].sum()}, indep={(llps['label']==0).sum()})")

seqs = llps['sequence'].tolist()
labels = llps['label'].values
species = llps['species_code'].values
species_names = {0: 'Zea mays', 1: 'Oryza sativa', 2: 'Arabidopsis thaliana'}

# Extract pooled vectors
print("\n[2] Extracting 512-dim pooled vectors...")
extractor = ESM2Extractor(ESM2_DIR, DEVICE)
model = ResidueLLPSClassifier(d_model=1280, hidden=512).to(DEVICE).eval()
model.load_state_dict(torch.load(LLPS_MODEL_PATH, map_location=DEVICE))

pooled_features = []
hook = model.query_pool.register_forward_hook(
    lambda m, i, o: pooled_features.append(o.detach().cpu().numpy())
)

t0 = time.time()
for i in tqdm(range(0, len(seqs), BATCH_SIZE), desc="Extracting"):
    batch = seqs[i:i+BATCH_SIZE]
    iup = [torch.tensor(get_iupred_scores(s), dtype=torch.float32) for s in batch]
    iup_pad = pad_sequence(iup, batch_first=True, padding_value=0.0).to(DEVICE)
    esm, lens = extractor.extract_residue_batch_tensors(batch, pad_to_length=iup_pad.size(1))
    esm, lens = esm.to(DEVICE), lens.to(DEVICE)
    B, L = esm.shape[:2]
    if iup_pad.size(1) < L:
        iup_pad = torch.cat([iup_pad, torch.zeros(B, L - iup_pad.size(1), device=DEVICE)], dim=1)
    elif iup_pad.size(1) > L:
        iup_pad = iup_pad[:, :L]
    with torch.no_grad():
        _ = model(esm, iup_pad, lens)
hook.remove()
pooled = np.concatenate(pooled_features, axis=0).astype(np.float64)
print(f"  Done: {pooled.shape} ({time.time()-t0:.0f}s)")

# Standardize
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
pooled_scaled = scaler.fit_transform(pooled)

# UMAP
print("\n[3] UMAP dimensionality reduction...")
umap_model = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.3, random_state=42, verbose=True)
umap_2d = umap_model.fit_transform(pooled_scaled)
print(f"  UMAP shape: {umap_2d.shape}")

# Plot: by RNA dependency (2 colors)
print("\n[4] Plotting...")
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# Plot A: colored by RNA dependency
ax = axes[0]
colors_dep = {0: '#3498DB', 1: '#E74C3C'}
for lbl in [0, 1]:
    mask = labels == lbl
    name = 'RNA-independent' if lbl == 0 else 'RNA-dependent'
    ax.scatter(umap_2d[mask, 0], umap_2d[mask, 1], c=colors_dep[lbl], s=3, alpha=0.5,
               label=f'{name} (n={mask.sum()})', edgecolors='none', rasterized=True)
ax.set_title('PhytoRNP Pooled Vectors\nColored by RNA Dependency', fontsize=13, fontweight='bold')
ax.set_xlabel('UMAP 1')
ax.set_ylabel('UMAP 2')
ax.legend(fontsize=9, markerscale=3)
ax.spines[['top', 'right']].set_visible(False)

# Plot B: colored by species
ax = axes[1]
colors_sp = {0: '#E67E22', 1: '#2ECC71', 2: '#9B59B6'}
for sp_code, sp_name in species_names.items():
    mask = species == sp_code
    ax.scatter(umap_2d[mask, 0], umap_2d[mask, 1], c=colors_sp[sp_code], s=3, alpha=0.4,
               label=f'{sp_name} (n={mask.sum()})', edgecolors='none', rasterized=True)
ax.set_title('PhytoRNP Pooled Vectors\nColored by Species', fontsize=13, fontweight='bold')
ax.set_xlabel('UMAP 1')
ax.set_ylabel('UMAP 2')
ax.legend(fontsize=9, markerscale=3)
ax.spines[['top', 'right']].set_visible(False)

plt.suptitle('RNA Dependency in PhytoRNP Feature Space', fontsize=15, fontweight='bold')
plt.tight_layout()
out_path = os.path.join(OUT_DIR, "rna_dep_pooled_umap.png")
plt.savefig(out_path, dpi=200, bbox_inches='tight')
plt.close()
print(f"  Saved: {out_path}")

# Statistics
print(f"\n{'='*60}")
print("Summary")
print(f"{'='*60}")
for lbl, name in [(0, 'RNA-indep'), (1, 'RNA-dep')]:
    mask = labels == lbl
    print(f"  {name}: {mask.sum()} proteins, center=({umap_2d[mask,0].mean():.2f}, {umap_2d[mask,1].mean():.2f})")

# Per-species RNA dep vs indep center
print(f"\n  Per-species centers:")
for sp_code, sp_name in species_names.items():
    for lbl, name in [(0, 'indep'), (1, 'dep')]:
        mask = (species == sp_code) & (labels == lbl)
        if mask.sum() > 0:
            print(f"    {sp_name} {name}: n={mask.sum()} center=({umap_2d[mask,0].mean():.2f}, {umap_2d[mask,1].mean():.2f})")
