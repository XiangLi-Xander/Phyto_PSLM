"""
Generate UMAP visualizations for RNA dependency models.
- "Before": 512d input features → UMAP → colored by true labels
- "After": model internal representation → UMAP → colored by true labels

XGBoost/LGB/RF: use leaf indices (apply) as internal representation
SVM: use predict_proba (2-class probabilities)
Control PhytoRNP: use 512d pooled vector from inside the model
"""
import os, sys, warnings, time
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import umap
import joblib

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, HERE)
FEAT_CACHE = os.path.join(HERE, "outputs", "benchmarks", "phyto_512d_features.npz")
MODEL_DIR = os.path.join(HERE, "outputs", "models")
OUT_BASE = os.path.join(HERE, "outputs", "rna_umap")

data = np.load(FEAT_CACHE)
X_all = np.concatenate([data["X_train"], data["X_val"], data["X_test"]])
y_all = np.concatenate([data["y_train"], data["y_val"], data["y_test"]]).astype(int)

# Standardize
sc_in = StandardScaler()
X_scaled = sc_in.fit_transform(X_all).astype(np.float32)

# ============================================================================
# Common "before" UMAP (512d features)
# ============================================================================
before_cache = os.path.join(OUT_BASE, "before_umap.npz")
if os.path.exists(before_cache):
    d = np.load(before_cache)
    X_before_umap = d["X_umap"]
    print(f"Loaded cached before UMAP: {X_before_umap.shape}")
else:
    print("Computing 'before' UMAP on 512d features...", flush=True)
    reducer = umap.UMAP(n_components=2, random_state=42, n_jobs=1)
    X_before_umap = reducer.fit_transform(X_scaled)
    np.savez(before_cache, X_umap=X_before_umap)
    print(f"Done: {X_before_umap.shape}")


def plot_umap(X_umap, labels, title, out_path):
    fig, ax = plt.subplots(figsize=(7, 6))
    for label, color, name in [(0, "steelblue", "RNA-independent"),
                                (1, "darkorange", "RNA-dependent")]:
        mask = labels == label
        ax.scatter(X_umap[mask, 0], X_umap[mask, 1], c=color, s=0.5,
                   alpha=0.4, label=f"{name} (n={mask.sum()})", rasterized=True)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(markerscale=8, fontsize=8)
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def compute_umap(name, features):
    cache = os.path.join(OUT_BASE, f"{name}_after_umap.npz")
    if os.path.exists(cache):
        return np.load(cache)["X_umap"]
    print(f"  Computing UMAP on {name} features ({features.shape[1]}d)...", flush=True)
    r = umap.UMAP(n_components=2, random_state=42, n_jobs=1)
    X_umap = r.fit_transform(features)
    np.savez(cache, X_umap=X_umap)
    print(f"  Done: {X_umap.shape}")
    return X_umap


# ============================================================================
# Per-model "after" UMAP
# ============================================================================
for model_name, pkl_name in [
    ("xgboost", "best_rna_xgb_model.pkl"),
    ("lightgbm", "best_rna_lgb_model.pkl"),
    ("rf", "best_rna_rf_model.pkl"),
    ("svm", "best_rna_svm_model.pkl"),
]:
    print(f"\n{'='*50}\n  {model_name}\n{'='*50}")
    out_dir = os.path.join(OUT_BASE, model_name)
    os.makedirs(out_dir, exist_ok=True)

    # "Before" plot
    plot_umap(X_before_umap, y_all,
              f"RNA Dependency — Before ({model_name})\nPhytoRNP 512d Input Features",
              os.path.join(out_dir, f"rna_{model_name}_before_umap.png"))

    # "After" plot — model internal representation
    obj = joblib.load(os.path.join(MODEL_DIR, pkl_name))
    model = obj["model"]; sc = obj["scaler"]
    X_s = sc.transform(X_all).astype(np.float32)

    if model_name in ("xgboost", "rf"):
        # Tree leaf indices as internal representation
        after_feat = model.apply(X_s).astype(np.float32)
    elif model_name == "lightgbm":
        after_feat = model.predict(X_s, pred_leaf=True).astype(np.float32)
    elif model_name == "svm":
        # SVM RBF: use predict_proba as 2D representation
        after_feat = model.predict_proba(X_s).astype(np.float32)

    after_umap = compute_umap(model_name, after_feat)
    plot_umap(after_umap, y_all,
              f"RNA Dependency — After ({model_name})\nInternal Representation ({after_feat.shape[1]}d → UMAP)",
              os.path.join(out_dir, f"rna_{model_name}_after_umap.png"))

    # Save CSV with both before/after coordinates
    df = pd.DataFrame({
        "before_umap_x": X_before_umap[:, 0], "before_umap_y": X_before_umap[:, 1],
        "after_umap_x": after_umap[:, 0], "after_umap_y": after_umap[:, 1],
        "true_label": y_all,
    })
    df.to_csv(os.path.join(out_dir, f"rna_{model_name}_umap.csv"), index=False)

# ============================================================================
# Control PhytoRNP — use 512d pooled vector from inside model
# ============================================================================
print(f"\n{'='*50}\n  control\n{'='*50}")
out_dir = os.path.join(OUT_BASE, "control")
os.makedirs(out_dir, exist_ok=True)

plot_umap(X_before_umap, y_all,
          "RNA Dependency — Before (Control)\nPhytoRNP 512d Input Features",
          os.path.join(out_dir, "rna_control_before_umap.png"))

# Extract 512d pooled from control model
import torch
from src.model import ResidueLLPSClassifier
from src.utils import ESM2Extractor, get_iupred_scores
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

DEV = torch.device("cuda:0")
m = ResidueLLPSClassifier(d_model=1280, hidden=512).to(DEV).eval()
m.load_state_dict(torch.load(os.path.join(MODEL_DIR, "best_rna_control_model.pth"),
                              map_location=DEV))
esm2 = ESM2Extractor(os.path.join(HERE, "esm2"), DEV)

pooled_list = []
def capture(_m, _i, output): pooled_list.append(output.detach().cpu().numpy())
hook = m.query_pool.register_forward_hook(capture)

seqs_train = pd.read_csv(os.path.join(HERE, "data", "rna_dep", "train.csv"))["sequence"].tolist()
seqs_val = pd.read_csv(os.path.join(HERE, "data", "rna_dep", "val.csv"))["sequence"].tolist()
seqs_test = pd.read_csv(os.path.join(HERE, "data", "rna_dep", "test.csv"))["sequence"].tolist()
all_seqs = seqs_train + seqs_val + seqs_test

with torch.no_grad():
    for i in tqdm(range(0, len(all_seqs), 32), desc="Extracting control pooled"):
        batch = all_seqs[i:i+32]
        iup = [torch.tensor(get_iupred_scores(s), dtype=torch.float32) for s in batch]
        iup_pad = pad_sequence(iup, batch_first=True, padding_value=0.0).to(DEV)
        emb, lens = esm2.extract_residue_batch_tensors(batch, pad_to_length=iup_pad.size(1))
        emb, lens = emb.to(DEV), lens.to(DEV)
        Bv, Lv = emb.shape[:2]
        if iup_pad.size(1) < Lv:
            iup_pad = torch.cat([iup_pad, torch.zeros(Bv, Lv-iup_pad.size(1), device=DEV)], dim=1)
        elif iup_pad.size(1) > Lv:
            iup_pad = iup_pad[:, :Lv]
        _ = m(emb, iup_pad, lens)
hook.remove()
control_pooled = np.concatenate(pooled_list, axis=0)
print(f"  Control pooled: {control_pooled.shape}")

ctrl_umap = compute_umap("control", control_pooled)
plot_umap(ctrl_umap, y_all,
          "RNA Dependency — After (Control PhytoRNP)\n512d Pooled Vector → UMAP",
          os.path.join(out_dir, "rna_control_after_umap.png"))

df = pd.DataFrame({
    "before_umap_x": X_before_umap[:, 0], "before_umap_y": X_before_umap[:, 1],
    "after_umap_x": ctrl_umap[:, 0], "after_umap_y": ctrl_umap[:, 1],
    "true_label": y_all,
})
df.to_csv(os.path.join(out_dir, "rna_control_umap.csv"), index=False)

print(f"\nDone. {OUT_BASE}/")
