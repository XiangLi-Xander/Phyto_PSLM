"""
Apply RNA dependency classifier to all LLPS-positive proteins.

Output: table with protein info, LLPS score, RNA dependency score & prediction.
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, HERE)

from src.model import ResidueLLPSClassifier
from src.utils import ESM2Extractor, get_iupred_scores, clean_sequence_df

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
ESM2_DIR = os.path.join(HERE, "esm2")
RNA_MODEL_PATH = os.path.join(HERE, "outputs", "models", "best_rna_dep_model.pth")
OUT_PATH = os.path.join(HERE, "outputs", "predictions", "rna_dep_predictions.csv")

# --------------------------------------------------------------------------
# 1. Load RNA dependency model
# --------------------------------------------------------------------------
print("[1] Loading RNA dependency model...", flush=True)
extractor = ESM2Extractor(ESM2_DIR, DEVICE)
rna_model = ResidueLLPSClassifier(d_model=1280, hidden=512).to(DEVICE).eval()
rna_model.load_state_dict(torch.load(RNA_MODEL_PATH, map_location=DEVICE))

# --------------------------------------------------------------------------
# 2. Load all data with LLPS scores and filter for LLPS_Score >= 0.5
# --------------------------------------------------------------------------
print("[2] Loading data...", flush=True)

# Load combined_with_umap.csv which has Background + LLPS + RBP with LLPS scores
combined = pd.read_csv(os.path.join(HERE, "outputs", "predictions", "combined_with_umap.csv"))
print(f"  Total proteins: {len(combined)}", flush=True)

# Filter LLPS-positive (LLPS_Score >= 0.5)
# Note: Background proteins don't have LLPS_Score (NaN for many)
# We want to predict RNA dependency for LLPS and RBP proteins
llps_positive = combined[combined["source"].isin(["LLPS", "RBP"])].copy()
print(f"  LLPS + RBP proteins: {len(llps_positive)}", flush=True)

# Clean sequences
llps_positive["sequence"] = llps_positive["sequence"].astype(str).str.strip().str.upper()
llps_positive = llps_positive.dropna(subset=["sequence"])

seqs = llps_positive["sequence"].tolist()
N = len(seqs)
print(f"  Valid sequences: {N}", flush=True)

# --------------------------------------------------------------------------
# 3. Predict RNA dependency
# --------------------------------------------------------------------------
print("[3] Predicting RNA dependency...", flush=True)
rnadep_probs = []

with torch.no_grad():
    for i in tqdm(range(0, N, BATCH_SIZE), desc="Predicting"):
        batch_seqs = seqs[i : i + BATCH_SIZE]

        # IUPred3
        batch_iupred = [torch.tensor(get_iupred_scores(s), dtype=torch.float32)
                        for s in batch_seqs]
        iupred_padded = pad_sequence(batch_iupred, batch_first=True, padding_value=0.0).to(DEVICE)

        # ESM2
        esm_embeds, _ = extractor.extract_residue_batch_tensors(
            batch_seqs, pad_to_length=iupred_padded.size(1)
        )
        esm_embeds = esm_embeds.to(DEVICE)

        B, L = esm_embeds.shape[:2]
        if iupred_padded.size(1) < L:
            pad = torch.zeros(B, L - iupred_padded.size(1), device=DEVICE)
            iupred_padded = torch.cat([iupred_padded, pad], dim=1)
        elif iupred_padded.size(1) > L:
            iupred_padded = iupred_padded[:, :L]

        logits = rna_model(esm_embeds, iupred_padded, _)
        rnadep_probs.append(torch.sigmoid(logits).cpu())

rnadep_probs = torch.cat(rnadep_probs).numpy().flatten()
llps_positive["RNA_Dep_Score"] = rnadep_probs
llps_positive["RNA_Dependent"] = (rnadep_probs >= 0.5).astype(int)
llps_positive["RNA_Dependent_Label"] = llps_positive["RNA_Dependent"].map(
    {1: "Yes", 0: "No"}
)

# --------------------------------------------------------------------------
# 4. Save
# --------------------------------------------------------------------------
print("\n[4] Saving...", flush=True)
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
llps_positive.to_csv(OUT_PATH, index=False)
print(f"Saved: {OUT_PATH}", flush=True)

# Summary
n_rna_dep = (rnadep_probs >= 0.5).sum()
n_non = (rnadep_probs < 0.5).sum()
print(f"\n{'='*60}")
print(f"Total LLPS proteins classified: {N}")
print(f"  RNA-dependent:   {n_rna_dep} ({n_rna_dep/N*100:.1f}%)")
print(f"  RNA-independent: {n_non} ({n_non/N*100:.1f}%)")
print(f"  RNA_Dep_Score: mean={rnadep_probs.mean():.4f}, median={np.median(rnadep_probs):.4f}")
for src in ["LLPS", "RBP"]:
    sub = llps_positive[llps_positive["source"] == src]
    dep = (sub["RNA_Dep_Score"] >= 0.5).sum()
    print(f"  {src}: {len(sub)} proteins, {dep} RNA-dependent ({dep/len(sub)*100:.1f}%)")
