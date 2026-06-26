"""
Predict LLPS scores for all background (non-LLPS) proteins using the main model.
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, HERE)

from src.model import ResidueLLPSClassifier
from src.utils import ESM2Extractor, get_iupred_scores

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
ESM2_DIR = os.path.join(HERE, "esm2")
MODEL_PATH = os.path.join(HERE, "outputs", "models", "best_model.pth")
OUT_PATH = os.path.join(HERE, "outputs", "predictions", "background_llps_scores.csv")

# 1. Load data
print("[1] Loading background sequences...", flush=True)
combined = pd.read_csv(os.path.join(HERE, "outputs", "predictions", "combined_with_umap.csv"))
bg = combined[combined["source"] == "Background"].copy()
bg["sequence"] = bg["sequence"].astype(str).str.strip().str.upper()
seqs = bg["sequence"].tolist()
N = len(seqs)
print(f"  Background proteins: {N}", flush=True)

# 2. Load model
print("[2] Loading main LLPS model...", flush=True)
extractor = ESM2Extractor(ESM2_DIR, DEVICE)
model = ResidueLLPSClassifier(d_model=1280, hidden=512).to(DEVICE).eval()
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

# 3. Predict
print(f"[3] Predicting ({N} proteins, ~{N//BATCH_SIZE} batches)...", flush=True)
probs_list = []

with torch.no_grad():
    for i in tqdm(range(0, N, BATCH_SIZE), desc="Background"):
        batch_seqs = seqs[i : i + BATCH_SIZE]

        # IUPred3
        batch_iupred = [torch.tensor(get_iupred_scores(s), dtype=torch.float32)
                        for s in batch_seqs]
        iupred_padded = pad_sequence(batch_iupred, batch_first=True, padding_value=0.0).to(DEVICE)

        # ESM2
        esm_embeds, lengths = extractor.extract_residue_batch_tensors(
            batch_seqs, pad_to_length=iupred_padded.size(1)
        )
        esm_embeds = esm_embeds.to(DEVICE)
        lengths = lengths.to(DEVICE)

        B, L = esm_embeds.shape[:2]
        if iupred_padded.size(1) < L:
            pad = torch.zeros(B, L - iupred_padded.size(1), device=DEVICE)
            iupred_padded = torch.cat([iupred_padded, pad], dim=1)
        elif iupred_padded.size(1) > L:
            iupred_padded = iupred_padded[:, :L]

        logits = model(esm_embeds, iupred_padded, lengths)
        probs_list.append(torch.sigmoid(logits).cpu())

probs = torch.cat(probs_list).numpy().flatten()
bg["LLPS_Score"] = probs
bg["LLPS_Prediction"] = (probs >= 0.5).astype(int)

# 4. Save
print(f"\n[4] Saving...", flush=True)
bg.to_csv(OUT_PATH, index=False)
print(f"Saved: {OUT_PATH}", flush=True)

n_llps = (probs >= 0.5).sum()
print(f"\n{'='*60}")
print(f"Total: {N}")
print(f"Predicted LLPS:    {n_llps} ({n_llps/N*100:.2f}%)")
print(f"Predicted Non-LLPS: {N - n_llps} ({(N-n_llps)/N*100:.2f}%)")
print(f"LLPS_Score: mean={probs.mean():.6f}, median={np.median(probs):.6f}")
