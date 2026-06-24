"""
Predict LLPS scores for ALL 3 sheets of RNA-labeled data (ZM, OS, ARA),
then predict RNA dependency for LLPS-positive proteins.
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
from src.utils import ESM2Extractor, get_iupred_scores

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
ESM2_DIR = os.path.join(HERE, "esm2")
LLPS_MODEL_PATH = os.path.join(HERE, "outputs", "models", "best_model.pth")
RNA_MODEL_PATH = os.path.join(HERE, "outputs", "models", "best_rna_dep_model.pth")
XLSX_PATH = os.path.join(HERE, "ara_all_data", "ALL_RNA_label_with_seq_filtered.xlsx")
OUT_DIR = os.path.join(HERE, "outputs", "predictions")
os.makedirs(OUT_DIR, exist_ok=True)

# --------------------------------------------------------------------------
# 1. Load all 3 sheets
# --------------------------------------------------------------------------
print("[1] Loading all 3 sheets...", flush=True)
dfs = {}
for sheet in ["ZM", "OS", "ARA"]:
    df = pd.read_excel(XLSX_PATH, sheet_name=sheet)
    df["species"] = {"ZM": "Zea mays", "OS": "Oryza sativa", "ARA": "Arabidopsis thaliana"}[sheet]
    df["protein_name"] = df["ProteinGroups"].fillna("Unknown")
    df["rna_dependent"] = df["Label"].map({1: "Yes", 0: "No"})
    df["sequence"] = df["Sequence"].astype(str).str.strip().str.upper()
    dfs[sheet] = df[["protein_name", "species", "rna_dependent", "sequence"]]
    print(f"  {sheet}: {len(df)} proteins (RNA-dep={(df['rna_dependent']=='Yes').sum()}, RNA-indep={(df['rna_dependent']=='No').sum()})", flush=True)

all_df = pd.concat(dfs.values(), ignore_index=True).dropna(subset=["sequence"])
all_df = all_df[all_df["sequence"].str.len() >= 50]
all_df = all_df.drop_duplicates(subset=["sequence"])
print(f"  Total clean: {len(all_df)}", flush=True)

# --------------------------------------------------------------------------
# 2. Predict LLPS with main model
# --------------------------------------------------------------------------
print("\n[2] Loading LLPS model & predicting...", flush=True)
extractor = ESM2Extractor(ESM2_DIR, DEVICE)
llps_model = ResidueLLPSClassifier(d_model=1280, hidden=512).to(DEVICE).eval()
llps_model.load_state_dict(torch.load(LLPS_MODEL_PATH, map_location=DEVICE))

seqs = all_df["sequence"].tolist()
N = len(seqs)
llps_probs = []

with torch.no_grad():
    for i in tqdm(range(0, N, BATCH_SIZE), desc="LLPS predict"):
        batch_seqs = seqs[i : i + BATCH_SIZE]
        batch_iupred = [torch.tensor(get_iupred_scores(s), dtype=torch.float32) for s in batch_seqs]
        iupred_padded = pad_sequence(batch_iupred, batch_first=True, padding_value=0.0).to(DEVICE)
        esm_embeds, lengths = extractor.extract_residue_batch_tensors(batch_seqs, pad_to_length=iupred_padded.size(1))
        esm_embeds = esm_embeds.to(DEVICE)
        lengths = lengths.to(DEVICE)
        B, L = esm_embeds.shape[:2]
        if iupred_padded.size(1) < L:
            pad = torch.zeros(B, L - iupred_padded.size(1), device=DEVICE)
            iupred_padded = torch.cat([iupred_padded, pad], dim=1)
        elif iupred_padded.size(1) > L:
            iupred_padded = iupred_padded[:, :L]
        logits = llps_model(esm_embeds, iupred_padded, lengths)
        llps_probs.append(torch.sigmoid(logits).cpu())

all_df["LLPS_Score"] = torch.cat(llps_probs).numpy().flatten()

# Save LLPS predictions
llps_out = os.path.join(OUT_DIR, "rna_dependent_predictions_all3.xlsx")
all_df.to_excel(llps_out, index=False)
print(f"  Saved: {llps_out}", flush=True)

n_llps = (all_df["LLPS_Score"] >= 0.5).sum()
print(f"\n  Total: {N}, Predicted LLPS: {n_llps} ({n_llps/N*100:.1f}%)", flush=True)
for sp in all_df["species"].unique():
    sub = all_df[all_df["species"] == sp]
    l = (sub["LLPS_Score"] >= 0.5).sum()
    print(f"  {sp}: {len(sub)} -> {l} LLPS ({l/len(sub)*100:.1f}%)", flush=True)

# --------------------------------------------------------------------------
# 3. Filter LLPS >= 0.5 and predict RNA dependency
# --------------------------------------------------------------------------
llps_pos = all_df[all_df["LLPS_Score"] >= 0.5].copy()
if len(llps_pos) == 0:
    print("\nNo LLPS-positive proteins found. Skipping RNA dependency prediction.", flush=True)
    exit()

print(f"\n[3] Predicting RNA dependency for {len(llps_pos)} LLPS-positive proteins...", flush=True)
rna_model = ResidueLLPSClassifier(d_model=1280, hidden=512).to(DEVICE).eval()
rna_model.load_state_dict(torch.load(RNA_MODEL_PATH, map_location=DEVICE))

seqs_llps = llps_pos["sequence"].tolist()
M = len(seqs_llps)
rnadep_probs = []

with torch.no_grad():
    for i in tqdm(range(0, M, BATCH_SIZE), desc="RNA dep predict"):
        batch_seqs = seqs_llps[i : i + BATCH_SIZE]
        batch_iupred = [torch.tensor(get_iupred_scores(s), dtype=torch.float32) for s in batch_seqs]
        iupred_padded = pad_sequence(batch_iupred, batch_first=True, padding_value=0.0).to(DEVICE)
        esm_embeds, lengths = extractor.extract_residue_batch_tensors(batch_seqs, pad_to_length=iupred_padded.size(1))
        esm_embeds = esm_embeds.to(DEVICE)
        lengths = lengths.to(DEVICE)
        B, L = esm_embeds.shape[:2]
        if iupred_padded.size(1) < L:
            pad = torch.zeros(B, L - iupred_padded.size(1), device=DEVICE)
            iupred_padded = torch.cat([iupred_padded, pad], dim=1)
        elif iupred_padded.size(1) > L:
            iupred_padded = iupred_padded[:, :L]
        logits = rna_model(esm_embeds, iupred_padded, lengths)
        rnadep_probs.append(torch.sigmoid(logits).cpu())

llps_pos["RNA_Dep_Score"] = torch.cat(rnadep_probs).numpy().flatten()
llps_pos["RNA_Dependent"] = (llps_pos["RNA_Dep_Score"] >= 0.5).astype(int)

# Save
rna_out = os.path.join(OUT_DIR, "rna_dep_final_3species.xlsx")
llps_pos.to_excel(rna_out, index=False)
print(f"  Saved: {rna_out}", flush=True)

# Summary
n_dep = (llps_pos["RNA_Dependent"] == 1).sum()
n_ind = (llps_pos["RNA_Dependent"] == 0).sum()
print(f"\n{'='*60}")
print(f"LLPS-positive classified: {M}")
print(f"  RNA-dependent:   {n_dep} ({n_dep/M*100:.1f}%)")
print(f"  RNA-independent: {n_ind} ({n_ind/M*100:.1f}%)")
for sp in llps_pos["species"].unique():
    sub = llps_pos[llps_pos["species"] == sp]
    dep = (sub["RNA_Dependent"] == 1).sum()
    print(f"  {sp}: {len(sub)} -> {dep} RNA-dep ({dep/len(sub)*100:.1f}%)")
