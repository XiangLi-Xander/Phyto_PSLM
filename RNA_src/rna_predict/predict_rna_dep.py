import os, sys, warnings
import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, HERE)

from src.model import ResidueLLPSClassifier
from src.utils import ESM2Extractor, get_iupred_scores

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
ESM2_DIR = os.path.join(HERE, "esm2")
MODEL_PATH = os.path.join(HERE, "outputs", "models", "best_model.pth")
OUT_PATH = os.path.join(HERE, "outputs", "rna_dependent_predictions.xlsx")

# --------------------------------------------------------------------------
# 1. Load and merge
# --------------------------------------------------------------------------
print("[1] Loading RNA-labeled data...")

# Animal (with Uniprot IDs)
df_animal = pd.read_excel(os.path.join(HERE, "ara_all_data", "merged_RNA_with_uniprot.xlsx"))
df_animal["protein_name"] = df_animal["uniprot_id"]
df_animal["species"] = "Animal"
df_animal["rna_dependent"] = df_animal["label"].map({1: "Yes", 0: "No"})
df_animal = df_animal[["protein_name", "species", "rna_dependent", "sequence"]]
print(f"  Animal: {len(df_animal)}")

# Plant
df_plant = pd.read_excel(os.path.join(HERE, "ara_all_data", "ALL_RNA_label_with_seq_filtered.xlsx"))
df_plant["protein_name"] = df_plant["ProteinGroups"].fillna("Unknown")
df_plant["species"] = "Zea mays"
df_plant["rna_dependent"] = df_plant["Label"].map({1: "Yes", 0: "No"})
df_plant["sequence"] = df_plant["Sequence"]
df_plant = df_plant[["protein_name", "species", "rna_dependent", "sequence"]]
print(f"  Plant:  {len(df_plant)}")

df = pd.concat([df_animal, df_plant], ignore_index=True)
print(f"  Total:  {len(df)}")

# --------------------------------------------------------------------------
# 2. Model prediction
# --------------------------------------------------------------------------
print("\n[2] Loading model...")
extractor = ESM2Extractor(ESM2_DIR, device=DEVICE)
model = ResidueLLPSClassifier(d_model=1280).to(DEVICE).eval()
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

print("[3] Predicting...")
seqs = df["sequence"].tolist()
probs_list = []

for i in range(0, len(seqs), BATCH_SIZE):
    batch_seqs = seqs[i : i + BATCH_SIZE]

    esm_embeds, lengths = extractor.extract_residue_batch_tensors(batch_seqs)
    esm_embeds = esm_embeds.to(DEVICE)
    lengths = lengths.to(DEVICE)

    batch_iupred = [torch.tensor(get_iupred_scores(s), dtype=torch.float32) for s in batch_seqs]
    iupred_padded = pad_sequence(batch_iupred, batch_first=True, padding_value=0.0).to(DEVICE)

    B, L = esm_embeds.shape[:2]
    if iupred_padded.size(1) < L:
        pad = torch.zeros(B, L - iupred_padded.size(1), device=DEVICE)
        iupred_padded = torch.cat([iupred_padded, pad], dim=1)
    elif iupred_padded.size(1) > L:
        iupred_padded = iupred_padded[:, :L]

    with torch.no_grad():
        logits = model(esm_embeds, iupred_padded, lengths)
        probs_list.append(torch.sigmoid(logits).cpu())

    if (i // BATCH_SIZE) % 50 == 0:
        print(f"  {min(i + BATCH_SIZE, len(seqs))}/{len(seqs)}", flush=True)

probs = torch.cat(probs_list).numpy().flatten()
df["LLPS_Score"] = probs
df["prediction"] = (probs >= 0.5).astype(int)

# --------------------------------------------------------------------------
# 3. Save
# --------------------------------------------------------------------------
df.to_excel(OUT_PATH, index=False)
print(f"\n{'='*60}")
print(f"Saved: {OUT_PATH}")
print(f"Total proteins: {len(df)}")
print(f"  Predicted LLPS (score>=0.5): {(probs >= 0.5).sum()}")
print(f"  Predicted Non-LLPS:          {(probs < 0.5).sum()}")
print(f"\nBy RNA dependency:")
for dep in ["Yes", "No"]:
    sub = df[df["rna_dependent"] == dep]
    pos = (sub["LLPS_Score"] >= 0.5).sum()
    print(f"  RNA-dependent={dep}: {len(sub)} proteins, {pos} predicted LLPS ({pos/len(sub)*100:.1f}%)")
print(f"\nBy species:")
for sp in df["species"].unique():
    sub = df[df["species"] == sp]
    print(f"  {sp}: {len(sub)} proteins, mean LLPS_Score={sub['LLPS_Score'].mean():.4f}")
