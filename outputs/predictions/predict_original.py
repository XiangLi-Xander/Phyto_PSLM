import os, sys, warnings
import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence

warnings.filterwarnings("ignore", message="Precision is ill-defined")
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, HERE)

from src.model import ResidueLLPSClassifier
from src.utils import ESM2Extractor, get_iupred_scores

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
TOP = HERE
TEST_CSV = os.path.join(TOP, "data", "processed", "test.csv")
ESM2_DIR = os.path.join(TOP, "esm2")
MODEL_PATH = os.path.join(TOP, "outputs", "models", "best_model.pth")
OUT_PATH = os.path.join(TOP, "outputs", "predictions", "original_test_predictions.xlsx")

extractor = ESM2Extractor(ESM2_DIR, device=DEVICE)
model = ResidueLLPSClassifier(d_model=1280).to(DEVICE).eval()
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

df = pd.read_csv(TEST_CSV)
seqs = df["sequence"].tolist()

probs_list = []
for i in range(0, len(seqs), BATCH_SIZE):
    batch_seqs = seqs[i:i+BATCH_SIZE]

    esm_embeds, lengths = extractor.extract_residue_batch_tensors(batch_seqs)
    esm_embeds = esm_embeds.to(DEVICE)
    lengths = lengths.to(DEVICE)

    batch_iupred = []
    for seq in batch_seqs:
        batch_iupred.append(torch.tensor(get_iupred_scores(seq), dtype=torch.float32))
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
        print(f"  {min(i+BATCH_SIZE, len(seqs))}/{len(seqs)}", flush=True)

probs = torch.cat(probs_list).numpy()
print(f"\nTest -> F1: {f1_score(df['label'], (probs>=0.5).astype(int)):.4f} | AUC: {roc_auc_score(df['label'], probs):.4f} | AUPR: {average_precision_score(df['label'], probs):.4f}", flush=True)

df_out = df[["UniprotEntry","sequence","Species","label"]].copy()
df_out.columns = ["uniprot_entry","sequence","species","true_label"]
df_out["probability"] = probs
df_out["prediction"] = (probs >= 0.5).astype(int)
df_out.to_excel(OUT_PATH, index=False)
print(f"Saved: {OUT_PATH}", flush=True)
