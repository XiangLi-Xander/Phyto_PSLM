"""Test XGBoost with species one-hot features added to pooled+motif."""
import os, sys, warnings, time, re
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, roc_auc_score, average_precision_score
from xgboost import XGBClassifier
from tqdm import tqdm

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from src.model import ResidueLLPSClassifier
from src.utils import ESM2Extractor, get_iupred_scores, PROJECT_ROOT

DEVICE = torch.device("cuda:0")
BATCH_SIZE = 32
ESM2_DIR = os.path.join(PROJECT_ROOT, "esm2")
LLPS_MODEL_PATH = os.path.join(PROJECT_ROOT, "outputs", "models", "best_model.pth")
OUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "benchmarks")
os.makedirs(OUT_DIR, exist_ok=True)

# Load full data with species info
print("[1] Loading data with species info...")
pred = pd.read_excel(os.path.join(PROJECT_ROOT, "outputs", "predictions", "rna_dependent_predictions_all3.xlsx"))
pred = pred[pred['LLPS_Score'] >= 0.5].copy()
pred['label'] = (pred['rna_dependent'] == 'Yes').astype(int)

from sklearn.model_selection import train_test_split
tmp, test_df = train_test_split(pred, test_size=0.15, random_state=42, stratify=pred['label'])
train_df, val_df = train_test_split(tmp, test_size=0.1765, random_state=42, stratify=tmp['label'])
print(f"  Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}")

all_seqs = train_df["sequence"].tolist() + val_df["sequence"].tolist() + test_df["sequence"].tolist()
all_species = train_df["species"].tolist() + val_df["species"].tolist() + test_df["species"].tolist()
species_list = sorted(pred['species'].unique())
sp2idx = {sp: i for i, sp in enumerate(species_list)}
species_onehot = np.zeros((len(all_species), len(species_list)), dtype=np.float32)
for i, sp in enumerate(all_species):
    species_onehot[i, sp2idx[sp]] = 1.0
print(f"  Species: {species_list}")

# Extract PhytoRNP pooled vectors
print("\n[2] Extracting 512-dim pooled vectors...")
extractor = ESM2Extractor(ESM2_DIR, DEVICE)
model = ResidueLLPSClassifier(d_model=1280, hidden=512).to(DEVICE).eval()
model.load_state_dict(torch.load(LLPS_MODEL_PATH, map_location=DEVICE))
pooled_features = []
hook = model.query_pool.register_forward_hook(lambda m, i, o: pooled_features.append(o.detach().cpu().numpy()))
t0 = time.time()
for i in tqdm(range(0, len(all_seqs), BATCH_SIZE), desc="Extracting"):
    batch = all_seqs[i:i+BATCH_SIZE]
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
pooled = np.concatenate(pooled_features, axis=0)
print(f"  Done: {pooled.shape}  ({time.time()-t0:.0f}s)")

# Motif features
print("\n[3] Computing motif features...")
def rna_motif(seqs):
    feats = []
    for s in seqs:
        s = s.upper(); L = max(len(s), 1)
        aa_ch = {'R':1,'K':1,'D':-1,'E':-1}
        aa_hp = {'A':1,'I':1,'L':1,'M':1,'F':1,'W':1,'V':1}
        if L >= 20:
            ch = [sum(aa_ch.get(a,0) for a in s[i:i+20])/20 for i in range(L-19)]
            hp = [sum(aa_hp.get(a,0) for a in s[i:i+20])/20 for i in range(L-19)]
        else:
            ch = [sum(aa_ch.get(a,0) for a in s)/L]
            hp = [sum(aa_hp.get(a,0) for a in s)/L]
        f = [len(re.findall(r'G[A-Z]G', s))/L, len(re.findall(r'G[A-Z][A-Z]G', s))/L,
             s.count('GG')/L, np.mean(ch), np.std(ch), np.mean(hp), np.std(hp)]
        mr, cr = 0, 1
        for i in range(1, L):
            if (s[i] in 'RKHD' and s[i-1] in 'RKHD') or (s[i]=='G' and s[i-1]=='G'):
                cr += 1
            else:
                mr = max(mr, cr); cr = 1
        f.append(max(mr, cr) / L)
        feats.append(f)
    return np.array(feats, dtype=np.float32)
motif = rna_motif(all_seqs)

# Combine: pool + motif + species onehot
X = np.concatenate([pooled, motif, species_onehot], axis=1).astype(np.float32)
print(f"  Combined: {X.shape[1]} dims")

n_train, n_val = len(train_df), len(val_df)
X_train, y_train = X[:n_train], train_df["label"].values.astype(int)
X_val, y_val = X[n_train:n_train+n_val], val_df["label"].values.astype(int)
X_test, y_test = X[n_train+n_val:], test_df["label"].values.astype(int)

freeze_n = pooled.shape[1]  # don't standardize onehot
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train[:, :freeze_n] = scaler.fit_transform(X_train[:, :freeze_n])
X_val[:, :freeze_n] = scaler.transform(X_val[:, :freeze_n])
X_test[:, :freeze_n] = scaler.transform(X_test[:, :freeze_n])

# Train
print("\n[4] Training XGBoost with species features...")
t0 = time.time()
xgb = XGBClassifier(
    n_estimators=500, max_depth=6, learning_rate=0.05, subsample=0.8,
    colsample_bytree=0.8, scale_pos_weight=(y_train==0).sum()/max((y_train==1).sum(),1),
    random_state=42, n_jobs=-1, verbosity=0, eval_metric="logloss",
)
xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
print(f"  Training time: {time.time()-t0:.1f}s")

probs = xgb.predict_proba(X_test)[:, 1]
preds = xgb.predict(X_test)
print(f"\n  Test: Acc={accuracy_score(y_test, preds):.4f} F1={f1_score(y_test, preds):.4f} AUC={roc_auc_score(y_test, probs):.4f} AUPR={average_precision_score(y_test, probs):.4f}")
print(confusion_matrix(y_test, preds))
print(classification_report(y_test, preds, digits=4))
print(f"\n  Without species: F1=0.7544 AUC=0.8278")
print(f"  With species:    F1={f1_score(y_test, preds):.4f} AUC={roc_auc_score(y_test, probs):.4f}")
