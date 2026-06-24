"""
Predict on RNA-dep test set with all 5 models and save to CSV.
- Models 1-3 (DL RNA-dep, RF原, XGBoost原): existing models
- Models 4-5 (RF新, XGBoost新): ESM2+IUPred3-based ML models
"""
import os, sys, warnings, time
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, average_precision_score
from xgboost import XGBClassifier
from tqdm import tqdm

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, HERE)

from src.model import ResidueLLPSClassifier
from src.utils import ESM2Extractor, get_iupred_scores
from src.features.features_ml import extract_features
from src.features.features_esm_iupred import extract_features_esm_iupred

DEVICE = torch.device("cuda:1" if torch.cuda.device_count() > 1 else ("cuda:0" if torch.cuda.is_available() else "cpu"))
ESM2_DIR = os.path.join(HERE, "esm2")
DATA_DIR = os.path.join(HERE, "data", "rna_dep")
OUT_DIR = os.path.join(HERE, "outputs", "benchmarks")
os.makedirs(OUT_DIR, exist_ok=True)

print(f"Device: {DEVICE}")
print("=" * 60)

# =========================================================================
# 1. Load data
# =========================================================================
train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
val_df = pd.read_csv(os.path.join(DATA_DIR, "val.csv"))
test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

X_train_raw = train_df["sequence"].tolist()
y_train = train_df["label"].values.astype(int)
X_test_raw = test_df["sequence"].tolist()
y_test = test_df["label"].values.astype(int)

test_seqs = X_test_raw
test_labels = y_test
print(f"Test samples: {len(test_seqs)}")

# =========================================================================
# Prepare output dataframe
# =========================================================================
df_out = test_df[["sequence", "label"]].copy()
df_out.columns = ["sequence", "true_label"]

# =========================================================================
# 2. Original RF (传统特征)
# =========================================================================
print("\n[1/5] Original Random Forest ...")
t0 = time.time()
X_train_ml = extract_features(X_train_raw)
X_test_ml = extract_features(X_test_raw)
rf = RandomForestClassifier(n_estimators=500, max_depth=30, min_samples_split=5,
                             min_samples_leaf=2, max_features="sqrt",
                             class_weight="balanced", random_state=42, n_jobs=-1)
rf.fit(X_train_ml, y_train)
probs = rf.predict_proba(X_test_ml)[:, 1]
df_out["RF_原_prob"] = probs
df_out["RF_原_pred"] = (probs >= 0.5).astype(int)
print(f"  Acc: {accuracy_score(y_test, df_out['RF_原_pred']):.4f}  "
      f"F1: {f1_score(y_test, df_out['RF_原_pred']):.4f}  "
      f"AUC: {roc_auc_score(y_test, probs):.4f}  Time: {time.time()-t0:.1f}s")

# =========================================================================
# 3. Original XGBoost (传统特征)
# =========================================================================
print("\n[2/5] Original XGBoost ...")
t0 = time.time()
scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
xgb = XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8,
                    scale_pos_weight=scale_pos_weight,
                    random_state=42, n_jobs=-1, verbosity=0, eval_metric="logloss")
xgb.fit(X_train_ml, y_train)
probs = xgb.predict_proba(X_test_ml)[:, 1]
df_out["XGB_原_prob"] = probs
df_out["XGB_原_pred"] = (probs >= 0.5).astype(int)
print(f"  Acc: {accuracy_score(y_test, df_out['XGB_原_pred']):.4f}  "
      f"F1: {f1_score(y_test, df_out['XGB_原_pred']):.4f}  "
      f"AUC: {roc_auc_score(y_test, probs):.4f}  Time: {time.time()-t0:.1f}s")

# =========================================================================
# 4. New RF (ESM2 + IUPred3)
# =========================================================================
print("\n[3/5] New RF (ESM2 + IUPred3) ...")
t0 = time.time()
X_train_new = extract_features_esm_iupred(X_train_raw, ESM2_DIR, str(DEVICE), 16)
X_test_new = extract_features_esm_iupred(X_test_raw, ESM2_DIR, str(DEVICE), 16)
rf_new = RandomForestClassifier(n_estimators=500, max_depth=30, min_samples_split=5,
                                 min_samples_leaf=2, max_features="sqrt",
                                 class_weight="balanced", random_state=42, n_jobs=-1)
rf_new.fit(X_train_new, y_train)
probs = rf_new.predict_proba(X_test_new)[:, 1]
df_out["RF_新_prob"] = probs
df_out["RF_新_pred"] = (probs >= 0.5).astype(int)
print(f"  Acc: {accuracy_score(y_test, df_out['RF_新_pred']):.4f}  "
      f"F1: {f1_score(y_test, df_out['RF_新_pred']):.4f}  "
      f"AUC: {roc_auc_score(y_test, probs):.4f}  Time: {time.time()-t0:.1f}s")

# =========================================================================
# 5. New XGBoost (ESM2 + IUPred3)
# =========================================================================
print("\n[4/5] New XGBoost (ESM2 + IUPred3) ...")
t0 = time.time()
xgb_new = XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8,
                         scale_pos_weight=scale_pos_weight,
                         random_state=42, n_jobs=-1, verbosity=0, eval_metric="logloss")
xgb_new.fit(X_train_new, y_train, eval_set=[(X_test_new, y_test)], verbose=0)
probs = xgb_new.predict_proba(X_test_new)[:, 1]
df_out["XGB_新_prob"] = probs
df_out["XGB_新_pred"] = (probs >= 0.5).astype(int)
print(f"  Acc: {accuracy_score(y_test, df_out['XGB_新_pred']):.4f}  "
      f"F1: {f1_score(y_test, df_out['XGB_新_pred']):.4f}  "
      f"AUC: {roc_auc_score(y_test, probs):.4f}  Time: {time.time()-t0:.1f}s")

# =========================================================================
# 6. DL RNA-dep model (ResidueLLPSClassifier)
# =========================================================================
print("\n[5/5] DL RNA-dep model ...")
t0 = time.time()
RNA_MODEL_PATH = os.path.join(HERE, "outputs", "models", "best_rna_dep_model.pth")
extractor = ESM2Extractor(ESM2_DIR, DEVICE)
dl_model = ResidueLLPSClassifier(d_model=1280, hidden=512).to(DEVICE).eval()
dl_model.load_state_dict(torch.load(RNA_MODEL_PATH, map_location=DEVICE))

BATCH_SIZE = 32
probs_list = []
with torch.no_grad():
    for i in tqdm(range(0, len(test_seqs), BATCH_SIZE), desc="DL predicting"):
        batch_seqs = test_seqs[i:i+BATCH_SIZE]
        batch_iupred = [torch.tensor(get_iupred_scores(s), dtype=torch.float32) for s in batch_seqs]
        iupred_padded = pad_sequence(batch_iupred, batch_first=True, padding_value=0.0).to(DEVICE)
        esm_embeds, _ = extractor.extract_residue_batch_tensors(batch_seqs, pad_to_length=iupred_padded.size(1))
        esm_embeds = esm_embeds.to(DEVICE)
        B, L = esm_embeds.shape[:2]
        if iupred_padded.size(1) < L:
            pad = torch.zeros(B, L - iupred_padded.size(1), device=DEVICE)
            iupred_padded = torch.cat([iupred_padded, pad], dim=1)
        elif iupred_padded.size(1) > L:
            iupred_padded = iupred_padded[:, :L]
        logits = dl_model(esm_embeds, iupred_padded, _)
        probs_list.append(torch.sigmoid(logits).cpu())
probs = torch.cat(probs_list).numpy().flatten()
df_out["DL_RNA_dep_prob"] = probs
df_out["DL_RNA_dep_pred"] = (probs >= 0.5).astype(int)
print(f"  Acc: {accuracy_score(y_test, df_out['DL_RNA_dep_pred']):.4f}  "
      f"F1: {f1_score(y_test, df_out['DL_RNA_dep_pred']):.4f}  "
      f"AUC: {roc_auc_score(y_test, probs):.4f}  Time: {time.time()-t0:.1f}s")

# =========================================================================
# 7. Summary & Save
# =========================================================================
print("\n" + "=" * 60)
print("Test Set Result Summary")
print("=" * 60)
for col in ["RF_原", "XGB_原", "RF_新", "XGB_新", "DL_RNA_dep"]:
    preds = df_out[f"{col}_pred"].values
    probs = df_out[f"{col}_prob"].values
    print(f"  {col:15s} | Acc:{accuracy_score(y_test, preds):.4f} F1:{f1_score(y_test, preds):.4f} "
          f"Prec:{precision_score(y_test, preds):.4f} Rec:{recall_score(y_test, preds):.4f} "
          f"AUC:{roc_auc_score(y_test, probs):.4f} AUPR:{average_precision_score(y_test, probs):.4f}")

OUT_PATH = os.path.join(OUT_DIR, "all_models_test_predictions.csv")
df_out.to_csv(OUT_PATH, index=False, float_format="%.6f")
print(f"\nSaved: {OUT_PATH}")
print(f"Shape: {df_out.shape}")
