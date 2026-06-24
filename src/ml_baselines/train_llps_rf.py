"""
Train Random Forest classifier for LLPS prediction using handcrafted sequence features.
Uses same train/val/test splits as the deep learning model.
"""
import os, sys, warnings, time, pickle
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score, classification_report, confusion_matrix,
)

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from src.features.features_ml import extract_features

DATA_DIR = os.path.join(HERE, "data", "processed")
OUT_DIR = os.path.join(HERE, "outputs", "benchmarks")
os.makedirs(OUT_DIR, exist_ok=True)

AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")

# Load data
print("Loading data...")
train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
val_df = pd.read_csv(os.path.join(DATA_DIR, "val.csv"))
test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

X_train_raw = train_df["sequence"].tolist()
y_train = train_df["label"].values.astype(int)
X_val_raw = val_df["sequence"].tolist()
y_val = val_df["label"].values.astype(int)
X_test_raw = test_df["sequence"].tolist()
y_test = test_df["label"].values.astype(int)
print(f"Train: {len(X_train_raw)}  Val: {len(X_val_raw)}  Test: {len(X_test_raw)}")

# Extract features
print("Extracting features...")
t0 = time.time()
X_train = extract_features(X_train_raw)
X_val = extract_features(X_val_raw)
X_test = extract_features(X_test_raw)
print(f"Feature dim: {X_train.shape[1]}  Time: {time.time()-t0:.1f}s")

# Train
print("\n" + "=" * 60)
print("Random Forest (handcrafted features) — LLPS")
print("=" * 60)

rf = RandomForestClassifier(
    n_estimators=500,
    max_depth=30,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features="sqrt",
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
    verbose=0,
)
t0 = time.time()
rf.fit(X_train, y_train)
print(f"Training time: {time.time()-t0:.1f}s")

# Evaluate
for name, X, y in [("Train", X_train, y_train), ("Val", X_val, y_val), ("Test", X_test, y_test)]:
    probs = rf.predict_proba(X)[:, 1]
    preds = rf.predict(X)
    print(f"  {name:5s} | Acc:{accuracy_score(y, preds):.4f} F1:{f1_score(y, preds):.4f} "
          f"Prec:{precision_score(y, preds):.4f} Rec:{recall_score(y, preds):.4f} "
          f"AUC:{roc_auc_score(y, probs):.4f} AUPR:{average_precision_score(y, probs):.4f}")

print("\nTest Confusion Matrix:")
print(confusion_matrix(y_test, rf.predict(X_test)))
print(classification_report(y_test, rf.predict(X_test), digits=4))

# Feature importance
importances = rf.feature_importances_
feature_names = (
    [f"AAC_{aa}" for aa in AA_LIST]
    + [f"DPC_{a}{b}" for a in AA_LIST for b in AA_LIST]
    + [f"GTPC_{g3}" for g3 in range(512)]
    + [f"CTD_{i}" for i in range(56)]
    + [f"CKSAAP_{i}" for i in range(400)]
    + ["net_charge", "aromatic", "hydrophobic", "small", "proline", "glycine",
       "polar", "rg_motif", "rgg_motif", "sr_frac", "rs_motif", "pos_count",
       "neg_count", "len_norm"]
)
top_idx = np.argsort(importances)[::-1][:20]
print("\nTop-20 features:")
for i in top_idx:
    print(f"  {feature_names[i]:30s} {importances[i]:.6f}")

# Save model and test predictions
model_path = os.path.join(OUT_DIR, "llps_rf_handcrafted.pkl")
with open(model_path, "wb") as f:
    pickle.dump(rf, f)
print(f"\nModel saved to {model_path}")

pred_path = os.path.join(OUT_DIR, "llps_rf_handcrafted_predictions.csv")
test_probs = rf.predict_proba(X_test)[:, 1]
test_preds = rf.predict(X_test)
pred_df = test_df.copy()
pred_df["prob"] = test_probs
pred_df["pred"] = test_preds
pred_df.to_csv(pred_path, index=False)
print(f"Test predictions saved to {pred_path}")

print("\nDone.")
