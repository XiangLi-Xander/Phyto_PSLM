"""
Train XGBoost for RNA dependency using PhytoRNP 512d pooled features (unified input).
"""
import os, sys, warnings, time
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, \
    accuracy_score, f1_score, roc_auc_score, average_precision_score
from xgboost import XGBClassifier

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, HERE)

from src.model import ResidueLLPSClassifier
from src.utils import ESM2Extractor, get_iupred_scores, PROJECT_ROOT
import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

DEVICE = torch.device("cuda:1" if torch.cuda.device_count() > 1 else "cuda:0")
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "rna_dep")
ESM2_DIR = os.path.join(PROJECT_ROOT, "esm2")
LLPS_MODEL = os.path.join(PROJECT_ROOT, "outputs", "models", "best_model.pth")
MODEL_DIR = os.path.join(PROJECT_ROOT, "outputs", "models")
FEAT_CACHE = os.path.join(PROJECT_ROOT, "outputs", "benchmarks", "phyto_512d_features.npz")
BATCH = 32


def load_512d():
    if not os.path.exists(FEAT_CACHE):
        raise FileNotFoundError(f"Run train_rna_dep_lgb.py first to extract features")
    d = np.load(FEAT_CACHE)
    return d["X_train"], d["y_train"], d["X_val"], d["y_val"], d["X_test"], d["y_test"]


def main():
    print("=" * 60, flush=True)
    print("XGBoost — RNA Dependency (PhytoRNP 512d)", flush=True)
    print("=" * 60, flush=True)

    X_train, y_train, X_val, y_val, X_test, y_test = load_512d()
    X_all = np.concatenate([X_train, X_val])
    y_all = np.concatenate([y_train, y_val])

    sc = StandardScaler()
    X_tr = sc.fit_transform(X_all).astype(np.float32)
    X_te = sc.transform(X_test).astype(np.float32)

    model = XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=(y_train == 0).sum() / max((y_train == 1).sum(), 1),
        random_state=42, n_jobs=4, verbosity=0)

    t0 = time.time()
    model.fit(X_tr, y_all)
    print(f"Training: {time.time()-t0:.1f}s", flush=True)

    probs = model.predict_proba(X_te)[:, 1]
    preds = model.predict(X_te)

    print(f"\nTest Results:", flush=True)
    print(f"  Acc:   {accuracy_score(y_test, preds):.4f}", flush=True)
    print(f"  F1:    {f1_score(y_test, preds):.4f}", flush=True)
    print(f"  AUC:   {roc_auc_score(y_test, probs):.4f}", flush=True)
    print(f"  AUPR:  {average_precision_score(y_test, probs):.4f}", flush=True)
    print(f"\n{confusion_matrix(y_test, preds)}", flush=True)
    print(classification_report(y_test, preds, digits=4), flush=True)

    import joblib
    joblib.dump({"model": model, "scaler": sc},
                os.path.join(MODEL_DIR, "best_rna_xgb_model.pkl"))
    print(f"Saved.", flush=True)


if __name__ == "__main__":
    main()
