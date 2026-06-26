"""
Train Random Forest for RNA dependency using PhytoRNP 512d pooled features (unified input).
"""
import os, sys, warnings, time
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, \
    accuracy_score, f1_score, roc_auc_score, average_precision_score
from sklearn.ensemble import RandomForestClassifier

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, HERE)

from src.utils import PROJECT_ROOT
import joblib

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "rna_dep")
MODEL_DIR = os.path.join(PROJECT_ROOT, "outputs", "models")
FEAT_CACHE = os.path.join(PROJECT_ROOT, "outputs", "benchmarks", "phyto_512d_features.npz")


def main():
    print("=" * 60, flush=True)
    print("Random Forest — RNA Dependency (PhytoRNP 512d)", flush=True)
    print("=" * 60, flush=True)

    if not os.path.exists(FEAT_CACHE):
        raise FileNotFoundError(f"Run train_rna_dep_lgb.py first to extract features")
    d = np.load(FEAT_CACHE)
    X_train, y_train = d["X_train"], d["y_train"]
    X_val, y_val = d["X_val"], d["y_val"]
    X_test, y_test = d["X_test"], d["y_test"]

    X_all = np.concatenate([X_train, X_val])
    y_all = np.concatenate([y_train, y_val])

    sc = StandardScaler()
    X_tr = sc.fit_transform(X_all).astype(np.float32)
    X_te = sc.transform(X_test).astype(np.float32)

    model = RandomForestClassifier(
        n_estimators=500, max_depth=20, min_samples_split=5,
        class_weight="balanced", random_state=42, n_jobs=4)

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

    joblib.dump({"model": model, "scaler": sc},
                os.path.join(MODEL_DIR, "best_rna_rf_model.pkl"))
    print(f"Saved.", flush=True)


if __name__ == "__main__":
    main()
