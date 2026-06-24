"""
Generate test predictions for the four RNA dependency models (RF, XGB, LGB, SVM)
from saved checkpoints and save to CSV files.
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                              average_precision_score, confusion_matrix,
                              classification_report)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from src.utils import PROJECT_ROOT

FEAT_CACHE = os.path.join(PROJECT_ROOT, "outputs", "benchmarks", "phyto_512d_features.npz")
MODEL_DIR = os.path.join(PROJECT_ROOT, "outputs", "models")
OUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "benchmarks")
TEST_CSV = os.path.join(PROJECT_ROOT, "data", "rna_dep", "test.csv")

MODELS = {
    "RF":  "best_rna_rf_model.pkl",
    "XGB": "best_rna_xgb_model.pkl",
    "LGB": "best_rna_lgb_model.pkl",
    "SVM": "best_rna_svm_model.pkl",
}


def main():
    # 1. Load test data (sequences, labels, features)
    test_df = pd.read_csv(TEST_CSV)
    seqs = test_df["sequence"].tolist()
    y_true = test_df["label"].values

    d = np.load(FEAT_CACHE)
    X_test = d["X_test"]
    # y_test from cache should match test_df["label"]
    assert np.array_equal(d["y_test"], y_true), "Label mismatch between cache and test.csv"

    os.makedirs(OUT_DIR, exist_ok=True)

    for name, fname in MODELS.items():
        model_path = os.path.join(MODEL_DIR, fname)
        if not os.path.exists(model_path):
            print(f"[SKIP] {name}: model not found at {model_path}")
            continue

        print(f"\n{'='*60}")
        print(f"Processing {name}")
        print(f"{'='*60}")

        bundle = joblib.load(model_path)
        model = bundle["model"]
        scaler = bundle["scaler"]

        # Scale test features
        X_te = scaler.transform(X_test).astype(np.float32)

        # Predict
        probs = model.predict_proba(X_te)[:, 1]
        preds = model.predict(X_te)

        # Metrics
        print(f"Acc:  {accuracy_score(y_true, preds):.4f}")
        print(f"F1:   {f1_score(y_true, preds):.4f}")
        print(f"AUC:  {roc_auc_score(y_true, probs):.4f}")
        print(f"AUPR: {average_precision_score(y_true, probs):.4f}")
        print(confusion_matrix(y_true, preds))
        print(classification_report(y_true, preds, digits=4))

        # Save
        out_df = pd.DataFrame({
            "sequence": seqs,
            "true_label": y_true,
            "prob_RNA_dep": probs,
            "pred_RNA_dep": preds,
        })
        out_path = os.path.join(OUT_DIR, f"rna_dep_{name.lower()}_test_predictions.csv")
        out_df.to_csv(out_path, index=False)
        print(f"Saved: {out_path} ({len(out_df)} rows)")


if __name__ == "__main__":
    main()
