"""
Train XGBoost for LLPS prediction using ESM2 + IUPred3 features.

Uses the same data splits as the deep learning model, but with pooled
fixed-length features suitable for classical ML classifiers.
"""
import os, sys, warnings, time, argparse, pickle
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score, classification_report, confusion_matrix,
)
from xgboost import XGBClassifier

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from src.features.features_esm_iupred import extract_features_all_splits

DATA_DIR = os.path.join(HERE, "data", "processed")
ESM2_DIR = os.path.join(HERE, "esm2")
OUT_DIR = os.path.join(HERE, "outputs", "benchmarks")
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None,
                        help="Device for ESM2 extraction (default: cuda if available)")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_estimators", type=int, default=500)
    parser.add_argument("--max_depth", type=int, default=6)
    parser.add_argument("--learning_rate", type=float, default=0.05)
    args = parser.parse_args()

    if args.device is None:
        try:
            import torch
            args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        except ImportError:
            args.device = "cpu"

    print(f"Device: {args.device}")

    # Load data
    print("\nLoading data ...")
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

    # Extract ESM2 + IUPred3 features (single model load for all splits)
    print("\nExtracting features ...")
    t0 = time.time()
    feat_cache = os.path.join(OUT_DIR, "esm_iupred_features.npz")
    if os.path.exists(feat_cache):
        print(f"Loading cached features from {feat_cache}...")
        data = np.load(feat_cache)
        X_train, X_val, X_test = data["X_train"], data["X_val"], data["X_test"]
    else:
        X_train, X_val, X_test = extract_features_all_splits(
            X_train_raw, X_val_raw, X_test_raw,
            ESM2_DIR, args.device, args.batch_size,
        )
        np.savez_compressed(feat_cache, X_train=X_train, X_val=X_val, X_test=X_test)
        print(f"Features cached to {feat_cache}")
    print(f"Feature dim: {X_train.shape[1]}  Time: {time.time()-t0:.1f}s")

    # Train
    print("\n" + "=" * 60)
    print("XGBoost (ESM2 + IUPred3 features) — LLPS")
    print("=" * 60)

    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    xgb = XGBClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
        eval_metric="logloss",
    )
    t0 = time.time()
    xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=0)
    print(f"Training time: {time.time()-t0:.1f}s")

    # Evaluate
    for name, X, y in [("Train", X_train, y_train), ("Val", X_val, y_val), ("Test", X_test, y_test)]:
        probs = xgb.predict_proba(X)[:, 1]
        preds = xgb.predict(X)
        print(f"  {name:5s} | Acc:{accuracy_score(y, preds):.4f} F1:{f1_score(y, preds):.4f} "
              f"Prec:{precision_score(y, preds):.4f} Rec:{recall_score(y, preds):.4f} "
              f"AUC:{roc_auc_score(y, probs):.4f} AUPR:{average_precision_score(y, probs):.4f}")

    print("\nTest Confusion Matrix:")
    print(confusion_matrix(y_test, xgb.predict(X_test)))
    print(classification_report(y_test, xgb.predict(X_test), digits=4))

    # Feature importance
    importances = xgb.feature_importances_
    iupred_names = ["IUPred_mean", "IUPred_std", "IUPred_min", "IUPred_max",
                    "IUPred_median", "IUPred_Q25", "IUPred_Q75", "IUPred_frac_disorder", "IUPred_len"]
    esm2_names = [f"ESM2_{i}" for i in range(1280)]
    all_names = esm2_names + iupred_names
    top_idx = np.argsort(importances)[::-1][:20]
    print("\nTop-20 features:")
    for idx in top_idx:
        print(f"  {all_names[idx]:30s} {importances[idx]:.6f}")

    # Save model and test predictions
    model_path = os.path.join(OUT_DIR, "llps_xgb_esm_iupred.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(xgb, f)
    print(f"\nModel saved to {model_path}")

    pred_path = os.path.join(OUT_DIR, "llps_xgb_esm_iupred_predictions.csv")
    test_probs = xgb.predict_proba(X_test)[:, 1]
    test_preds = xgb.predict(X_test)
    pred_df = test_df.copy()
    pred_df["prob"] = test_probs
    pred_df["pred"] = test_preds
    pred_df.to_csv(pred_path, index=False)
    print(f"Test predictions saved to {pred_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
