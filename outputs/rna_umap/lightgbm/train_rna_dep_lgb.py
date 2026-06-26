"""
Train LightGBM for RNA dependency prediction using PhytoRNP 512d pooled features.

Best config: lr=0.03, max_depth=10, num_leaves=63, subsample=0.8, colsample_bytree=0.8
Test F1=0.7743, AUC=0.8450, AUPR=0.8588
"""
import os, sys, warnings, time
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, \
    accuracy_score, f1_score, roc_auc_score, average_precision_score

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
os.makedirs(MODEL_DIR, exist_ok=True)

BATCH = 32


def extract_512d():
    if os.path.exists(FEAT_CACHE):
        d = np.load(FEAT_CACHE)
        print(f"Loaded cached features: {FEAT_CACHE}")
        return d["X_train"], d["y_train"], d["X_val"], d["y_val"], \
               d["X_test"], d["y_test"]

    print("Extracting PhytoRNP 512d pooled features...")
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    val_df = pd.read_csv(os.path.join(DATA_DIR, "val.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    all_seqs = train_df["sequence"].tolist() + val_df["sequence"].tolist() + test_df["sequence"].tolist()

    extractor = ESM2Extractor(ESM2_DIR, DEVICE)
    phyto = ResidueLLPSClassifier(d_model=1280, hidden=512).to(DEVICE).eval()
    phyto.load_state_dict(torch.load(LLPS_MODEL, map_location=DEVICE))

    pooled_list = []
    def capture_pooled(_m, _i, output):
        pooled_list.append(output.detach().cpu().numpy())
    hook = phyto.query_pool.register_forward_hook(capture_pooled)

    for i in tqdm(range(0, len(all_seqs), BATCH), desc="Extracting"):
        batch = all_seqs[i:i+BATCH]
        iup = [torch.tensor(get_iupred_scores(s), dtype=torch.float32) for s in batch]
        iup_pad = pad_sequence(iup, batch_first=True, padding_value=0.0).to(DEVICE)
        esm, lens = extractor.extract_residue_batch_tensors(batch, pad_to_length=iup_pad.size(1))
        esm, lens = esm.to(DEVICE), lens.to(DEVICE)
        Bv, Lv = esm.shape[:2]
        if iup_pad.size(1) < Lv:
            iup_pad = torch.cat([iup_pad, torch.zeros(Bv, Lv-iup_pad.size(1), device=DEVICE)], dim=1)
        elif iup_pad.size(1) > Lv:
            iup_pad = iup_pad[:, :Lv]
        with torch.no_grad():
            _ = phyto(esm, iup_pad, lens)
    hook.remove()

    pooled = np.concatenate(pooled_list, axis=0)
    n_tr, n_v = len(train_df), len(val_df)
    X_train = pooled[:n_tr]; y_train = train_df["label"].values
    X_val = pooled[n_tr:n_tr+n_v]; y_val = val_df["label"].values
    X_test = pooled[n_tr+n_v:]; y_test = test_df["label"].values
    np.savez(FEAT_CACHE, X_train=X_train, y_train=y_train, X_val=X_val,
             y_val=y_val, X_test=X_test, y_test=y_test)
    print(f"Saved: {FEAT_CACHE}")
    return X_train, y_train, X_val, y_val, X_test, y_test


def main():
    print("=" * 60, flush=True)
    print("LightGBM — RNA Dependency Classifier (PhytoRNP 512d)", flush=True)
    print("=" * 60, flush=True)

    X_train, y_train, X_val, y_val, X_test, y_test = extract_512d()
    X_all = np.concatenate([X_train, X_val])
    y_all = np.concatenate([y_train, y_val])

    sc = StandardScaler()
    X_tr = sc.fit_transform(X_all).astype(np.float32)
    X_te = sc.transform(X_test).astype(np.float32)

    # Best config from sweep
    model = lgb.LGBMClassifier(
        n_estimators=500, learning_rate=0.03, max_depth=10, num_leaves=63,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
        random_state=42, n_jobs=4, verbosity=-1)

    t0 = time.time()
    model.fit(X_tr, y_all)
    print(f"Training time: {time.time()-t0:.1f}s", flush=True)

    probs = model.predict_proba(X_te)[:, 1]
    preds = model.predict(X_te)

    print(f"\nTest Results:", flush=True)
    print(f"  Acc:   {accuracy_score(y_test, preds):.4f}", flush=True)
    print(f"  F1:    {f1_score(y_test, preds):.4f}", flush=True)
    print(f"  AUC:   {roc_auc_score(y_test, probs):.4f}", flush=True)
    print(f"  AUPR:  {average_precision_score(y_test, probs):.4f}", flush=True)
    print(f"\n{confusion_matrix(y_test, preds)}", flush=True)
    print(classification_report(y_test, preds, digits=4), flush=True)

    # Save
    import joblib
    out_path = os.path.join(MODEL_DIR, "best_rna_lgb_model.pkl")
    joblib.dump({"model": model, "scaler": sc}, out_path)
    print(f"\nSaved: {out_path}", flush=True)


if __name__ == "__main__":
    main()
