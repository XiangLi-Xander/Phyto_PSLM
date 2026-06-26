"""
Unified benchmark: extract PhytoRNP 512d pooled features once,
train multiple DL classifiers on the same input, compare fairly.

Feature: PhytoRNP MultiQueryAttentionPooling output (512d).
All classifiers see identical input.  No ESM2/LoRA at train time.
"""
import os, sys, warnings, time, json
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             roc_auc_score, average_precision_score)

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from src.model import ResidueLLPSClassifier
from src.utils import ESM2Extractor, get_iupred_scores, PROJECT_ROOT, compute_metrics

DEVICE = torch.device("cuda:1" if torch.cuda.device_count() > 1 else "cuda:0")
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "rna_dep")
ESM2_DIR = os.path.join(PROJECT_ROOT, "esm2")
FEAT_CACHE = os.path.join(PROJECT_ROOT, "outputs", "benchmarks", "phyto_512d_features.npz")
LLPS_MODEL = os.path.join(PROJECT_ROOT, "outputs", "models", "best_model.pth")
MODEL_DIR = os.path.join(PROJECT_ROOT, "outputs", "models")
LOG_DIR = os.path.join(PROJECT_ROOT, "outputs", "logs")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(os.path.dirname(FEAT_CACHE), exist_ok=True)

BATCH = 32
TRAIN_EPOCHS = 300
EARLY_STOP = 25


# ============================================================================
# 1. Classifier architectures (all take 512d input)
# ============================================================================

class MLP2(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(256, 1))
    def forward(self, x): return self.net(x).squeeze(-1)

class MLP3(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(128, 1))
    def forward(self, x): return self.net(x).squeeze(-1)

class MLP4(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(64, 1))
    def forward(self, x): return self.net(x).squeeze(-1)

class MLP5(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.BatchNorm1d(32), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(32, 1))
    def forward(self, x): return self.net(x).squeeze(-1)

class MLPResidual(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(512, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.block1 = nn.Sequential(nn.Linear(256, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.3))
        self.block2 = nn.Sequential(nn.Linear(256, 128), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.3))
        self.block3 = nn.Sequential(nn.Linear(128, 64), nn.BatchNorm1d(64), nn.GELU(), nn.Dropout(0.2))
        self.head = nn.Linear(64, 1)
        self.skip = nn.Linear(256, 256)

    def forward(self, x):
        x = F.gelu(self.bn1(self.proj(x)))
        x = x + self.skip(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.head(x).squeeze(-1)

class MLPWide(nn.Module):
    """Wider but shallower: 512→512→256→1"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(512, 512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(256, 1))
    def forward(self, x): return self.net(x).squeeze(-1)

class MLPNarrow(nn.Module):
    """Narrow bottleneck: 512→64→32→1"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(512, 64), nn.BatchNorm1d(64), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.BatchNorm1d(32), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(32, 1))
    def forward(self, x): return self.net(x).squeeze(-1)

class MLPAttention(nn.Module):
    """Self-attention: project 512→8*256, reshape to 8 tokens, attend, pool"""
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(512, 8 * 128)
        self.attn = nn.MultiheadAttention(128, num_heads=4, dropout=0.2, batch_first=True)
        self.norm = nn.LayerNorm(128)
        self.head = nn.Sequential(
            nn.Linear(128, 64), nn.GELU(), nn.Dropout(0.2), nn.Linear(64, 1))
    def forward(self, x):
        x = self.proj(x).view(x.size(0), 8, 128)  # (B, 8, 128)
        x, _ = self.attn(x, x, x)
        x = x.mean(dim=1)  # (B, 128)
        x = self.norm(x)
        return self.head(x).squeeze(-1)


ARCHITECTURES = {
    "MLP2": MLP2,
    "MLP3": MLP3,
    "MLP4": MLP4,
    "MLP5": MLP5,
    "MLP_Residual": MLPResidual,
    "MLP_Wide": MLPWide,
    "MLP_Narrow": MLPNarrow,
    "MLP_Attention": MLPAttention,
}


# ============================================================================
# 2. Feature extraction
# ============================================================================

def extract_512d_features():
    """Extract PhytoRNP 512d pooled vectors for all RNA-dep sequences."""
    if os.path.exists(FEAT_CACHE):
        data = np.load(FEAT_CACHE)
        print(f"[Cache] Loaded features from {FEAT_CACHE}")
        return (data["X_train"], data["y_train"], data["X_val"], data["y_val"],
                data["X_test"], data["y_test"])

    print("[Extract] Computing PhytoRNP 512d pooled vectors...")
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    val_df = pd.read_csv(os.path.join(DATA_DIR, "val.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    all_seqs = train_df["sequence"].tolist() + val_df["sequence"].tolist() + test_df["sequence"].tolist()
    y_train = train_df["label"].values.astype(np.float32)
    y_val = val_df["label"].values.astype(np.float32)
    y_test = test_df["label"].values.astype(np.float32)

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
            iup_pad = torch.cat([iup_pad, torch.zeros(Bv, Lv - iup_pad.size(1), device=DEVICE)], dim=1)
        elif iup_pad.size(1) > Lv:
            iup_pad = iup_pad[:, :Lv]
        with torch.no_grad():
            _ = phyto(esm, iup_pad, lens)
    hook.remove()

    pooled = np.concatenate(pooled_list, axis=0)
    n_train, n_val = len(train_df), len(val_df)
    X_train = pooled[:n_train]
    X_val = pooled[n_train:n_train+n_val]
    X_test = pooled[n_train+n_val:]

    np.savez(FEAT_CACHE, X_train=X_train, y_train=y_train, X_val=X_val,
             y_val=y_val, X_test=X_test, y_test=y_test)
    print(f"[Extract] Saved to {FEAT_CACHE} | shapes: {X_train.shape} {X_val.shape} {X_test.shape}")
    return X_train, y_train, X_val, y_val, X_test, y_test


# ============================================================================
# 3. Training + Evaluation
# ============================================================================

def train_one(name, model_cls, X_train, y_train, X_val, y_val, X_test, y_test, seed=42):
    torch.manual_seed(seed); np.random.seed(seed)

    # Standardize
    sc = StandardScaler()
    Xt = torch.tensor(sc.fit_transform(X_train), dtype=torch.float32)
    Xv = torch.tensor(sc.transform(X_val), dtype=torch.float32)
    Xe = torch.tensor(sc.transform(X_test), dtype=torch.float32)

    train_ds = TensorDataset(Xt, torch.tensor(y_train))
    val_ds = TensorDataset(Xv, torch.tensor(y_val))
    test_ds = TensorDataset(Xe, torch.tensor(y_test))

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)

    model = model_cls().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    criterion = nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=TRAIN_EPOCHS)

    best_val_f1, best_state, no_improve = 0.0, None, 0

    for epoch in range(TRAIN_EPOCHS):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_loss += loss.item()
        sched.step()

        if epoch % 20 == 0 or epoch == TRAIN_EPOCHS - 1:
            model.eval()
            val_probs, val_labels = [], []
            with torch.no_grad():
                for xb, yb in val_loader:
                    val_probs.append(torch.sigmoid(model(xb.to(DEVICE))).cpu())
                    val_labels.append(yb)
            val_probs = torch.cat(val_probs).numpy()
            val_labels = torch.cat(val_labels).numpy()
            val_f1 = f1_score(val_labels, (val_probs >= 0.5).astype(int))

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= EARLY_STOP: break

    # Test
    model.load_state_dict(best_state)
    model.eval()
    test_probs, test_labels = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            test_probs.append(torch.sigmoid(model(xb.to(DEVICE))).cpu())
            test_labels.append(yb)
    test_probs = torch.cat(test_probs).numpy()
    test_labels = torch.cat(test_labels).numpy()
    test_preds = (test_probs >= 0.5).astype(int)

    result = {
        "name": name,
        "params": n_params,
        "val_best_f1": float(best_val_f1),
        "test_acc": float(accuracy_score(test_labels, test_preds)),
        "test_f1": float(f1_score(test_labels, test_preds)),
        "test_prec": float(precision_score(test_labels, test_preds, zero_division=0)),
        "test_rec": float(recall_score(test_labels, test_preds, zero_division=0)),
        "test_auc": float(roc_auc_score(test_labels, test_probs)),
        "test_aupr": float(average_precision_score(test_labels, test_probs)),
    }
    return result


# ============================================================================
# 4. Main
# ============================================================================

def main():
    print("=" * 70)
    print("Unified PhytoRNP 512d Classifier Benchmark")
    print(f"Device: {DEVICE} | Epochs: {TRAIN_EPOCHS} | Early stop: {EARLY_STOP}")
    print("=" * 70)

    X_train, y_train, X_val, y_val, X_test, y_test = extract_512d_features()
    print(f"Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")

    results = []
    for name, model_cls in ARCHITECTURES.items():
        print(f"\n{'─'*50}\n  Training: {name}\n{'─'*50}")
        t0 = time.time()
        r = train_one(name, model_cls, X_train, y_train, X_val, y_val, X_test, y_test)
        r["time_s"] = time.time() - t0
        results.append(r)
        print(f"  {name:20s} | Params:{r['params']:>8,} | Val F1:{r['val_best_f1']:.4f} | "
              f"Test F1:{r['test_f1']:.4f} | AUC:{r['test_auc']:.4f} | AUPR:{r['test_aupr']:.4f} | "
              f"{r['time_s']:.0f}s")

    # Summary table
    print(f"\n{'='*80}")
    print(f"{'Model':<20s} {'Params':>8s} {'Test F1':>8s} {'Test AUC':>8s} {'Test AUPR':>8s} {'Test Acc':>8s} {'Val F1':>8s} {'Time':>5s}")
    print(f"{'─'*80}")
    results.sort(key=lambda x: x["test_f1"], reverse=True)
    for r in results:
        print(f"{r['name']:<20s} {r['params']:>8,} {r['test_f1']:>8.4f} {r['test_auc']:>8.4f} "
              f"{r['test_aupr']:>8.4f} {r['test_acc']:>8.4f} {r['val_best_f1']:>8.4f} {r['time_s']:>4.0f}s")

    # Save
    out_path = os.path.join(os.path.dirname(FEAT_CACHE), "phyto_512d_benchmark.csv")
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"\nResults saved: {out_path}")


if __name__ == "__main__":
    main()
