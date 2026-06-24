"""
Control experiment: train the full ResidueLLPSClassifier (same architecture
as the main LLPS model) on RNA-dep data for comparison purposes.

Expected: low F1 due to 24M params on only ~2450 training samples.
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, \
    accuracy_score, f1_score, roc_auc_score, average_precision_score
from tqdm import tqdm

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, HERE)
from src.model import ResidueLLPSClassifier
from src.utils import PROJECT_ROOT, ESM2Extractor, LLPSDataset, collate_fn, compute_metrics

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "rna_dep")
ESM2_DIR = os.path.join(PROJECT_ROOT, "esm2")
MODEL_DIR = os.path.join(PROJECT_ROOT, "outputs", "models")
LOG_DIR = os.path.join(PROJECT_ROOT, "outputs", "logs")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

BATCH_SIZE = 32
LR = 2e-5
WEIGHT_DECAY = 1e-4
TOTAL_STEPS = 3000
EVAL_INTERVAL = 100
PATIENCE = 7
GRAD_CLIP = 1.0


def main():
    print("=" * 60, flush=True)
    print("Control: Full ResidueLLPSClassifier on RNA-dep (24M params)", flush=True)
    print(f"Device: {DEVICE} | Steps: {TOTAL_STEPS}", flush=True)
    print("=" * 60, flush=True)

    esm2 = ESM2Extractor(ESM2_DIR, DEVICE)
    train_set = LLPSDataset(os.path.join(DATA_DIR, "train.csv"))
    val_set = LLPSDataset(os.path.join(DATA_DIR, "val.csv"))
    test_set = LLPSDataset(os.path.join(DATA_DIR, "test.csv"))
    print(f"Data: Train={len(train_set)} Val={len(val_set)} Test={len(test_set)}")

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_fn, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False,
                            collate_fn=collate_fn, num_workers=0, pin_memory=False)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False,
                             collate_fn=collate_fn, num_workers=0, pin_memory=False)

    model = ResidueLLPSClassifier(d_model=1280, hidden=512).to(DEVICE)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    best_path = os.path.join(MODEL_DIR, "best_rna_control_model.pth")
    start_step = 1
    if os.path.exists(best_path):
        ckp = torch.load(best_path, map_location=DEVICE)
        model.load_state_dict(ckp)
        start_step = 201
        print(f"Resumed from checkpoint, continuing from step {start_step}", flush=True)

    criterion = nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=TOTAL_STEPS)
    for _ in range(start_step - 1):
        sched.step()
    scaler = GradScaler("cuda")

    log_file = os.path.join(LOG_DIR,
        datetime.now().strftime("train_rna_dep_control_%Y%m%d_%H%M%S.txt"))
    with open(log_file, "w") as f:
        f.write("Step\tPhase\tLoss\tAcc\tPrecision\tRecall\tF1\tAUC\tAUPR\tLR\n")

    def eval_model(loader):
        model.eval()
        total_loss, all_logits, all_labels = 0.0, [], []
        with torch.no_grad():
            for seqs, iup, lengths, labels in tqdm(loader, desc="Eval", leave=False):
                esm_embeds, _ = esm2.extract_residue_batch_tensors(seqs, pad_to_length=iup.size(1))
                esm_embeds, iup = esm_embeds.to(DEVICE), iup.to(DEVICE)
                lengths, labels = lengths.to(DEVICE), labels.to(DEVICE)
                logits = model(esm_embeds, iup, lengths)
                loss = criterion(logits, labels)
                total_loss += loss.item()
                all_logits.append(logits.cpu()); all_labels.append(labels.cpu())
        all_logits = torch.cat(all_logits); all_labels = torch.cat(all_labels)
        m = compute_metrics(all_logits, all_labels)
        m["loss"] = total_loss / len(loader)
        return m

    best_f1, no_improve = 0.0, 0
    train_iter = iter(train_loader)

    for step in range(start_step, TOTAL_STEPS + 1):
        try:
            seqs, iup, lengths, labels = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            seqs, iup, lengths, labels = next(train_iter)
        esm_embeds, _ = esm2.extract_residue_batch_tensors(seqs, pad_to_length=iup.size(1))
        esm_embeds, iup = esm_embeds.to(DEVICE), iup.to(DEVICE)
        lengths, labels = lengths.to(DEVICE), labels.to(DEVICE)

        model.train(); opt.zero_grad()
        with autocast("cuda"):
            loss = criterion(model(esm_embeds, iup, lengths), labels)
        scaler.scale(loss).backward(); scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(opt); scaler.update(); sched.step()

        if step % EVAL_INTERVAL == 0:
            vm = eval_model(val_loader)
            lr = sched.get_last_lr()[0]
            print(f"Step {step:4d} | LR:{lr:.6f} Val F1:{vm['f1']:.4f} AUC:{vm['auc']:.4f}",
                  flush=True)
            with open(log_file, "a") as f:
                f.write(f"{step}\tval\t{vm['loss']:.6f}\t{vm['acc']:.6f}\t"
                        f"{vm['precision']:.6f}\t{vm['recall']:.6f}\t"
                        f"{vm['f1']:.6f}\t{vm['auc']:.6f}\t{vm['aupr']:.6f}\t{lr:.8f}\n")
            if vm["f1"] > best_f1:
                best_f1 = vm["f1"]; torch.save(model.state_dict(), best_path)
                print(f"  Best (Val F1: {best_f1:.4f})", flush=True); no_improve = 0
            else:
                no_improve += 1
                if no_improve >= PATIENCE:
                    print(f"  Early stop at step {step}", flush=True); break

    print("\n" + "=" * 60 + "\nTest\n" + "=" * 60, flush=True)
    model.load_state_dict(torch.load(best_path, map_location=DEVICE))
    tm = eval_model(test_loader)
    print(f"Test Acc:{tm['acc']:.4f} F1:{tm['f1']:.4f} AUC:{tm['auc']:.4f} "
          f"AUPR:{tm['aupr']:.4f}", flush=True)


if __name__ == "__main__":
    main()
