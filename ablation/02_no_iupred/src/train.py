"""
Train the residue-level LLPS classifier (IUPred3 zeroed out ablation).
Model architecture unchanged — IUPred3 input is set to zero.
"""

import os
import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from src.model import ResidueLLPSClassifier  # noqa: E402
from src.utils import (  # noqa: E402
    PROJECT_ROOT,
    TOP_PROJECT_ROOT,
    ESM2Extractor,
    LLPSDataset,
    collate_fn,
    compute_metrics,
)


# ============================================================================
# Configuration
# ============================================================================


class Config:
    def __init__(self):
        self.data_dir = os.path.join(TOP_PROJECT_ROOT, "data", "processed")
        self.model_dir = os.path.join(PROJECT_ROOT, "outputs", "models")
        self.log_dir = os.path.join(PROJECT_ROOT, "outputs", "logs")
        self.esm2_dir = os.path.join(TOP_PROJECT_ROOT, "esm2")
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        self.batch_size = 32
        self.d_model = 1280
        self.hidden = 512
        self.lr = 2e-5
        self.weight_decay = 1e-4
        self.total_steps = 10000
        self.eval_interval = 300
        self.patience_evals = 5
        self.grad_clip = 1.0


# ============================================================================
# GPU monitoring
# ============================================================================


def log_gpu_stats():
    try:
        import subprocess
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        lines = result.stdout.strip().split("\n")
        print("\n[GPU Status]")
        print("  idx | name                 | mem used / total  | util%")
        print("  ----+----------------------+-------------------+------")
        for line in lines:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 5:
                idx, name, mem_used, mem_total, util = parts[:5]
                print(f"  {idx:<3} | {name:<20} | {mem_used:>5} / {mem_total:<5} MB | {util}%")
        print()
    except Exception as e:
        print(f"[GPU Monitor] Warning: {e}")


def log_model_gpu_usage(model: nn.Module, label: str = "model"):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    mem_mb = total_params * 4 / (1024 ** 2)
    print(f"[{label}] Params: {total_params:,} ({trainable_params:,} trainable) | ~{mem_mb:.1f} MB")


# ============================================================================
# Training utilities
# ============================================================================


def train_step(
    model, esm_embeds, iupred_scores, lengths, labels,
    criterion, optimizer, scaler, grad_clip,
):
    model.train()
    optimizer.zero_grad()
    with autocast("cuda"):
        logits = model(esm_embeds, iupred_scores, lengths)
        loss = criterion(logits, labels)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(optimizer)
    scaler.update()
    return loss.item(), logits.detach().cpu(), labels.detach().cpu()


def validate(model, loader, criterion, device, extractor, return_raw=False):
    model.eval()
    total_loss = 0.0
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for seqs, iupred_scores, lengths, labels in tqdm(loader, desc="Evaluating"):
            esm_embeds, _ = extractor.extract_residue_batch_tensors(
                seqs, pad_to_length=iupred_scores.size(1)
            )
            esm_embeds = esm_embeds.to(device)
            iupred_scores = iupred_scores.to(device)
            lengths = lengths.to(device)
            labels = labels.to(device)

            with autocast("cuda"):
                logits = model(esm_embeds, iupred_scores, lengths)
                loss = criterion(logits, labels)

            total_loss += loss.item()
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    metrics = compute_metrics(all_logits, all_labels)
    metrics["loss"] = total_loss / len(loader)

    if return_raw:
        return metrics, all_logits, all_labels
    return metrics


# ============================================================================
# Main
# ============================================================================


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg = Config()

    print("=" * 60)
    print("LLPS Model Training (Ablation: IUPred3 zeroed out)")
    print(f"Device: {device}")
    print(f"Total steps: {cfg.total_steps} | Eval every: {cfg.eval_interval}")
    print(f"Early stop patience: {cfg.patience_evals} evals")
    print("=" * 60)
    log_gpu_stats()

    esm2_extractor = ESM2Extractor(cfg.esm2_dir, device)

    train_set = LLPSDataset(os.path.join(cfg.data_dir, "train.csv"))
    val_set = LLPSDataset(os.path.join(cfg.data_dir, "val.csv"))
    test_set = LLPSDataset(os.path.join(cfg.data_dir, "test.csv"))

    n_pos = int(train_set.labels.sum())
    n_neg = len(train_set.labels) - n_pos
    print(
        f"Train: {len(train_set)} (pos={n_pos}, neg={n_neg}, "
        f"ratio={n_pos / len(train_set):.4f}) | "
        f"Val: {len(val_set)} | Test: {len(test_set)}"
    )

    train_loader = DataLoader(
        train_set, batch_size=cfg.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0, pin_memory=False,
    )
    val_loader = DataLoader(
        val_set, batch_size=cfg.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0, pin_memory=False,
    )
    test_loader = DataLoader(
        test_set, batch_size=cfg.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0, pin_memory=False,
    )

    model = ResidueLLPSClassifier(d_model=cfg.d_model, hidden=cfg.hidden).to(device)
    log_model_gpu_usage(model, label="Classifier")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = GradScaler("cuda")

    best_f1 = 0.0
    no_improve_count = 0
    best_path = os.path.join(cfg.model_dir, "best_model.pth")

    log_name = datetime.now().strftime("train_%Y%m%d_%H%M%S.txt")
    log_file = os.path.join(cfg.log_dir, log_name)
    with open(log_file, "w") as f:
        f.write("Step\tPhase\tLoss\tAcc\tPrecision\tRecall\tF1\tAUC\tAUPR\tLR\n")
    print(f"\n[Log] Metrics saved to: {log_file}\n")

    def write_log(step, phase, metrics, lr):
        with open(log_file, "a") as f:
            f.write(
                f"{step}\t{phase}\t{metrics['loss']:.6f}\t{metrics['acc']:.6f}\t"
                f"{metrics['precision']:.6f}\t{metrics['recall']:.6f}\t"
                f"{metrics['f1']:.6f}\t{metrics['auc']:.6f}\t{metrics['aupr']:.6f}\t{lr:.8f}\n"
            )

    train_iter = iter(train_loader)
    train_logits_buffer, train_labels_buffer, train_loss_buffer = [], [], []
    current_lr = cfg.lr

    for step in range(1, cfg.total_steps + 1):
        try:
            seqs, iupred_scores, lengths, labels = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            seqs, iupred_scores, lengths, labels = next(train_iter)

        esm_embeds, _ = esm2_extractor.extract_residue_batch_tensors(
            seqs, pad_to_length=iupred_scores.size(1)
        )
        esm_embeds = esm_embeds.to(device)
        iupred_scores = iupred_scores.to(device)
        lengths = lengths.to(device)
        labels = labels.to(device)

        loss_item, logits_cpu, labels_cpu = train_step(
            model, esm_embeds, iupred_scores, lengths, labels,
            criterion, optimizer, scaler, cfg.grad_clip,
        )

        train_loss_buffer.append(loss_item)
        train_logits_buffer.append(logits_cpu)
        train_labels_buffer.append(labels_cpu)

        if step % cfg.eval_interval == 0:
            train_logits = torch.cat(train_logits_buffer)
            train_labels = torch.cat(train_labels_buffer)
            train_metrics = compute_metrics(train_logits, train_labels)
            train_metrics["loss"] = sum(train_loss_buffer) / len(train_loss_buffer)
            train_logits_buffer.clear()
            train_labels_buffer.clear()
            train_loss_buffer.clear()

            val_metrics = validate(model, val_loader, criterion, device, esm2_extractor)

            print(f"\nStep {step}/{cfg.total_steps} | LR: {current_lr:.6f}")
            print(
                f"Train -> Loss: {train_metrics['loss']:.4f} | "
                f"Acc: {train_metrics['acc']:.4f} | F1: {train_metrics['f1']:.4f} | "
                f"AUC: {train_metrics['auc']:.4f}"
            )
            print(
                f"Val   -> Loss: {val_metrics['loss']:.4f} | "
                f"Acc: {val_metrics['acc']:.4f} | F1: {val_metrics['f1']:.4f} | "
                f"AUC: {val_metrics['auc']:.4f} | AUPR: {val_metrics['aupr']:.4f}"
            )
            write_log(step, "train", train_metrics, current_lr)
            write_log(step, "val", val_metrics, current_lr)

            if val_metrics["f1"] > best_f1:
                best_f1 = val_metrics["f1"]
                torch.save(model.state_dict(), best_path)
                print(f"Best model saved (Val F1: {best_f1:.4f})")
                no_improve_count = 0
            else:
                no_improve_count += 1
                print(f"No F1 improvement ({no_improve_count}/{cfg.patience_evals})")
                if no_improve_count >= cfg.patience_evals:
                    print(f"Early stopping at step {step}.")
                    break

    # Final test evaluation
    print("\n" + "=" * 60)
    print("Final Test Evaluation")
    print("=" * 60)
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))
    else:
        print(f"Warning: {best_path} not found, saving current model as fallback")
        torch.save(model.state_dict(), best_path)

    test_metrics, test_logits, test_labels_t = validate(
        model, test_loader, criterion, device, esm2_extractor, return_raw=True
    )
    print(
        f"Test -> Loss: {test_metrics['loss']:.4f} | "
        f"Acc: {test_metrics['acc']:.4f} | Precision: {test_metrics['precision']:.4f} | "
        f"Recall: {test_metrics['recall']:.4f}"
    )
    print(
        f"        F1: {test_metrics['f1']:.4f} | AUC: {test_metrics['auc']:.4f} | "
        f"AUPR: {test_metrics['aupr']:.4f}"
    )
    write_log(step, "test", test_metrics, current_lr)

    probs = torch.sigmoid(test_logits).numpy()
    preds = (probs >= 0.5).astype(int)
    labels_np = test_labels_t.numpy()

    print("\nConfusion Matrix:")
    print(confusion_matrix(labels_np, preds))
    print("\nClassification Report:")
    print(classification_report(labels_np, preds, digits=4))


if __name__ == "__main__":
    main()
