#!/usr/bin/env python3
"""
Benchmark PhytoRNP on catGRANULE2.0's published test set.

1. Fetch sequences from UniProt for the 2,799 test set proteins
2. Run PhytoRNP inference
3. Compute and print metrics side-by-side with catGRANULE2.0 results
"""
import os
import sys
import json
import time
import urllib.request
import urllib.error
import pickle
import numpy as np
import pandas as pd
import torch
from torch.amp import autocast
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.model import ResidueLLPSClassifier
from src.utils import ESM2Extractor, get_iupred_scores, clean_sequence_df
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    precision_score, recall_score, f1_score, matthews_corrcoef,
)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CAT_BASE = os.path.join(os.path.dirname(PROJECT_ROOT), "catGRANULE2.0")
ESM2_DIR = os.path.join(PROJECT_ROOT, "esm2")
MODEL_PATH = os.path.join(PROJECT_ROOT, "outputs", "models", "best_model.pth")
CACHE_PATH = os.path.join(PROJECT_ROOT, "outputs", "benchmarks", "cat_test_sequences.pkl")
os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)

BATCH_SIZE = 32
MAX_LEN = 1024


def fetch_uniprot_batch(ids_batch: list) -> dict:
    """Fetch sequences from UniProt REST API (GET, accessions endpoint)."""
    url = "https://rest.uniprot.org/uniprotkb/accessions"
    params = urllib.parse.urlencode({
        "accessions": ",".join(ids_batch),
        "format": "fasta",
        "size": str(len(ids_batch)),
    })
    full_url = f"{url}?{params}"
    req = urllib.request.Request(full_url)
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            text = resp.read().decode()
    except urllib.error.HTTPError as e:
        print(f"  HTTP error {e.code} for batch: {ids_batch[:3]}...")
        return {}
    # Parse FASTA
    seqs = {}
    current_id = None
    current_seq = []
    for line in text.strip().split("\n"):
        if line.startswith(">"):
            if current_id:
                seqs[current_id] = "".join(current_seq)
            # Extract accession from header: >sp|Q8N104|...
            parts = line[1:].strip().split("|")
            if len(parts) >= 2:
                current_id = parts[1]
            else:
                current_id = line[1:].split()[0]
            current_seq = []
        else:
            current_seq.append(line.strip())
    if current_id:
        seqs[current_id] = "".join(current_seq)
    return seqs


def get_sequences(all_ids: list) -> dict:
    """Get sequences with caching and rate limiting."""
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "rb") as f:
            cache = pickle.load(f)
        missing = [uid for uid in all_ids if uid not in cache]
        print(f"Cache hit: {len(cache)}, missing: {len(missing)}")
    else:
        cache = {}
        missing = all_ids

    if missing:
        batch_size_api = 100
        for i in tqdm(range(0, len(missing), batch_size_api), desc="Fetching sequences"):
            batch = missing[i:i + batch_size_api]
            seqs = fetch_uniprot_batch(batch)
            cache.update(seqs)
            time.sleep(0.3)  # Rate limit: ~3 req/sec
            if i % 500 == 0 and i > 0:
                with open(CACHE_PATH, "wb") as f:
                    pickle.dump(cache, f)
        with open(CACHE_PATH, "wb") as f:
            pickle.dump(cache, f)

    return cache


def predict_batch(model, extractor, seqs, device):
    """Run inference on a batch of sequences."""
    batch_iupred = [get_iupred_scores(seq) for seq in seqs]
    iupred_tensors = [torch.tensor(s, dtype=torch.float32) for s in batch_iupred]
    lengths = torch.tensor([t.size(0) for t in iupred_tensors], dtype=torch.long)
    iupred_padded = pad_sequence(iupred_tensors, batch_first=True, padding_value=0.0)

    esm_embeds, _ = extractor.extract_residue_batch_tensors(
        seqs, pad_to_length=iupred_padded.size(1)
    )
    esm_embeds = esm_embeds.to(device)
    iupred_padded = iupred_padded.to(device)
    lengths = lengths.to(device)

    with torch.no_grad():
        with autocast("cuda"):
            logits = model(esm_embeds, iupred_padded, lengths)
        probs = torch.sigmoid(logits).cpu().numpy()
    return probs


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Load catGRANULE2.0 test set (Uniprot IDs + labels)
    print("\n[1/5] Loading catGRANULE2.0 test set...")
    test_ids = pd.read_csv(
        os.path.join(CAT_BASE, "DATASETS", "TestSet_IDs.csv"), index_col=0
    )
    all_ids = test_ids.index.tolist()
    labels = test_ids["labels"]
    print(f"  Test set: {len(all_ids)} proteins ({int(labels.sum())} pos, {int((labels==0).sum())} neg)")

    # 2. Fetch sequences from UniProt
    print("\n[2/5] Fetching sequences from UniProt...")
    seq_dict = get_sequences(all_ids)
    found_ids = [uid for uid in all_ids if uid in seq_dict and len(seq_dict[uid]) > 0]
    not_found = set(all_ids) - set(seq_dict.keys())
    empty = [uid for uid in found_ids if len(seq_dict[uid]) == 0]
    print(f"  Found: {len(found_ids)}, Not found in UniProt: {len(not_found)}, Empty seq: {len(empty)}")

    # Filter to found sequences
    valid_ids = [uid for uid in found_ids if len(seq_dict[uid]) >= 10]
    print(f"  Valid (len>=10): {len(valid_ids)}")
    valid_labels = test_ids.loc[valid_ids, "labels"]
    sequences = [seq_dict[uid] for uid in valid_ids]

    # 3. Load PhytoRNP model
    print("\n[3/5] Loading PhytoRNP model...")
    extractor = ESM2Extractor(ESM2_DIR, device)
    model = ResidueLLPSClassifier(d_model=1280, hidden=512).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print("  Model loaded.")

    # 4. Run prediction with checkpointing
    print(f"\n[4/5] Running PhytoRNP prediction (batch_size={BATCH_SIZE})...")
    all_probs = []
    checkpoint_path = os.path.join(PROJECT_ROOT, "outputs", "benchmarks", "checkpoint.npz")
    start_batch = 0

    if os.path.exists(checkpoint_path):
        ckpt = np.load(checkpoint_path, allow_pickle=True)
        all_probs = list(ckpt["probs"])
        start_batch = int(ckpt["batch"])
        print(f"  Resuming from checkpoint: {start_batch} batches done, {len(all_probs)} proteins predicted")

    total_batches = (len(sequences) + BATCH_SIZE - 1) // BATCH_SIZE
    tbar = tqdm(range(start_batch * BATCH_SIZE, len(sequences), BATCH_SIZE),
                desc="Predicting", initial=start_batch, total=total_batches)
    try:
        for i in tbar:
            batch_seqs = sequences[i:i + BATCH_SIZE]
            try:
                probs = predict_batch(model, extractor, batch_seqs, device)
                all_probs.extend(probs)
            except Exception as batch_err:
                tbar.write(f"\n  ERROR batch {i//BATCH_SIZE}: {batch_err}")
                torch.cuda.empty_cache()
                for j, seq in enumerate(batch_seqs):
                    try:
                        p = predict_batch(model, extractor, [seq], device)
                        all_probs.extend(p)
                    except Exception as seq_err:
                        tbar.write(f"  Skip seq {i+j}: {seq_err}")
                        all_probs.append(np.nan)
            batch_num = i // BATCH_SIZE + 1
            if batch_num % 5 == 0:
                np.savez(checkpoint_path, probs=np.array(all_probs), batch=batch_num)
    except Exception as e:
        print(f"\nFATAL ERROR at step {len(all_probs)}: {e}", flush=True)
        import traceback
        traceback.print_exc()
        np.savez(checkpoint_path, probs=np.array(all_probs), batch=len(all_probs)//BATCH_SIZE)

    all_probs = np.array(all_probs, dtype=np.float32)

    # 5. Compute metrics
    print("\n[5/5] Computing metrics...")
    # Filter out NaN predictions
    valid_mask = ~np.isnan(all_probs)
    all_probs_clean = all_probs[valid_mask]
    y_true = valid_labels.values[valid_mask]
    y_pred = (all_probs_clean >= 0.5).astype(int)
    valid_ids_clean = [valid_ids[i] for i in range(len(valid_ids)) if valid_mask[i]]
    print(f"  Valid predictions: {len(y_true)} / {len(all_probs)} ({(~valid_mask).sum()} NaN skipped)")

    auc = roc_auc_score(y_true, all_probs)
    aupr = average_precision_score(y_true, all_probs)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)

    # Print comparison table
    print("\n" + "=" * 90)
    print(f"PhytoRNP vs catGRANULE2.0 on catGRANULE2.0 test set (n={len(y_true)})")
    print(f"Pos: {int(y_true.sum())}, Neg: {int((y_true==0).sum())}")
    print("=" * 90)
    print(f'{"Model":<28s} {"AUC":>8s} {"AUPR":>8s} {"Acc":>8s} {"Prec":>8s} {"Rec":>8s} {"F1":>8s} {"MCC":>8s}')
    print("-" * 90)

    cat_models = [
        ("catGRANULE v1",             0.6605, 0.6132, 0.6080, 0.6266, 0.5675, 0.5956, 0.2181),
        ("catG2 RF (physchem)",       0.7349, 0.7193, 0.6640, 0.7002, 0.5928, 0.6420, 0.3337),
        ("catG2 MLP (physchem)",      0.7363, 0.7081, 0.6665, 0.7158, 0.5703, 0.6348, 0.3425),
        ("catG2 MLP (all feat)",      0.7646, 0.7375, 0.6883, 0.7371, 0.6013, 0.6623, 0.3852),
        ("PICNIC",                    0.7346, 0.7210, 0.6609, 0.7380, 0.5169, 0.6079, 0.3413),
        ("PICNICGO",                  0.7547, 0.7788, 0.6373, 0.8091, 0.4212, 0.5540, 0.3415),
    ]
    for name, a_auc, a_aupr, a_acc, a_prec, a_rec, a_f1, a_mcc in cat_models:
        print(f'{name:<28s} {a_auc:8.4f} {a_aupr:8.4f} {a_acc:8.4f} {a_prec:8.4f} {a_rec:8.4f} {a_f1:8.4f} {a_mcc:8.4f}')
    print("-" * 90)
    print(f'{"PhytoRNP (我们的)":<28s} {auc:8.4f} {aupr:8.4f} {acc:8.4f} {prec:8.4f} {rec:8.4f} {f1:8.4f} {mcc:8.4f}')
    print("=" * 90)

    # Save results
    out_dir = os.path.join(PROJECT_ROOT, "outputs", "benchmarks")
    out_csv = os.path.join(out_dir, "catGRANULE_testset_phytoRNP_predictions.csv")
    results_df = pd.DataFrame({
        "UniprotID": valid_ids_clean,
        "true_label": y_true.astype(int),
        "PhytoRNP_score": all_probs_clean,
        "PhytoRNP_pred": y_pred,
    })
    results_df.to_csv(out_csv, index=False)
    print(f"\nFull predictions saved to: {out_csv}")


if __name__ == "__main__":
    main()
