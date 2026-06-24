"""
ESM2 + IUPred3 feature extraction for ML classifiers.

Converts per-residue representations (variable-length) into fixed-length
feature vectors via pooling, suitable for Random Forest / XGBoost.
"""
import os
import sys
import numpy as np
from tqdm import tqdm
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from src.utils import ESM2Extractor, get_iupred_scores


def extract_iupred_stats(seqs: list) -> np.ndarray:
    """Compute per-sequence IUPred3 statistics.

    Returns (N, 9) array with columns:
      mean, std, min, max, median, Q25, Q75, frac_disordered, length_normalized
    """
    feats = []
    for seq in tqdm(seqs, desc="IUPred3 stats"):
        scores = get_iupred_scores(seq)
        L = max(len(scores), 1)

        f_mean = np.mean(scores)
        f_std = np.std(scores)
        f_min = np.min(scores) if len(scores) > 0 else 0.0
        f_max = np.max(scores) if len(scores) > 0 else 0.0
        f_median = np.median(scores) if len(scores) > 0 else 0.0
        f_q25 = np.percentile(scores, 25) if len(scores) > 0 else 0.0
        f_q75 = np.percentile(scores, 75) if len(scores) > 0 else 0.0
        f_frac = (scores > 0.5).mean() if len(scores) > 0 else 0.0
        f_len_norm = L / 1024.0

        feats.append([f_mean, f_std, f_min, f_max, f_median,
                      f_q25, f_q75, f_frac, f_len_norm])
    return np.array(feats, dtype=np.float32)


def extract_esm2_meanpool(seqs: list, esm2_dir: str, device: str = "cuda",
                          batch_size: int = 16) -> np.ndarray:
    """Extract ESM2 per-residue embeddings and mean-pool to (N, 1280).

    Returns (N, 1280) float32 array.
    """
    extractor = ESM2Extractor(esm2_dir, device)
    all_embeds = []
    n_batches = (len(seqs) + batch_size - 1) // batch_size

    for i in tqdm(range(0, len(seqs), batch_size), desc="ESM2 mean-pool",
                  total=n_batches):
        batch_seqs = seqs[i: i + batch_size]
        residue_list, _ = extractor.extract_residue_batch(batch_seqs, batch_size=batch_size)
        for res in residue_list:
            if res.shape[0] > 0:
                all_embeds.append(res.mean(axis=0))
            else:
                all_embeds.append(np.zeros(1280, dtype=np.float32))

    return np.array(all_embeds, dtype=np.float32)


def extract_esm2_cls_batch(seqs: list, esm2_dir: str, device: str = "cuda",
                           batch_size: int = 64) -> np.ndarray:
    """Extract ESM2 CLS token embeddings directly (much faster than per-residue).

    Uses AMP for mixed-precision inference.

    Returns (N, 1280) float32 array.
    """
    from transformers import AutoTokenizer, AutoModel

    tokenizer = AutoTokenizer.from_pretrained(esm2_dir, local_files_only=True)
    model = AutoModel.from_pretrained(
        esm2_dir, local_files_only=True, dtype=torch.float16
    ).to(device)
    model.eval()

    all_feats = []
    n_batches = (len(seqs) + batch_size - 1) // batch_size

    for i in tqdm(range(0, len(seqs), batch_size), desc="ESM2 CLS",
                  total=n_batches):
        batch_seqs = seqs[i: i + batch_size]
        inputs = tokenizer(
            list(batch_seqs),
            truncation=True,
            max_length=1024,
            padding=True,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=(device != "cpu")):
            outputs = model(**inputs)
        all_feats.append(outputs.last_hidden_state[:, 0, :].float().cpu().numpy())

    return np.concatenate(all_feats, axis=0).astype(np.float32)


def extract_features_esm_iupred(seqs: list, esm2_dir: str,
                                device: str = "cuda",
                                batch_size: int = 64) -> np.ndarray:
    """Extract combined ESM2 + IUPred3 features for ML models.

    Features (1289-dim):
      - ESM2 CLS token: 1280
      - IUPred3 statistics: 9 (mean, std, min, max, median, Q25, Q75, frac, len)

    Returns (N, 1289) float32 array.
    """
    print(f"Extracting features for {len(seqs)} sequences ...")

    esm2_feats = extract_esm2_cls_batch(seqs, esm2_dir, device, batch_size)
    iupred_feats = extract_iupred_stats(seqs)

    combined = np.concatenate([esm2_feats, iupred_feats], axis=1)
    print(f"Feature shape: {combined.shape}")
    return combined


def extract_features_all_splits(train_seqs, val_seqs, test_seqs,
                                esm2_dir, device="cuda", batch_size=64):
    """Extract ESM2+IUPred3 features for all splits in one ESM2 model load.

    Returns (X_train, X_val, X_test) as float32 arrays.
    """
    from transformers import AutoTokenizer, AutoModel

    print(f"Loading ESM2 model from {esm2_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(esm2_dir, local_files_only=True)
    model = AutoModel.from_pretrained(
        esm2_dir, local_files_only=True, dtype=torch.float16
    ).to(device)
    model.eval()

    use_amp = device != "cpu"

    def extract_esm2_batch(seqs, desc):
        all_feats = []
        n_batches = (len(seqs) + batch_size - 1) // batch_size
        for i in tqdm(range(0, len(seqs), batch_size), desc=desc, total=n_batches):
            batch_seqs = seqs[i: i + batch_size]
            inputs = tokenizer(
                list(batch_seqs), truncation=True, max_length=1024,
                padding=True, return_tensors="pt",
            ).to(device)
            with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(**inputs)
            all_feats.append(outputs.last_hidden_state[:, 0, :].float().cpu().numpy())
        return np.concatenate(all_feats, axis=0).astype(np.float32)

    print(f"Extracting ESM2 CLS for train ({len(train_seqs)} seqs)...")
    esm2_train = extract_esm2_batch(train_seqs, "ESM2 train")
    print(f"Extracting ESM2 CLS for val ({len(val_seqs)} seqs)...")
    esm2_val = extract_esm2_batch(val_seqs, "ESM2 val")
    print(f"Extracting ESM2 CLS for test ({len(test_seqs)} seqs)...")
    esm2_test = extract_esm2_batch(test_seqs, "ESM2 test")

    print("Extracting IUPred3 stats...")
    iupred_train = extract_iupred_stats(train_seqs)
    iupred_val = extract_iupred_stats(val_seqs)
    iupred_test = extract_iupred_stats(test_seqs)

    X_train = np.concatenate([esm2_train, iupred_train], axis=1)
    X_val = np.concatenate([esm2_val, iupred_val], axis=1)
    X_test = np.concatenate([esm2_test, iupred_test], axis=1)

    return X_train, X_val, X_test
