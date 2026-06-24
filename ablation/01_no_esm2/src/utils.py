"""
Shared utilities for dataset construction, model training, and inference.
(IUPred3 only — no ESM2 dependencies.)
"""

import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from src.features import compute_iupred_scores

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TOP_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
MAX_LEN: int = 1024
MIN_LEN: int = 50
VALID_AA_SET: set = set("ACDEFGHIKLMNPQRSTVWY")


# ============================================================================
# Sequence helpers
# ============================================================================


def truncate_for_model(seq: str, max_len: int = MAX_LEN) -> str:
    """Truncate a sequence, reserving tokens for ``<cls>`` and ``<eos>``."""
    return seq[: max_len - 2]


def clean_sequence_df(
    df: pd.DataFrame,
    seq_col: str,
    label: int = None,
    min_len: int = MIN_LEN,
    valid_aa: set = None,
) -> pd.DataFrame:
    if valid_aa is None:
        valid_aa = VALID_AA_SET
    df = df.dropna(subset=[seq_col]).copy()
    df[seq_col] = df[seq_col].astype(str).str.strip().str.upper()
    df = df[df[seq_col].apply(lambda s: set(s).issubset(valid_aa))]
    df = df[(df[seq_col] != "") & (df[seq_col].str.len() >= min_len)]
    df = df.drop_duplicates(subset=[seq_col])
    if label is not None:
        df["label"] = label
    return df


# ============================================================================
# IUPred3 per-residue scores
# ============================================================================


def get_iupred_scores(sequence: str) -> np.ndarray:
    """Compute per-residue IUPred3 disorder scores on the truncated sequence."""
    truncated = truncate_for_model(sequence)
    return compute_iupred_scores(truncated)


# ============================================================================
# Dataset
# ============================================================================


class LLPSDataset(Dataset):
    """PyTorch dataset for LLPS protein sequences.

    Reads a preprocessed CSV (``sequence`` and ``label`` columns).
    IUPred3 per-residue scores are computed on-the-fly in ``__getitem__``.
    """

    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path)
        self.sequences = self.df["sequence"].tolist()
        self.labels = self.df["label"].values.astype(np.float32)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        seq = self.sequences[idx]
        iupred_scores = get_iupred_scores(seq)
        return seq, iupred_scores, self.labels[idx]


def collate_fn(batch):
    """Collate a batch of ``(seq, iupred_scores, label)`` tuples."""
    seqs, iupred_scores, labels = zip(*batch)
    iupred_tensors = [torch.tensor(s, dtype=torch.float32) for s in iupred_scores]
    lengths = torch.tensor([t.size(0) for t in iupred_tensors], dtype=torch.long)
    iupred_padded = pad_sequence(iupred_tensors, batch_first=True, padding_value=0.0)
    labels = torch.tensor(labels, dtype=torch.float32)
    return seqs, iupred_padded, lengths, labels


# ============================================================================
# Metrics
# ============================================================================


def compute_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict:
    probs = torch.sigmoid(logits).detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    preds = (probs >= 0.5).astype(int)
    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, zero_division=0)
    rec = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)
    auc = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.5
    aupr = average_precision_score(labels, probs) if len(np.unique(labels)) > 1 else 0.5
    return {
        "acc": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auc": auc,
        "aupr": aupr,
    }
