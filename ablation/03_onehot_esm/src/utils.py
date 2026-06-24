"""
Shared utilities for dataset construction, model training, and inference.
(One-hot + IUPred3 — no ESM2 dependencies.)
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

from src.features import compute_iupred_scores, compute_onehot_encoding

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TOP_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
MAX_LEN: int = 1024
MIN_LEN: int = 50
VALID_AA_SET: set = set("ACDEFGHIKLMNPQRSTVWY")


# ============================================================================
# Sequence helpers
# ============================================================================


def truncate_for_model(seq: str, max_len: int = MAX_LEN) -> str:
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
    truncated = truncate_for_model(sequence)
    return compute_iupred_scores(truncated)


# ============================================================================
# One-hot extractor
# ============================================================================


class OneHotExtractor:
    """One-hot residue-level feature extractor (CPU, no GPU needed)."""

    def extract_residue_batch_tensors(
        self, seqs, pad_to_length: int = None
    ):
        batch_onehots = []
        lengths_list = []
        for seq in seqs:
            truncated = truncate_for_model(seq)
            onehot = compute_onehot_encoding(truncated)
            batch_onehots.append(torch.tensor(onehot, dtype=torch.float32))
            lengths_list.append(len(truncated))

        lengths = torch.tensor(lengths_list, dtype=torch.long)
        seq_padded = pad_sequence(batch_onehots, batch_first=True, padding_value=0.0)

        if pad_to_length is not None:
            curr_len = seq_padded.size(1)
            if curr_len < pad_to_length:
                B, L, D = seq_padded.size()
                pad = torch.zeros(B, pad_to_length - L, D, dtype=seq_padded.dtype)
                seq_padded = torch.cat([seq_padded, pad], dim=1)
            elif curr_len > pad_to_length:
                seq_padded = seq_padded[:, :pad_to_length, :]
                lengths = lengths.clamp(max=pad_to_length)

        return seq_padded, lengths


# ============================================================================
# Dataset
# ============================================================================


class LLPSDataset(Dataset):
    """PyTorch dataset for LLPS protein sequences.

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
