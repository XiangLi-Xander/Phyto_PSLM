"""
Shared utilities (IUPred3 zeroed out ablation — no IUPred3 computation).
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
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

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
# ESM2 feature extractor
# ============================================================================


class ESM2Extractor:
    """ESM2 residue-level feature extractor (batch, GPU)."""

    def __init__(self, model_dir: str, device: str = "cuda"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        self.model = AutoModel.from_pretrained(model_dir, local_files_only=True).to(device)
        self.model.eval()

    def extract_cls_batch_tensors(self, seqs, max_len: int = MAX_LEN) -> torch.Tensor:
        inputs = self.tokenizer(
            list(seqs),
            truncation=True,
            max_length=max_len,
            padding=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state[:, 0, :]

    def extract_cls_batch(
        self, sequences, max_len: int = MAX_LEN, batch_size: int = 16
    ) -> np.ndarray:
        all_feats = []
        n_batches = (len(sequences) + batch_size - 1) // batch_size
        for i in tqdm(
            range(0, len(sequences), batch_size),
            desc="ESM2 cls extracting",
            total=n_batches,
        ):
            batch_seqs = sequences[i : i + batch_size]
            inputs = self.tokenizer(
                batch_seqs,
                truncation=True,
                max_length=max_len,
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            cls_feats = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_feats.append(cls_feats)
        return np.concatenate(all_feats, axis=0)

    def extract_residue_batch_tensors(
        self, seqs, max_len: int = MAX_LEN, pad_to_length: int = None
    ):
        inputs = self.tokenizer(
            list(seqs),
            truncation=True,
            max_length=max_len,
            padding=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)

        hidden = outputs.last_hidden_state
        attention_mask = inputs["attention_mask"]

        lengths = attention_mask.sum(dim=1) - 2
        lengths = lengths.clamp(min=0)

        valid_embeds = []
        for i in range(hidden.size(0)):
            L = lengths[i].item()
            if L > 0:
                valid_embeds.append(hidden[i, 1 : 1 + L])
            else:
                valid_embeds.append(
                    torch.zeros(0, hidden.size(-1), device=hidden.device, dtype=hidden.dtype)
                )

        residue_embeds = pad_sequence(valid_embeds, batch_first=True, padding_value=0.0)

        if pad_to_length is not None:
            curr_len = residue_embeds.size(1)
            if curr_len < pad_to_length:
                B, L, D = residue_embeds.size()
                pad = torch.zeros(
                    B, pad_to_length - L, D,
                    device=residue_embeds.device,
                    dtype=residue_embeds.dtype,
                )
                residue_embeds = torch.cat([residue_embeds, pad], dim=1)
            elif curr_len > pad_to_length:
                residue_embeds = residue_embeds[:, :pad_to_length, :]
                lengths = lengths.clamp(max=pad_to_length)

        return residue_embeds, lengths

    def extract_residue_batch(self, sequences, max_len: int = MAX_LEN, batch_size: int = 16):
        all_embeds = []
        all_lengths = []
        n_batches = (len(sequences) + batch_size - 1) // batch_size
        for i in tqdm(
            range(0, len(sequences), batch_size),
            desc="ESM2 residue extracting",
            total=n_batches,
        ):
            batch_seqs = sequences[i : i + batch_size]
            inputs = self.tokenizer(
                batch_seqs,
                truncation=True,
                max_length=max_len,
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)

            hidden = outputs.last_hidden_state
            attention_mask = inputs["attention_mask"]
            lengths = (attention_mask.sum(dim=1) - 2).clamp(min=0)

            for j in range(hidden.size(0)):
                L = lengths[j].item()
                if L > 0:
                    all_embeds.append(hidden[j, 1 : 1 + L].cpu().numpy())
                else:
                    all_embeds.append(np.zeros((0, hidden.size(-1)), dtype=np.float32))
                all_lengths.append(L)

        return all_embeds, all_lengths


# ============================================================================
# Dataset (returns zero IUPred scores)
# ============================================================================


class LLPSDataset(Dataset):
    """PyTorch dataset for LLPS protein sequences.
    IUPred3 scores are returned as zeros (this ablation).
    """

    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path)
        self.sequences = self.df["sequence"].tolist()
        self.labels = self.df["label"].values.astype(np.float32)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        seq = self.sequences[idx]
        truncated = truncate_for_model(seq)
        iupred_scores = np.zeros(len(truncated), dtype=np.float32)
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
