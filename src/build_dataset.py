"""
Build the LLPS dataset with similarity-aware splitting.

This script performs the following steps:

1. Loads positive (LLPS) and negative (non-LLPS) sequences from CSV files.
2. Deduplicates negative sequences using MMseqs2 clustering at 30% identity.
3. Clusters positive sequences (retaining all) for similarity-aware splitting.
4. Performs a stratified split at the cluster level (70/15/15).
5. Saves the resulting train/val/test CSV files (sequence + label only).

IUPred3 per-residue scores are **not** pre-computed; they are generated
on-the-fly during training and inference via ``get_iupred_scores``.
"""

import os
import random
import subprocess
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from src.utils import PROJECT_ROOT, clean_sequence_df  # noqa: E402

# Paths
POS_CSV = os.path.join(PROJECT_ROOT, "all_species_llps_sequences.csv")
NEG_CSV = os.path.join(PROJECT_ROOT, "all_species_no_llps_merged.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

MMSEQS_BIN = "/root/miniconda3/envs/LLPS_model/bin/mmseqs"


def load_labeled_csv(csv_path: str, label: int) -> pd.DataFrame:
    """Load a CSV, clean sequences, attach a label, keep known columns."""
    df = pd.read_csv(csv_path)
    df = clean_sequence_df(df, seq_col="sequence", label=label)
    keep_cols = ["UniprotEntry", "sequence", "label", "Species"]
    df = df[[c for c in keep_cols if c in df.columns]]
    return df


def write_fasta(df: pd.DataFrame, out_path: str, id_col="UniprotEntry", seq_col="sequence"):
    """Write sequences in FASTA format for MMseqs2."""
    with open(out_path, "w") as f:
        for idx, row in df.iterrows():
            seq_id = str(row.get(id_col, f"seq_{idx}"))
            seq = str(row[seq_col])
            f.write(f">{seq_id}\n{seq}\n")


def run_mmseqs_cluster(
    input_fasta: str,
    output_prefix: str,
    tmp_dir: str,
    min_seq_id: float = 0.3,
):
    """Run MMseqs2 easy-cluster with default parameters."""
    os.makedirs(tmp_dir, exist_ok=True)
    cmd = [
        MMSEQS_BIN, "easy-cluster",
        input_fasta, output_prefix, tmp_dir,
        "--min-seq-id", str(min_seq_id),
        "-c", "0.8",
        "--cov-mode", "0",
        "--cluster-mode", "0",
        "-v", "1",
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def parse_mmseqs_cluster(cluster_tsv_path: str):
    """Parse an MMseqs2 ``_cluster.tsv`` file into a list of clusters."""
    clusters_dict = {}
    with open(cluster_tsv_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                rep_id, member_id = parts[0], parts[1]
                if rep_id not in clusters_dict:
                    clusters_dict[rep_id] = [rep_id]
                if member_id != rep_id:
                    clusters_dict[rep_id].append(member_id)
    return list(clusters_dict.values())


def cluster_based_split(
    df: pd.DataFrame,
    clusters: list,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    seed: int = RANDOM_SEED,
):
    """Split data at the cluster level to prevent information leakage."""
    rng = np.random.default_rng(seed)
    seq_to_cluster = {}
    for cid, members in enumerate(clusters):
        for m in members:
            seq_to_cluster[m] = cid

    id_col = df.columns[0] if "UniprotEntry" not in df.columns else "UniprotEntry"
    df = df.copy()
    df["cluster_id"] = df[id_col].astype(str).map(seq_to_cluster)

    missing_mask = df["cluster_id"].isna()
    next_cid = df["cluster_id"].max() + 1 if not df["cluster_id"].isna().all() else 0
    for i in range(missing_mask.sum()):
        df.loc[df[missing_mask].index[i], "cluster_id"] = next_cid + i

    cluster_groups = df.groupby("cluster_id")
    cluster_ids = list(cluster_groups.groups.keys())
    rng.shuffle(cluster_ids)

    n_total = len(df)
    n_train = int(n_total * train_frac)
    n_val = int(n_total * val_frac)

    train_clusters, val_clusters, test_clusters = [], [], []
    cumsum = 0
    for cid in cluster_ids:
        size = len(cluster_groups.get_group(cid))
        if cumsum < n_train:
            train_clusters.append(cid)
        elif cumsum < n_train + n_val:
            val_clusters.append(cid)
        else:
            test_clusters.append(cid)
        cumsum += size

    train_df = df[df["cluster_id"].isin(train_clusters)].drop(columns=["cluster_id"])
    val_df = df[df["cluster_id"].isin(val_clusters)].drop(columns=["cluster_id"])
    test_df = df[df["cluster_id"].isin(test_clusters)].drop(columns=["cluster_id"])
    return train_df, val_df, test_df


def main():
    print("=" * 60)
    print("LLPS Dataset Construction")
    print("=" * 60)

    print("\n[1/5] Loading data...")
    pos_df = load_labeled_csv(POS_CSV, label=1)
    neg_df = load_labeled_csv(NEG_CSV, label=0)
    print(f"  Positive: {len(pos_df)} | Negative (raw): {len(neg_df)}")

    print("\n[2/5] Negative deduplication (MMseqs2 at 30% identity)...")
    tmp_dir = os.path.join(PROJECT_ROOT, "data", "tmp")
    mmseqs_tmp = os.path.join(tmp_dir, "mmseqs_tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    neg_fasta = os.path.join(tmp_dir, "neg.fasta")
    neg_prefix = os.path.join(tmp_dir, "neg_cluster")
    neg_cluster_tsv = neg_prefix + "_cluster.tsv"

    if not os.path.exists(neg_cluster_tsv):
        write_fasta(neg_df, neg_fasta)
        run_mmseqs_cluster(neg_fasta, neg_prefix, mmseqs_tmp)
    else:
        print("  Using existing MMseqs2 results.")

    neg_clusters = parse_mmseqs_cluster(neg_cluster_tsv)
    rep_ids = {c[0] for c in neg_clusters if c}
    neg_df = neg_df[neg_df["UniprotEntry"].astype(str).isin(rep_ids)]
    print(f"  Negative clusters: {len(neg_clusters)} | Representatives: {len(rep_ids)}")

    print("\n[3/5] Positive clustering (MMseqs2 at 30% identity)...")
    pos_fasta = os.path.join(tmp_dir, "pos.fasta")
    pos_prefix = os.path.join(tmp_dir, "pos_cluster")
    pos_cluster_tsv = pos_prefix + "_cluster.tsv"

    if not os.path.exists(pos_cluster_tsv):
        write_fasta(pos_df, pos_fasta)
        run_mmseqs_cluster(pos_fasta, pos_prefix, mmseqs_tmp)
    else:
        print("  Using existing MMseqs2 results.")

    pos_clusters = parse_mmseqs_cluster(pos_cluster_tsv)
    print(f"  Positive clusters: {len(pos_clusters)}")

    print("\n[4/5] Similarity-aware stratified split...")
    pos_train, pos_val, pos_test = cluster_based_split(
        pos_df, pos_clusters, train_frac=0.70, val_frac=0.15
    )
    neg_train, neg_val, neg_test = cluster_based_split(
        neg_df, neg_clusters, train_frac=0.70, val_frac=0.15
    )

    train_df = (
        pd.concat([pos_train, neg_train])
        .sample(frac=1, random_state=RANDOM_SEED)
        .reset_index(drop=True)
    )
    val_df = (
        pd.concat([pos_val, neg_val])
        .sample(frac=1, random_state=RANDOM_SEED)
        .reset_index(drop=True)
    )
    test_df = (
        pd.concat([pos_test, neg_test])
        .sample(frac=1, random_state=RANDOM_SEED)
        .reset_index(drop=True)
    )

    print(f"  Train: {len(train_df)} (pos={len(pos_train)}, neg={len(neg_train)})")
    print(f"  Val:   {len(val_df)} (pos={len(pos_val)}, neg={len(neg_val)})")
    print(f"  Test:  {len(test_df)} (pos={len(pos_test)}, neg={len(neg_test)})")

    print("\n[5/5] Saving splits...")
    for split_name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        out_path = os.path.join(OUTPUT_DIR, f"{split_name}.csv")
        df.to_csv(out_path, index=False)
        print(f"  Saved: {out_path} ({len(df)} samples)")

    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)

    print("\n" + "=" * 60)
    print("Dataset construction complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
