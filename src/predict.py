"""
Predict LLPS scores for protein sequences from a CSV file.

Loads a trained model, extracts ESM2 and IUPred3 features on-the-fly,
and produces per-sequence LLPS probability scores along with a
histogram + KDE visualisation.
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import gaussian_kde
from torch.amp import autocast
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from src.model import ResidueLLPSClassifier  # noqa: E402
from src.utils import (  # noqa: E402
    PROJECT_ROOT,
    ESM2Extractor,
    clean_sequence_df,
    get_iupred_scores,
)


def predict_csv(
    input_path: str,
    output_dir: str,
    seq_col: str = "sequence",
    label: int = None,
    batch_size: int = 32,
):
    """Run LLPS prediction on sequences in a CSV file."""
    os.makedirs(output_dir, exist_ok=True)

    csv_name = os.path.splitext(os.path.basename(input_path))[0]
    output_csv = os.path.join(output_dir, f"{csv_name}_with_scores.csv")
    output_fig = os.path.join(output_dir, f"{csv_name}_score_distribution.png")

    print(f"\n{'=' * 60}")
    print(f"Predicting: {csv_name}")
    print(f"{'=' * 60}")

    # 1. Load and clean
    print(f"\n[1/4] Loading and cleaning data...")
    df = pd.read_csv(input_path)
    df = clean_sequence_df(df, seq_col=seq_col, label=label)
    print(f"  Total sequences after cleaning: {len(df)}")
    sequences = df[seq_col].tolist()

    # 2. Prepare model
    print(f"\n[2/4] Preparing model and ESM2 extractor...")
    esm2_dir = os.path.join(PROJECT_ROOT, "esm2")
    model_path = os.path.join(PROJECT_ROOT, "outputs", "models", "best_model.pth")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    extractor = ESM2Extractor(esm2_dir, device)
    model = ResidueLLPSClassifier(d_model=1280, hidden=512).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 3. Predict
    print(f"\n[3/4] Predicting LLPS scores (batch_size={batch_size})...")
    all_probs = []
    with torch.no_grad():
        for i in tqdm(range(0, len(sequences), batch_size), desc="Predicting"):
            batch_seqs = sequences[i : i + batch_size]

            batch_iupred = [get_iupred_scores(seq) for seq in batch_seqs]
            iupred_tensors = [torch.tensor(s, dtype=torch.float32) for s in batch_iupred]
            lengths = torch.tensor([t.size(0) for t in iupred_tensors], dtype=torch.long)
            iupred_padded = pad_sequence(iupred_tensors, batch_first=True, padding_value=0.0)

            esm_embeds, _ = extractor.extract_residue_batch_tensors(
                batch_seqs, pad_to_length=iupred_padded.size(1)
            )
            esm_embeds = esm_embeds.to(device)
            iupred_padded = iupred_padded.to(device)
            lengths = lengths.to(device)

            with autocast("cuda"):
                logits = model(esm_embeds, iupred_padded, lengths)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs)

    all_probs = np.array(all_probs)
    df["LLPS_Score"] = all_probs
    df.to_csv(output_csv, index=False)
    print(f"  Results saved to {output_csv}")

    # 4. Visualise
    print(f"\n[4/4] Visualising score distribution...")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(
        all_probs, bins=50, density=True, alpha=0.6,
        color="steelblue", edgecolor="black", label="Histogram",
    )
    if len(all_probs) > 1:
        kde = gaussian_kde(all_probs)
        x_range = np.linspace(all_probs.min(), all_probs.max(), 500)
        ax.plot(x_range, kde(x_range), color="darkred", linewidth=2, label="KDE")

    mean_val = np.mean(all_probs)
    median_val = np.median(all_probs)
    ax.axvline(
        mean_val, color="green", linestyle="--", linewidth=1.5,
        label=f"Mean={mean_val:.4f}",
    )
    ax.axvline(
        median_val, color="orange", linestyle="-.", linewidth=1.5,
        label=f"Median={median_val:.4f}",
    )

    ax.set_xlabel("Predicted LLPS Probability", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(
        f"{csv_name} LLPS Score Distribution\n"
        f"(n={len(all_probs)}, Mean={mean_val:.4f}, Median={median_val:.4f})",
        fontsize=13,
    )
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_fig, dpi=300)
    plt.close()
    print(f"  Figure saved to {output_fig}")

    print(f"\n  Score Statistics:")
    print(f"    Mean:   {mean_val:.4f}")
    print(f"    Median: {median_val:.4f}")
    print(f"    Std:    {np.std(all_probs):.4f}")
    print(f"    Max:    {np.max(all_probs):.4f}")
    print(f"    Min:    {np.min(all_probs):.4f}")
    print(f"    >0.5:   {np.sum(all_probs > 0.5)} ({100 * np.mean(all_probs > 0.5):.2f}%)")

    return df


def main():
    output_dir = os.path.join(PROJECT_ROOT, "outputs", "predictions")

    pos_csv = os.path.join(PROJECT_ROOT, "all_species_llps_sequences.csv")
    predict_csv(pos_csv, output_dir, seq_col="sequence", label=1)

    neg_csv = os.path.join(PROJECT_ROOT, "all_species_no_llps_merged.csv")
    predict_csv(neg_csv, output_dir, seq_col="sequence", label=0)

    print(f"\nOutput directory: {output_dir}")


if __name__ == "__main__":
    main()
