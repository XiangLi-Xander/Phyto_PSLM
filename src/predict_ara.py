"""
Predict LLPS scores for the *Arabidopsis thaliana* proteome.

Specialised prediction script for the A. thaliana proteome CSV.
Produces per-protein LLPS probabilities and a score-distribution figure.
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


def predict_ara(batch_size: int = 128):
    """Run LLPS prediction on the *A. thaliana* proteome."""
    ara_csv = os.path.join(PROJECT_ROOT, "ara_all_data", "ara_all_with_sequence.csv")
    esm2_dir = os.path.join(PROJECT_ROOT, "esm2")
    model_path = os.path.join(PROJECT_ROOT, "outputs", "models", "best_model.pth")
    output_csv = os.path.join(PROJECT_ROOT, "outputs", "predictions", "ara_with_scores.csv")
    output_fig = os.path.join(PROJECT_ROOT, "outputs", "figures", "ara_score_distribution.png")

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    os.makedirs(os.path.dirname(output_fig), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("A. thaliana Proteome LLPS Prediction")
    print("=" * 60)

    print("\n[1/4] Loading and cleaning A. thaliana data...")
    df = pd.read_csv(ara_csv)
    df = clean_sequence_df(df, seq_col="ProteinSequence")
    print(f"  Total sequences after cleaning: {len(df)}")
    sequences = df["ProteinSequence"].tolist()

    print("\n[2/4] Preparing model and ESM2 extractor...")
    extractor = ESM2Extractor(esm2_dir, device)
    model = ResidueLLPSClassifier(d_model=1280, hidden=512).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

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

    print("\n[4/4] Visualising score distribution...")
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
        f"A. thaliana LLPS Score Distribution\n"
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
    print("=" * 60)


if __name__ == "__main__":
    predict_ara()
