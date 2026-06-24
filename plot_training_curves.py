"""
Plot training curves (Loss, F1, AUC) for original model and ablations.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

TOP = os.path.dirname(os.path.abspath(__file__))

ABLATIONS = {
    "Original (ESM2+IUPred3)": ("outputs", "logs"),
    "01_no_esm2 (No ESM2)": ("ablation/01_no_esm2/outputs", "logs"),
    "02_no_iupred (No IUPred3)": ("ablation/02_no_iupred/outputs", "logs"),
    "03_onehot_esm (One-Hot)": ("ablation/03_onehot_esm/outputs", "logs"),
}

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
LINESTYLES = {"train": "-", "val": "--"}


def find_latest_log(base, log_rel):
    log_dir = os.path.join(TOP, base, log_rel)
    if not os.path.isdir(log_dir):
        return None
    files = sorted([f for f in os.listdir(log_dir) if f.endswith(".txt")])
    return os.path.join(log_dir, files[-1]) if files else None


def load_metrics(log_path):
    df = pd.read_csv(log_path, sep="\t")
    return df


def plot_curves(ax, df, label, color, metric="F1"):
    train_df = df[df["Phase"] == "train"]
    val_df = df[df["Phase"] == "val"]

    if len(train_df) > 0:
        ax.plot(train_df["Step"], train_df[metric],
                label=f"{label} (train)", color=color, linestyle="-", alpha=0.7)
    if len(val_df) > 0:
        ax.plot(val_df["Step"], val_df[metric],
                label=f"{label} (val)", color=color, linestyle="--", alpha=0.7)


def main():
    os.makedirs(os.path.join(TOP, "outputs", "figures"), exist_ok=True)

    dfs = {}
    for name, (base, log_rel) in ABLATIONS.items():
        log_path = find_latest_log(base, log_rel)
        if log_path and os.path.getsize(log_path) > 50:
            dfs[name] = load_metrics(log_path)
            print(f"  {name:<35s} {log_path}  ({len(dfs[name])} rows)")
        else:
            print(f"  {name:<35s} NOT FOUND or empty")

    if not dfs:
        print("No training logs found. Train the models first.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    metrics = ["Loss", "F1", "AUC"]

    for i, metric in enumerate(metrics):
        ax = axes[i]
        for j, (name, df) in enumerate(dfs.items()):
            color = COLORS[j % len(COLORS)]
            plot_curves(ax, df, name, color, metric)

        ax.set_title(f"{metric} Curve", fontsize=13, fontweight="bold")
        ax.set_xlabel("Step", fontsize=11)
        ax.set_ylabel(metric, fontsize=11)
        ax.legend(fontsize=7, loc="best", ncol=2)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(TOP, "outputs", "figures", "training_curves.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"\nSaved: {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
