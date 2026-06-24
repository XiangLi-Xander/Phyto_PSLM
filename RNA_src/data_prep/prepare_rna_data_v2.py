"""
Prepare RNA dependency training data from 3-species LLPS-positive proteins.
Split into train/val/test (70/15/15).
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import pandas as pd
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, HERE)

pred_path = os.path.join(HERE, "outputs", "predictions", "rna_dependent_predictions_all3.xlsx")
out_dir = os.path.join(HERE, "data", "rna_dep")
os.makedirs(out_dir, exist_ok=True)

df = pd.read_excel(pred_path)

# Filter LLPS >= 0.5
df_llps = df[df["LLPS_Score"] >= 0.5].copy()
print(f"Total LLPS-positive: {len(df_llps)}")

# Convert RNA label to binary (ground truth from experiment)
df_llps["rna_label"] = (df_llps["rna_dependent"] == "Yes").astype(int)
df_llps["sequence"] = df_llps["sequence"].astype(str).str.strip().str.upper()

n_dep = df_llps["rna_label"].sum()
n_ind = len(df_llps) - n_dep
print(f"  RNA-dependent: {n_dep}")
print(f"  RNA-independent: {n_ind}")

# Stratified split: 70/15/15
tmp_df, test_df = train_test_split(
    df_llps, test_size=0.15, random_state=42,
    stratify=df_llps["rna_label"],
)
train_df, val_df = train_test_split(
    tmp_df, test_size=0.1765, random_state=42,
    stratify=tmp_df["rna_label"],
)

for name, data in [("train", train_df), ("val", val_df), ("test", test_df)]:
    out = data[["sequence", "rna_label"]].rename(columns={"rna_label": "label"})
    out.to_csv(os.path.join(out_dir, f"{name}.csv"), index=False)
    pos = int(out["label"].sum())
    neg = len(out) - pos
    print(f"  {name}.csv: {len(out)} rows, pos={pos}, neg={neg}")
