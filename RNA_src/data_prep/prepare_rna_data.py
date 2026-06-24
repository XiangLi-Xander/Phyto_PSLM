"""
Prepare RNA dependency training data from RNP_input.csv.
Split into train/val/test (70/15/15, stratified).
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import pandas as pd
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, HERE)
from src.utils import clean_sequence_df

csv_path = os.path.join(HERE, "data", "RNP_input.csv")
out_dir = os.path.join(HERE, "data", "rna_dep")
os.makedirs(out_dir, exist_ok=True)

# Read RNP_input.csv
df = pd.read_csv(csv_path)
print(f"Total rows: {len(df)}")

# Clean sequences
df = clean_sequence_df(df, seq_col="Sequence")
print(f"After cleaning: {len(df)}")

# Label is already 0/1 in RNP_input.csv
df["label"] = df["Label"].astype(int)
n_pos = df["label"].sum()
n_neg = len(df) - n_pos
print(f"  RNA-dependent (1):   {n_pos}")
print(f"  RNA-independent (0): {n_neg}")

# Split: train 70% / val 15% / test 15% (stratified)
tmp_df, test_df = train_test_split(
    df, test_size=0.15, random_state=42,
    stratify=df["label"],
)
train_df, val_df = train_test_split(
    tmp_df, test_size=0.1765, random_state=42,
    stratify=tmp_df["label"],
)

# Keep only sequence + label columns
for name, data in [("train", train_df), ("val", val_df), ("test", test_df)]:
    out = data[["Sequence", "label"]].rename(columns={"Sequence": "sequence"})
    out.to_csv(os.path.join(out_dir, f"{name}.csv"), index=False)
    pos = out["label"].sum()
    neg = len(out) - pos
    print(f"  {name}.csv: {len(out)} rows, pos={pos}, neg={neg}")
