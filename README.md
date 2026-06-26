# PhytoRNP\_PSLM

A residue-level deep learning model for predicting protein liquid-liquid phase separation (LLPS) propensity from primary amino acid sequences, with a focus on plant proteomes.

## Overview

The model architecture combines two complementary residue-level feature streams:

- **ESM2\_t33\_650M** per-residue hidden states (1280 dimensions) capturing evolutionary and biophysical context
- **IUPred3** per-residue disorder scores (1 dimension) providing a structural prior

These streams are projected into a shared hidden space via asymmetric two-layer MLPs, fused by element-wise addition, enriched with learned positional encodings, processed by a six-layer Transformer encoder (Pre-LN, 8 heads), and aggregated via multi-query cross-attention pooling into a sequence-level logit.

## Requirements

- Python 3.11+
- PyTorch 2.7.1+ (CUDA 12.6 recommended)
- An NVIDIA GPU with at least 24 GB of VRAM

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/PhytoRNP_PSLM.git
cd PhytoRNP_PSLM

# 2. (Recommended) Create a conda environment
conda create -n phyto_llps python=3.11
conda activate phyto_llps

# 3. Install dependencies
pip install torch>=2.7.1 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

### Model weights & large files

Some files are too large for GitHub and must be downloaded separately.  
Use the automated download script:

```bash
# Install requirements
pip install huggingface_hub requests

# Download all models (ESM2 + pre-trained checkpoint)
python scripts/download_models.py
```

**What gets downloaded:**

| File | Size | Source | Destination |
|------|------|--------|-------------|
| ESM2-t33-650M (`model.safetensors`) | ~2.5 GB | [HuggingFace: facebook/esm2_t33_650M_UR50D](https://huggingface.co/facebook/esm2_t33_650M_UR50D) | `esm2/` |
| ESM+IUPred benchmark features | ~422 MB | [HuggingFace Assets](https://huggingface.co/XiangLi-Xander/Phyto_PSLM_assets) | `outputs/benchmarks/` |
| RF handcrafted benchmark model | ~126 MB | [HuggingFace Assets](https://huggingface.co/XiangLi-Xander/Phyto_PSLM_assets) | `outputs/benchmarks/` |

> These 3 files exceed GitHub's file size limits and are hosted on HuggingFace.
> Before running the downloader, upload your local copies to a HuggingFace repo,
> then set `HF_REPO_ID` in `scripts/download_models.py`.

**Upload your own copies to HuggingFace:**

```bash
# 1. Install
pip install huggingface_hub
huggingface-cli login

# 2. Create a model repo (or use an existing one)
#    https://huggingface.co/new

# 3. Upload the 2 benchmark files
huggingface-cli upload <your-username>/<your-repo> outputs/benchmarks/esm_iupred_features.npz esm_iupred_features.npz
huggingface-cli upload <your-username>/<your-repo> outputs/benchmarks/llps_rf_handcrafted.pkl llps_rf_handcrafted.pkl
```

**Manual download:**

```bash
# ESM2 model
pip install huggingface_hub
huggingface-cli download facebook/esm2_t33_650M_UR50D --local-dir esm2

# Benchmark assets (after uploading to your own HF repo)
huggingface-cli download XiangLi-Xander/Phyto_PSLM_assets esm_iupred_features.npz --local-dir outputs/benchmarks
huggingface-cli download XiangLi-Xander/Phyto_PSLM_assets llps_rf_handcrafted.pkl --local-dir outputs/benchmarks
```

### IUPred3

The IUPred3 library is included in the `iupred3/` directory. No additional setup is required.

## Usage

### Predict LLPS for custom protein sequences

Prepare a CSV file with a column containing amino acid sequences:

```csv
sequence
MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQVGQVELGGGPGAGSLQPLALEGSLQKRGIVEQCCTSICSLYQLENYCN
MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGKKVADALTNAVAHVDDMPNALSALSDLHAHKLRVDPVNFKLLSHCLLVTLAAHLPAEFTPAVHASLDKFLASVSTVLTSKYR
```

Run prediction:

```bash
python -m src.predict --input_path /path/to/your_sequences.csv
```

Optional arguments:
- `--seq_col` — column name for sequences (default: `sequence`)
- `--batch_size` — batch size (default: `32`)

Output:
- `outputs/predictions/your_sequences_with_scores.csv` — input data with an added `LLPS_Score` column
- `outputs/predictions/your_sequences_score_distribution.png` — histogram of predicted scores

### Predict the *Arabidopsis thaliana* proteome

```bash
python -m src.predict_ara
```

Requires `ara_all_data/ara_all_with_sequence.csv` (included).

### Train from scratch

```bash
# 1. Build the dataset
python -m src.build_dataset

# 2. Train the model
python -m src.train
```

The training loop is step-based (10,000 steps by default), with evaluation every 300 steps and early stopping based on validation F1-score. The best checkpoint is saved to `outputs/models/best_model.pth`.

## Repository structure

```
PhytoRNP_PSLM/
├── README.md
├── requirements.txt
├── scripts/
│   └── download_models.py  # Automated model weights downloader
├── data/
│   ├── processed/          # {train,val,test}.csv
│   ├── rna_dep/            # RNA-dependency dataset splits
│   └── all_species_*.csv   # Raw positive/negative sequences
├── src/
│   ├── __init__.py
│   ├── model.py            # Transformer-based LLPS classifier
│   ├── features.py         # Feature extraction (ESM2 + IUPred3)
│   ├── utils.py            # Dataset class, collate, metrics
│   ├── build_dataset.py    # Data preprocessing and splitting
│   ├── train.py            # Training loop
│   ├── predict.py          # Inference on CSV input
│   └── predict_ara.py      # Inference on A. thaliana proteome
├── esm2/                   # ESM2 model weights (download via scripts/download_models.py)
├── iupred3/                # IUPred3 library (included)
├── RNA_src/                # RNA-dependency analysis source code
├── outputs/
│   ├── models/             # Pre-trained checkpoints (download separately)
│   ├── logs/
│   ├── predictions/
│   └── figures/
└── ara_all_data/           # A. thaliana proteome data
```

## Input data format

For custom prediction, provide a CSV file with at minimum a sequence column (default column name: `sequence`). All other columns are preserved in the output.

For training, the expected CSV columns are `sequence` and `label` (1 for LLPS, 0 for non-LLPS).

## Model architecture

```
ESM2_embed (B, L, 1280) ──► [Linear → LN → GELU → Drop]×2 ──► e (B, L, 512) ──┐
                                (1280→640, 640→512)                             │
                                                                                 ├──► x = e + i
IUPred_score (B, L, 1)  ──► [Linear → LN → GELU → Drop]×2 ──► i (B, L, 512) ──┘
                                (1→256, 256→512)                                │
                                                                                 ▼
                                                         + Learned PosEmbed (L, 512)
                                                                                 │
                                                                                 ▼
                                                         [TransformerEncoderLayer] × 6
                                                         (Pre-LN, d=512, 8 heads, FFN=2048)
                                                                                 │
                                                                                 ▼
                                                         Mask padding → Multi-Query Pooling
                                                         (4 learnable queries, MHA 8 heads)
                                                                                 │
                                                                                 ▼
                                                         LN → Linear(512→256) → GELU
                                                         → Drop → Linear(256→1) → logit
```

Total trainable parameters: approximately 23–25 million.

## Performance

Best model checkpoint on the held-out test set (threshold = 0.5):

| Metric       | Value  |
|--------------|--------|
| Accuracy     | 94.66% |
| F1-score     | 0.616  |
| AUC-ROC      | 0.950  |
| AUPRC        | 0.648  |
| FPR          | 2.99%  |

## Citation

If you use this model or code in your research, please cite the corresponding publication (to be added).

## License

Distributed under the MIT License. See `LICENSE` for more information.
