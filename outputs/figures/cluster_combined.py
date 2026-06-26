"""
Unified UMAP: training data + ALL_RBP in the same coordinate space.

Produces CSVs combining background / LLPS / RBP with UMAP coordinates
from both raw ESM2 features and model-pooled features, all in one space.
"""

import gc, os, sys, warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import torch
import umap
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch.nn.utils.rnn import pad_sequence
from torch.amp import autocast
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from src.model import ResidueLLPSClassifier
from src.utils import PROJECT_ROOT, ESM2Extractor, clean_sequence_df, get_iupred_scores

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ESM2_DIR = os.path.join(PROJECT_ROOT, 'esm2')
MODEL_PATH = os.path.join(PROJECT_ROOT, 'outputs', 'models', 'best_model.pth')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'outputs', 'predictions')
os.makedirs(OUTPUT_DIR, exist_ok=True)
BATCH_SIZE = 128
NEG_SUBSAMPLE = None  # use all background sequences

print(f'Device: {DEVICE}')

# ---------------------------------------------------------------------------
# 1. Load all data
# ---------------------------------------------------------------------------
print('\n[1] Loading data...')

pos_orig = pd.read_csv(os.path.join(PROJECT_ROOT, 'all_species_llps_sequences.csv'))
neg_orig = pd.read_csv(os.path.join(PROJECT_ROOT, 'all_species_no_llps_merged.csv'))
rbp_orig = pd.read_csv(os.path.join(PROJECT_ROOT, 'ara_all_data', 'ALL_RBP.csv'))

pos_df = clean_sequence_df(pos_orig.copy(), seq_col='sequence', label=1)
neg_df = clean_sequence_df(neg_orig.copy(), seq_col='sequence', label=0)
rbp_df = clean_sequence_df(rbp_orig.copy(), seq_col='sequence')

# subsample negatives (disabled: use all)
if NEG_SUBSAMPLE and len(neg_df) > NEG_SUBSAMPLE:
    neg_df = neg_df.sample(n=NEG_SUBSAMPLE, random_state=42).reset_index(drop=True)

rbp_df = rbp_df.rename(columns={'Sequence': 'sequence'})

# tag sources
pos_df['source'] = 'LLPS'
neg_df['source'] = 'Background'
rbp_df['source'] = 'RBP'
rbp_df['label'] = -1

combined = pd.concat([neg_df, pos_df, rbp_df], ignore_index=True)
com_seqs = combined['sequence'].tolist()
N = len(combined)
print(f'  Background: {len(neg_df)}')
print(f'  LLPS:       {len(pos_df)}')
print(f'  RBP:        {len(rbp_df)}')
print(f'  Total:      {N}')

# ============================================================================
# Method 1: Raw ESM2 CLS + IUPred3
# ============================================================================
print('\n========== Method 1: Raw ESM2 CLS + IUPred3 ==========')

print('[1.1] Extracting ESM2 CLS embeddings...')
tokenizer = AutoTokenizer.from_pretrained(ESM2_DIR, local_files_only=True)
esm_model = (
    AutoModel.from_pretrained(ESM2_DIR, local_files_only=True, dtype=torch.float16)
    .to(DEVICE)
)
esm_model.eval()

def extract_esm(seqs, desc=''):
    out = []
    for i in range(0, len(seqs), BATCH_SIZE):
        batch = seqs[i:i+BATCH_SIZE]
        inputs = tokenizer(batch, truncation=True, max_length=512, padding=True, return_tensors='pt')
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.inference_mode():
            h = esm_model(**inputs).last_hidden_state[:, 0, :].float()
        out.append(h.cpu().numpy())
        if (i // BATCH_SIZE) % 10 == 0:
            print(f'  {desc}: {min(i+BATCH_SIZE, len(seqs))}/{len(seqs)}', flush=True)
    return np.concatenate(out, axis=0)

esm_feats = extract_esm(com_seqs, 'ESM2 CLS')
print(f'  Shape: {esm_feats.shape}')
del esm_model, tokenizer; gc.collect()

print('[1.2] Computing IUPred3 aggregate features...')
def iupred_feats(seq):
    scores = get_iupred_scores(seq)
    if len(scores) == 0:
        return np.zeros(8, dtype=np.float32)
    return np.array([
        scores.mean(), scores.std(),
        (scores > 0.5).mean(), (scores > 0.7).mean(),
        scores.max(), scores.min(),
        np.percentile(scores, 90), np.percentile(scores, 10),
    ], dtype=np.float32)

iup_all = []
for i, s in enumerate(com_seqs):
    iup_all.append(iupred_feats(s))
    if (i + 1) % 2000 == 0:
        print(f'  IUPred: {i+1}/{N}', flush=True)
iup_feats = np.stack(iup_all, axis=0)
print(f'  Shape: {iup_feats.shape}')

print('[1.3] PCA + UMAP...')
scaler = StandardScaler()
esm_scaled = scaler.fit_transform(esm_feats)
pca = PCA(n_components=20, random_state=42)
esm_pc = pca.fit_transform(esm_scaled)
print(f'  ESM2 PCA explained variance: {pca.explained_variance_ratio_.sum():.3f}')

iup_scaler = StandardScaler()
iup_scaled = iup_scaler.fit_transform(iup_feats)

feat = np.concatenate([esm_pc, iup_scaled], axis=1)
print(f'  Feature dim: {feat.shape[1]}')

umap_raw = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.2, random_state=42, verbose=False)
raw_2d = umap_raw.fit_transform(feat)

out_raw = combined[['sequence', 'source', 'label']].copy()
out_raw['UMAP1_raw_esm'] = raw_2d[:, 0]
out_raw['UMAP2_raw_esm'] = raw_2d[:, 1]
out_raw.to_csv(os.path.join(OUTPUT_DIR, 'combined_umap_raw_esm.csv'), index=False)
print(f'  Saved: combined_umap_raw_esm.csv')

# ============================================================================
# Method 2: Model pooled features
# ============================================================================
print('\n========== Method 2: Model Pooled Features ==========')

print('[2.1] Loading model...')
model = ResidueLLPSClassifier(d_model=1280, hidden=512).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
extractor = ESM2Extractor(ESM2_DIR, DEVICE)

print('[2.2] Extracting pooled features via forward hook...')
pooled_features = []
hook_handle = model.query_pool.register_forward_hook(
    lambda mod, inp, out: pooled_features.append(out.detach().cpu())
)

with torch.no_grad():
    for i in range(0, N, BATCH_SIZE):
        batch_seqs = com_seqs[i:i+BATCH_SIZE]
        batch_iupred = [get_iupred_scores(s) for s in batch_seqs]
        iupred_tensors = [torch.tensor(s, dtype=torch.float32) for s in batch_iupred]
        lengths = torch.tensor([t.size(0) for t in iupred_tensors], dtype=torch.long)
        iupred_padded = pad_sequence(iupred_tensors, batch_first=True, padding_value=0.0)

        esm_embeds, _ = extractor.extract_residue_batch_tensors(
            batch_seqs, pad_to_length=iupred_padded.size(1)
        )
        esm_embeds = esm_embeds.to(DEVICE)
        iupred_padded = iupred_padded.to(DEVICE)
        lengths = lengths.to(DEVICE)

        with autocast('cuda'):
            _ = model(esm_embeds, iupred_padded, lengths)

        if (i // BATCH_SIZE) % 5 == 0:
            print(f'  Pooled: {min(i+BATCH_SIZE, N)}/{N}', flush=True)

hook_handle.remove()
pooled = torch.cat(pooled_features, dim=0).numpy()
print(f'  Shape: {pooled.shape}')

print('[2.3] UMAP on pooled features...')
umap_pool = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.2, random_state=42, verbose=False)
pooled_2d = umap_pool.fit_transform(pooled)

out_pooled = combined[['sequence', 'source', 'label']].copy()
out_pooled['UMAP1_model_pooled'] = pooled_2d[:, 0]
out_pooled['UMAP2_model_pooled'] = pooled_2d[:, 1]
out_pooled.to_csv(os.path.join(OUTPUT_DIR, 'combined_umap_model_pooled.csv'), index=False)
print(f'  Saved: combined_umap_model_pooled.csv')

# ============================================================================
# Merged output
# ============================================================================
print('\n[3] Merging all coordinates + LLPS scores...')
merged = combined[['sequence', 'source', 'label']].copy()
merged['UMAP1_raw_esm'] = raw_2d[:, 0]
merged['UMAP2_raw_esm'] = raw_2d[:, 1]
merged['UMAP1_model_pooled'] = pooled_2d[:, 0]
merged['UMAP2_model_pooled'] = pooled_2d[:, 1]

# Add LLPS scores for RBP sequences
rbp_scores = pd.read_csv(os.path.join(OUTPUT_DIR, 'ALL_RBP_with_scores.csv'))
merged = merged.merge(rbp_scores[['sequence', 'LLPS_Score']], on='sequence', how='left')

merged.to_csv(os.path.join(OUTPUT_DIR, 'combined_with_umap.csv'), index=False)

# Also save per-source subsets
for src in ['Background', 'LLPS', 'RBP']:
    sub = merged[merged['source'] == src]
    sub.to_csv(os.path.join(OUTPUT_DIR, f'{src}_with_umap.csv'), index=False)
    print(f'  {src}_with_umap.csv: {len(sub)} rows')

print(f'\nAll outputs in {OUTPUT_DIR}/')
print('Done!')
