"""
Evaluate PSProteinPredict (LightGBM + PU Learning) model
on the PhytoRNP_PSLM held-out test set for direct comparison.
"""

import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, '/home/xiaojian/LLPS_model_project/PSProteinProject')

from sklearn.metrics import (roc_auc_score, accuracy_score, f1_score,
                             precision_score, recall_score, confusion_matrix,
                             average_precision_score)
from sklearn.utils import resample
from lightgbm import LGBMClassifier
from baggingPU import BaggingClassifierPU

BASE = '/home/xiaojian/LLPS_model_project/PhytoRNP_PSLM'

# 1. Load train/val/test splits by protein ID
train_ids = pd.read_csv(f'{BASE}/data/processed/train.csv')['UniprotEntry'].tolist()
val_ids   = pd.read_csv(f'{BASE}/data/processed/val.csv')['UniprotEntry'].tolist()
test_ids  = pd.read_csv(f'{BASE}/data/processed/test.csv')['UniprotEntry'].tolist()

# 2. Load raw feature data
pos = pd.read_csv(f'{BASE}/all_species_llps_sequences.csv')
neg = pd.read_csv(f'{BASE}/all_species_no_llps_merged.csv')

FEAT_COLS = ['Hydropathy', 'FCR', 'IDR', 'LCR', 'Pscore', 'PLAAC', 'catGRANULE']

# Merge pos and neg into single DataFrame indexed by UniprotEntry
pos = pos.set_index('UniprotEntry')
neg = neg.set_index('UniprotEntry')

# Ensure consistent columns
pos = pos[FEAT_COLS + ['Score']]
neg = neg[FEAT_COLS + ['Score']]

all_data = pd.concat([pos, neg])
all_data = all_data[~all_data.index.duplicated(keep='first')]

print(f'Total unique entries with features: {len(all_data)}')
print(f'  Train IDs: {len(train_ids)}, Val IDs: {len(val_ids)}, Test IDs: {len(test_ids)}')

# 3. Build X/y for each split
def make_split(id_list):
    available = [uid for uid in id_list if uid in all_data.index]
    missing = set(id_list) - set(available)
    if missing:
        print(f'  Warning: {len(missing)} IDs not found in feature data')
    df = all_data.loc[available]
    X = df[FEAT_COLS].fillna(0).values
    y = df['Score'].values.astype(int)
    return X, y, available

X_train, y_train, train_ok = make_split(train_ids)
X_val,   y_val,   val_ok   = make_split(val_ids)
X_test,  y_test,  test_ok  = make_split(test_ids)

print(f'\nTrain: {X_train.shape}, pos={y_train.sum()}, neg={(1-y_train).sum()}')
print(f'Val:   {X_val.shape}, pos={y_val.sum()}, neg={(1-y_val).sum()}')
print(f'Test:  {X_test.shape}, pos={y_test.sum()}, neg={(1-y_test).sum()}')

# 4. Train models & evaluate

def evaluate(y_true, y_pred, y_prob, name):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return {
        'Model': name,
        'Accuracy': accuracy_score(y_true, y_pred),
        'F1-score': f1_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'AUC-ROC': roc_auc_score(y_true, y_prob),
        'AUPRC': average_precision_score(y_true, y_prob),
        'FPR': fpr,
    }

results = []

# --- LightGBM (standard) ---
print('\n=== LightGBM (Standard) ===')
lgb = LGBMClassifier(objective='binary', boosting_type='dart', verbose=-1, random_state=2025)
lgb.fit(X_train, y_train)
y_prob_lgb = lgb.predict_proba(X_test)[:, 1]
y_pred_lgb = (y_prob_lgb >= 0.5).astype(int)
results.append(evaluate(y_test, y_pred_lgb, y_prob_lgb, 'LightGBM'))
print(f'  AUC-ROC: {results[-1]["AUC-ROC"]:.4f}, F1: {results[-1]["F1-score"]:.4f}')

# --- PU Learning + LightGBM ---
print('\n=== PU Learning + LightGBM ===')
lgb_base = LGBMClassifier(objective='binary', boosting_type='dart', verbose=-1)
pu = BaggingClassifierPU(base_estimator=lgb_base, n_estimators=10, random_state=2025)
pu.fit(X_train, y_train)
y_prob_pu = pu.predict_proba(X_test)[:, 1]
y_pred_pu = (y_prob_pu >= 0.5).astype(int)
results.append(evaluate(y_test, y_pred_pu, y_prob_pu, 'PU+LightGBM'))
print(f'  AUC-ROC: {results[-1]["AUC-ROC"]:.4f}, F1: {results[-1]["F1-score"]:.4f}')

# --- PhytoRNP_PSLM reference ---
results.append({
    'Model': 'PhytoRNP_PSLM (ESM2+Transformer)',
    'Accuracy': 0.9466,
    'F1-score': 0.616,
    'Precision': None,
    'Recall': None,
    'AUC-ROC': 0.950,
    'AUPRC': 0.648,
    'FPR': 0.0299,
})

print('\n\n' + '=' * 90)
print(f'{"Model":<35s} {"Acc":>8s} {"F1":>8s} {"AUC-ROC":>8s} {"AUPRC":>8s} {"FPR":>8s}')
print('=' * 90)
for r in results:
    acc  = f'{r["Accuracy"]*100:.2f}%' if r['Accuracy'] is not None else 'N/A'
    f1   = f'{r["F1-score"]:.4f}' if r['F1-score'] is not None else 'N/A'
    auc  = f'{r["AUC-ROC"]:.4f}' if r['AUC-ROC'] is not None else 'N/A'
    aupr = f'{r["AUPRC"]:.4f}' if r['AUPRC'] is not None else 'N/A'
    fpr  = f'{r["FPR"]*100:.2f}%' if r['FPR'] is not None else 'N/A'
    print(f'{r["Model"]:<35s} {acc:>8s} {f1:>8s} {auc:>8s} {aupr:>8s} {fpr:>8s}')
print('=' * 90)
