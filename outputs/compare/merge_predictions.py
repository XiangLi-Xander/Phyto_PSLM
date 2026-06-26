"""Merge PhytoRNP_PSLM predictions with PSProteinPredict model predictions."""

import pandas as pd
import os

COMPARE = '/home/xiaojian/LLPS_model_project/PhytoRNP_PSLM/outputs/compare'

# 1. Load PSProteinPredict predictions
psp = pd.read_csv(f'{COMPARE}/test_set_predictions_all_models.csv')

# 2. Load PhytoRNP_PSLM predictions
phyto_path = f'{COMPARE}/test_with_scores.csv'
if not os.path.exists(phyto_path):
    print(f'ERROR: {phyto_path} not found. PhytoRNP_PSLM inference still running?')
    print('Check: tail -f /home/xiaojian/LLPS_model_project/PhytoRNP_PSLM/outputs/compare/phyto_predict.log')
    exit(1)

phyto = pd.read_csv(phyto_path)
phyto = phyto[['UniprotEntry', 'LLPS_Score']].drop_duplicates('UniprotEntry')

# 3. Merge
merged = psp.merge(phyto, on='UniprotEntry', how='left')

# Move PhytoRNP_PSLM score to front
cols = ['UniprotEntry', 'true_label', 'LLPS_Score'] + \
       [c for c in merged.columns if c not in ['UniprotEntry', 'true_label', 'LLPS_Score']]
merged = merged[cols]
merged = merged.round(6)

out = f'{COMPARE}/test_set_predictions_all_models_with_phyto.csv'
merged.to_csv(out, index=False)
print(f'Saved: {out}')
print(f'Shape: {merged.shape}')
print(f'Columns: {list(merged.columns)}')
print(f'\nPhytoRNP_PSLM predictions available: {merged["LLPS_Score"].notna().sum()} / {len(merged)}')
print(f'Missing: {merged["LLPS_Score"].isna().sum()}')
print()

from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, average_precision_score
valid = merged.dropna(subset=['LLPS_Score'])
y = valid['true_label']
p = valid['LLPS_Score']
print('PhytoRNP_PSLM on test set:')
print(f'  AUC-ROC: {roc_auc_score(y, p):.4f}')
print(f'  AUPRC:   {average_precision_score(y, p):.4f}')
print(f'  F1:      {f1_score(y, (p>=0.5).astype(int)):.4f}')
print(f'  Acc:     {accuracy_score(y, (p>=0.5).astype(int))*100:.2f}%')
