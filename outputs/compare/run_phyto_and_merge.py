"""Run PhytoRNP_PSLM inference on test set, then merge with PSProteinPredict scores."""
import sys, os, time
sys.path.insert(0, '/home/xiaojian/LLPS_model_project/PhytoRNP_PSLM')

from src.predict import predict_csv

COMPARE = '/home/xiaojian/LLPS_model_project/PhytoRNP_PSLM/outputs/compare'

t0 = time.time()
result_df = predict_csv(
    input_path='/home/xiaojian/LLPS_model_project/PhytoRNP_PSLM/data/processed/test.csv',
    output_dir=COMPARE,
    seq_col='sequence',
    label=None,
    batch_size=32,
)
elapsed = (time.time() - t0) / 3600
print(f'PhytoRNP_PSLM inference done in {elapsed:.1f}h. {len(result_df)} predictions.')

# Merge with PSProteinPredict scores
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, average_precision_score

psp = pd.read_csv(f'{COMPARE}/test_set_predictions_all_models.csv')
phyto = pd.read_csv(f'{COMPARE}/test_with_scores.csv')
phyto = phyto[['UniprotEntry', 'LLPS_Score']].drop_duplicates('UniprotEntry')

merged = psp.merge(phyto, on='UniprotEntry', how='left')
cols = ['UniprotEntry', 'true_label', 'LLPS_Score'] + \
       [c for c in merged.columns if c not in ['UniprotEntry', 'true_label', 'LLPS_Score']]
merged = merged[cols].round(6)

out = f'{COMPARE}/test_set_predictions_all_models_with_phyto.csv'
merged.to_csv(out, index=False)

valid = merged.dropna(subset=['LLPS_Score'])
y, p = valid['true_label'], valid['LLPS_Score']
f1 = f1_score(y, (p >= 0.5).astype(int))
auc = roc_auc_score(y, p)
auprc = average_precision_score(y, p)
acc = accuracy_score(y, (p >= 0.5).astype(int))

print(f'\nSaved: {out}')
print(f'Shape: {merged.shape}')
print(f'\nPhytoRNP_PSLM on test set:')
print(f'  Accuracy: {acc*100:.2f}%')
print(f'  F1:       {f1:.4f}')
print(f'  AUC-ROC:  {auc:.4f}')
print(f'  AUPRC:    {auprc:.4f}')
