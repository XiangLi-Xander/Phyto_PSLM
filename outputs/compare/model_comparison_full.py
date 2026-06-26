"""
Comprehensive model comparison:
  PSProteinPredict models (RF, LGB, DT, SVM, NB, MLP + PU versions)
  vs PhytoRNP_PSLM (ESM2 + IUPred3 + Transformer)

Outputs figures and metrics to outputs/compare/
"""

import numpy as np
import pandas as pd
import sys, os, warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.metrics import (roc_auc_score, roc_curve, accuracy_score, f1_score,
                             precision_score, recall_score, confusion_matrix,
                             average_precision_score, precision_recall_curve)
from sklearn.utils import resample

BASE = '/home/xiaojian/LLPS_model_project/PhytoRNP_PSLM'
COMPARE = f'{BASE}/outputs/compare'
PSPROJECT = '/home/xiaojian/LLPS_model_project/PSProteinProject'
sys.path.insert(0, PSPROJECT)
os.makedirs(COMPARE, exist_ok=True)

# ============================================================
# 0. Model definitions
# ============================================================
def get_classifier(name, subsample=False):
    if name == 'RF':
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1)
    elif name == 'LGB':
        from lightgbm import LGBMClassifier
        return LGBMClassifier(objective='binary', boosting_type='dart', verbose=-1, random_state=2025)
    elif name == 'DT':
        from sklearn.tree import DecisionTreeClassifier
        return DecisionTreeClassifier(random_state=0)
    elif name == 'SVM':
        from sklearn.svm import SVC
        return SVC(kernel='rbf', gamma='auto', probability=True, random_state=0, max_iter=5000)
    elif name == 'NB':
        from sklearn.naive_bayes import GaussianNB
        return GaussianNB()
    elif name == 'MLP':
        from sklearn.neural_network import MLPClassifier
        return MLPClassifier(hidden_layer_sizes=(100,), activation='logistic',
                             solver='adam', max_iter=500, random_state=0)
    raise ValueError(name)

MODEL_NAMES = ['RF', 'LGB', 'DT', 'SVM', 'NB', 'MLP']
MODEL_COLORS = {'RF': '#e41a1c', 'LGB': '#377eb8', 'DT': '#4daf4a', 'SVM': '#984ea3',
                'NB': '#ff7f00', 'MLP': '#a65628', 'PhytoRNP_PSLM': '#000000'}

# ============================================================
# 1. Load data splits
# ============================================================
print('=' * 70)
print('Loading PhytoRNP_PSLM data splits...')
train_df = pd.read_csv(f'{BASE}/data/processed/train.csv')
val_df   = pd.read_csv(f'{BASE}/data/processed/val.csv')
test_df  = pd.read_csv(f'{BASE}/data/processed/test.csv')
pos_feat = pd.read_csv(f'{BASE}/all_species_llps_sequences.csv')
neg_feat = pd.read_csv(f'{BASE}/all_species_no_llps_merged.csv')
FEAT_COLS = ['Hydropathy', 'FCR', 'IDR', 'LCR', 'Pscore', 'PLAAC', 'catGRANULE']

pos_feat = pos_feat.set_index('UniprotEntry')[FEAT_COLS + ['Score']]
neg_feat = neg_feat.set_index('UniprotEntry')[FEAT_COLS + ['Score']]
all_feat = pd.concat([pos_feat, neg_feat])
all_feat = all_feat[~all_feat.index.duplicated(keep='first')]

def build_split(id_list):
    avail = [uid for uid in id_list if uid in all_feat.index]
    df = all_feat.loc[avail]
    return df[FEAT_COLS].fillna(0).values, df['Score'].values.astype(int)

X_train, y_train = build_split(train_df['UniprotEntry'].tolist())
X_val,   y_val   = build_split(val_df['UniprotEntry'].tolist())
X_test,  y_test  = build_split(test_df['UniprotEntry'].tolist())
X_train_full = np.vstack([X_train, X_val])
y_train_full = np.concatenate([y_train, y_val])

print(f'Train: {len(y_train)} ({y_train.sum()} pos / {(1-y_train).sum()} neg)')
print(f'Val:   {len(y_val)} ({y_val.sum()} pos / {(1-y_val).sum()} neg)')
print(f'Test:  {len(y_test)} ({y_test.sum()} pos / {(1-y_test).sum()} neg)')

# ============================================================
# 2. Train & evaluate PSProteinPredict models
# ============================================================
print('\n' + '=' * 70)
print('Training PSProteinPredict models...')
from baggingPU import BaggingClassifierPU

results = []

def evaluate(y_true, y_pred, y_prob, name):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return {
        'model': name,
        'accuracy': accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'auc_roc': roc_auc_score(y_true, y_prob),
        'auprc': average_precision_score(y_true, y_prob),
        'fpr': fpr,
        'y_true': y_true, 'y_prob': y_prob,
    }

# standard models
for mname in MODEL_NAMES:
    print(f'  {mname}...', end=' ')
    clf = get_classifier(mname)
    # SVM is O(n^2), use subset for SVM
    if mname == 'SVM':
        rng = np.random.RandomState(2025)
        idx = rng.choice(len(X_train), min(20000, len(X_train)), replace=False)
        X_sub, y_sub = X_train[idx], y_train[idx]
        clf.fit(X_sub, y_sub)
    else:
        clf.fit(X_train, y_train)
    y_prob = clf.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    res = evaluate(y_test, y_pred, y_prob, mname)
    results.append(res)
    print(f'AUC={res["auc_roc"]:.4f}  F1={res["f1"]:.4f}')

# PU models
for mname in MODEL_NAMES:
    print(f'  PU+{mname}...', end=' ')
    try:
        base = get_classifier(mname)
        pu = BaggingClassifierPU(base_estimator=base, n_estimators=10, random_state=2025)
        if mname == 'SVM':
            rng = np.random.RandomState(2025)
            idx = rng.choice(len(X_train), min(20000, len(X_train)), replace=False)
            pu.fit(X_train[idx], y_train[idx])
        else:
            pu.fit(X_train, y_train)
        y_prob = pu.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        res = evaluate(y_test, y_pred, y_prob, f'PU+{mname}')
        results.append(res)
        print(f'AUC={res["auc_roc"]:.4f}  F1={res["f1"]:.4f}')
    except Exception as e:
        print(f'SKIPPED ({e})')

# PhytoRNP_PSLM from published metrics (README)
results.append({
    'model': 'PhytoRNP_PSLM',
    'accuracy': 0.9466, 'f1': 0.616, 'precision': None, 'recall': None,
    'auc_roc': 0.950, 'auprc': 0.648, 'fpr': 0.0299,
    'y_true': None, 'y_prob': None,
})

# ============================================================
# 3. Figures
# ============================================================
print('\n' + '=' * 70)
print('Generating comparison figures...')

# Determine best model color from results
best_models = {'RF': None, 'LGB': None, 'DT': None, 'SVM': None, 'NB': None, 'MLP': None}
for r in results:
    name = r['model']
    base = name.replace('PU+', '')
    if base in best_models:
        if best_models[base] is None or (r['f1'] > best_models[base]['f1']):
            best_models[base] = r

# --- 3a. ROC curves (PSProteinPredict models only - PhytoRNP needs GPU) ---
fig, ax = plt.subplots(figsize=(9, 8))
ax.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.4, label='Random')

for mname in MODEL_NAMES:
    for r in results:
        if r['model'] == mname and r['y_true'] is not None:
            fpr, tpr, _ = roc_curve(r['y_true'], r['y_prob'])
            ax.plot(fpr, tpr, color=MODEL_COLORS[mname], lw=1.5,
                    label=f'{mname} (AUC={r["auc_roc"]:.3f})')
        elif r['model'] == f'PU+{mname}' and r['y_true'] is not None:
            fpr, tpr, _ = roc_curve(r['y_true'], r['y_prob'])
            ax.plot(fpr, tpr, color=MODEL_COLORS[mname], lw=1.0, linestyle=':', alpha=0.7,
                    label=f'PU+{mname} (AUC={r["auc_roc"]:.3f})')

ax.axhline(0.95, color='black', lw=2.5, linestyle='--', alpha=0.8,
           label=f'PhytoRNP_PSLM AUC=0.950 (from README)')
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.02])
ax.set_xlabel('False Positive Rate', fontsize=13)
ax.set_ylabel('True Positive Rate', fontsize=13)
ax.set_title('PSProteinPredict Models - ROC Curves\n(PhytoRNP_PSLM Test Set)', fontsize=14)
ax.legend(loc='lower right', fontsize=8, ncol=2)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{COMPARE}/roc_curves_comparison.png', dpi=200)
plt.close()

# --- 3b. PR curves ---
fig, ax = plt.subplots(figsize=(9, 8))
baseline = y_test.sum() / len(y_test)
ax.axhline(baseline, color='gray', linestyle='--', lw=1.5, alpha=0.5,
           label=f'Baseline ({baseline:.3f})')

for mname in MODEL_NAMES:
    for r in results:
        if r['model'] == mname and r['y_true'] is not None:
            prec, rec, _ = precision_recall_curve(r['y_true'], r['y_prob'])
            ax.plot(rec, prec, color=MODEL_COLORS[mname], lw=1.5,
                    label=f'{mname} (AUPRC={r["auprc"]:.3f})')
        elif r['model'] == f'PU+{mname}' and r['y_true'] is not None:
            prec, rec, _ = precision_recall_curve(r['y_true'], r['y_prob'])
            ax.plot(rec, prec, color=MODEL_COLORS[mname], lw=1.0, linestyle=':', alpha=0.7,
                    label=f'PU+{mname} (AUPRC={r["auprc"]:.3f})')

ax.axhline(0.07, color='black', lw=2.5, linestyle='--', alpha=0.8,
           label=f'PhytoRNP_PSLM AUPRC=0.648 (from README)')
ax.set_xlabel('Recall', fontsize=13)
ax.set_ylabel('Precision', fontsize=13)
ax.set_title('PSProteinPredict Models - PR Curves\n(PhytoRNP_PSLM Test Set)', fontsize=14)
ax.legend(loc='lower left', fontsize=8, ncol=2)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{COMPARE}/pr_curves_comparison.png', dpi=200)
plt.close()

# --- 3c. Metric bar chart: Standard vs PU vs PhytoRNP_PSLM ---
standards = [r for r in results if r['model'] in MODEL_NAMES and r['y_true'] is not None]
pu_models = [r for r in results if r['model'].startswith('PU+') and r['y_true'] is not None]
phyto_r = [r for r in results if r['model'] == 'PhytoRNP_PSLM']

metric_names = ['accuracy', 'f1', 'auc_roc', 'auprc', 'fpr']
metric_labels = ['Accuracy', 'F1-score', 'AUC-ROC', 'AUPRC', 'FPR']

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for idx, (mname, mlabel) in enumerate(zip(metric_names, metric_labels)):
    ax = axes[idx]
    x = np.arange(len(MODEL_NAMES))
    width = 0.25

    std_vals = [r[mname] for r in standards if r['model'] in MODEL_NAMES]
    pu_vals = []
    for m in MODEL_NAMES:
        pu_r = [r for r in pu_models if r['model'] == f'PU+{m}']
        pu_vals.append(pu_r[0][mname] if pu_r else 0)

    phyto_val = phyto_r[0][mname] if phyto_r and phyto_r[0][mname] is not None else 0

    bars1 = ax.bar(x - width, std_vals, width, label='Standard', color='steelblue', alpha=0.8)
    bars2 = ax.bar(x, pu_vals, width, label='PU Learning', color='coral', alpha=0.8)
    ax.axhline(phyto_val, color='black', linewidth=2.5, linestyle='-',
               label=f'PhytoRNP_PSLM = {phyto_val:.4f}')

    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_NAMES, fontsize=11)
    ax.set_ylabel(mlabel, fontsize=12)
    ax.set_title(mlabel, fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2, axis='y')

    for bar, val in zip(bars1, std_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:.3f}', ha='center', va='bottom', fontsize=7, rotation=45)
    for bar, val in zip(bars2, pu_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:.3f}', ha='center', va='bottom', fontsize=7, rotation=45)

axes[-1].axis('off')
plt.suptitle('Model Performance Comparison\n(PhytoRNP_PSLM Held-out Test Set)', fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig(f'{COMPARE}/metrics_comparison.png', dpi=200)
plt.close()

# --- 3d. Key models bar chart ---
fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(4)
width = 0.25
plot_metrics = ['accuracy', 'f1', 'auc_roc', 'auprc']
plot_labels = ['Accuracy', 'F1-score', 'AUC-ROC', 'AUPRC']
key_models = ['RF', 'LGB', 'PU+LGB', 'PhytoRNP_PSLM']
key_colors = ['#e41a1c', '#377eb8', '#ff7f00', '#000000']

for i, mname in enumerate(key_models):
    r = [res for res in results if res['model'] == mname]
    if r and r[0][plot_metrics[0]] is not None:
        vals = [r[0][m] for m in plot_metrics]
    elif mname == 'PhytoRNP_PSLM':
        vals = [0.9466, 0.616, 0.950, 0.648]
    else:
        continue
    ax.bar(x + i * width, vals, width, label=mname, color=key_colors[i], alpha=0.85)

ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(plot_labels, fontsize=11)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Key Models Comparison\n(PhytoRNP_PSLM Test Set)', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.2, axis='y')
plt.tight_layout()
plt.savefig(f'{COMPARE}/key_models_comparison.png', dpi=200)
plt.close()

# ============================================================
# 4. Save metrics
# ============================================================
dsp_cols = ['model', 'accuracy', 'f1', 'auc_roc', 'auprc', 'fpr', 'precision', 'recall']
results_df = pd.DataFrame([{k: r.get(k) for k in dsp_cols} for r in results])
results_df = results_df.round(4)
results_df.to_csv(f'{COMPARE}/all_model_metrics.csv', index=False)

print('\n' + '=' * 90)
print(f'{"Model":<20s} {"Acc":>8s} {"F1":>8s} {"AUC-ROC":>8s} {"AUPRC":>8s} {"FPR":>8s} {"Prec":>8s} {"Rec":>8s}')
print('-' * 90)
for _, row in results_df.iterrows():
    def fmt(v):
        if pd.isna(v) or v is None:
            return 'N/A'
        if isinstance(v, float) and v < 1:
            if row['model'] == 'PhytoRNP_PSLM' and v in [0.9466]:
                return f'{v*100:.2f}%'
            return f'{v*100:.2f}%' if v > 0.1 else f'{v:.4f}'
        return str(v)
    print(f'{row["model"]:<20s} '
          f'{fmt(row["accuracy"]) if row["accuracy"] is not None else "N/A":>8s} '
          f'{fmt(row["f1"]) if row["f1"] is not None else "N/A":>8s} '
          f'{fmt(row["auc_roc"]) if row["auc_roc"] is not None else "N/A":>8s} '
          f'{fmt(row["auprc"]) if row["auprc"] is not None else "N/A":>8s} '
          f'{fmt(row["fpr"]) if row["fpr"] is not None else "N/A":>8s} '
          f'{fmt(row["precision"]) if row["precision"] is not None else "N/A":>8s} '
          f'{fmt(row["recall"]) if row["recall"] is not None else "N/A":>8s}')
print('-' * 90)
print(f'\nOutputs saved to: {COMPARE}/')
print('Done!')
print()
print('NOTE: PhytoRNP_PSLM inference on the full test set (~13833 sequences)')
print('requires ~3 hours of GPU time (ESM2_t33_650M forward pass).')
print('Metrics from published README results are used instead.')
print(f'To run full inference: cd {BASE} && python -m src.predict')
