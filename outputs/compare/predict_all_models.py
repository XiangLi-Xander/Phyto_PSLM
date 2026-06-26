"""Predict all PSProteinPredict models on PhytoRNP_PSLM test set.
Model parameters match Figure1-2_LGB_PU_code1-6.py exactly."""

import numpy as np
import pandas as pd
import sys, os, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/xiaojian/LLPS_model_project/PSProteinProject')

from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, average_precision_score
from baggingPU import BaggingClassifierPU

BASE = '/home/xiaojian/LLPS_model_project/PhytoRNP_PSLM'
COMPARE = f'{BASE}/outputs/compare'
os.makedirs(COMPARE, exist_ok=True)

# -------------------------------------------------------------------
# 1. Data
# -------------------------------------------------------------------
test_df = pd.read_csv(f'{BASE}/data/processed/test.csv')
train_df = pd.read_csv(f'{BASE}/data/processed/train.csv')

pos_feat = pd.read_csv(f'{BASE}/all_species_llps_sequences.csv')
neg_feat = pd.read_csv(f'{BASE}/all_species_no_llps_merged.csv')

FEAT_COLS = ['Hydropathy', 'FCR', 'IDR', 'LCR', 'Pscore', 'PLAAC', 'catGRANULE']
pos_feat = pos_feat.set_index('UniprotEntry')[FEAT_COLS + ['Score']]
neg_feat = neg_feat.set_index('UniprotEntry')[FEAT_COLS + ['Score']]
all_feat = pd.concat([pos_feat, neg_feat])
all_feat = all_feat[~all_feat.index.duplicated(keep='first')]

def get_Xy(id_list):
    avail = [uid for uid in id_list if uid in all_feat.index]
    df = all_feat.loc[avail]
    return df[FEAT_COLS].fillna(0).values, df['Score'].values.astype(int), avail

X_train, y_train, train_ids = get_Xy(train_df['UniprotEntry'].tolist())
X_test,  y_test,  test_ids  = get_Xy(test_df['UniprotEntry'].tolist())

print(f'Train: {len(y_train)} ({y_train.sum()} pos)')
print(f'Test:  {len(y_test)} ({y_test.sum()} pos)')

# -------------------------------------------------------------------
# 2. Models - exact params from Figure1-2_LGB_PU_code1-6.py
# -------------------------------------------------------------------
def get_classifier(name):
    if name == "SVM":
        from sklearn.svm import SVC
        return SVC(kernel='rbf', gamma='auto', random_state=0)
    elif name == "DT":
        from sklearn.tree import DecisionTreeClassifier
        return DecisionTreeClassifier()
    elif name == "RF":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=10, random_state=0)
    elif name == "LGB":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(objective='binary', boosting_type='dart')
    elif name == "NB":
        from sklearn.naive_bayes import GaussianNB
        return GaussianNB()
    elif name == "MLP":
        from sklearn.neural_network import MLPClassifier
        return MLPClassifier(hidden_layer_sizes=(100,), activation='logistic', solver='adam')
    raise ValueError(name)

MODEL_NAMES = ['RF', 'LGB', 'DT', 'SVM', 'NB', 'MLP']

result_df = pd.DataFrame({'UniprotEntry': test_ids, 'true_label': y_test})

# Standard models
for mname in MODEL_NAMES:
    print(f'  {mname}...', end=' ')
    clf = get_classifier(mname)
    if mname == 'SVM':  # O(n^2), use 20000 subset
        rng = np.random.RandomState(2025)
        idx = rng.choice(len(X_train), min(20000, len(X_train)), replace=False)
        clf.fit(X_train[idx], y_train[idx])
    else:
        clf.fit(X_train, y_train)

    # SVM without probability=True -> use decision_function for ranking
    if mname == 'SVM':
        score = clf.decision_function(X_test)
        # Normalize to 0-1 for comparability
        score = (score - score.min()) / (score.max() - score.min() + 1e-10)
    else:
        score = clf.predict_proba(X_test)[:, 1]

    result_df[f'{mname}_score'] = score
    auc = roc_auc_score(y_test, score)
    print(f'AUC={auc:.4f}')

# PU Learning models (default params: n_estimators=10)
for mname in MODEL_NAMES:
    print(f'  PU+{mname}...', end=' ')
    try:
        base = get_classifier(mname)
        pu = BaggingClassifierPU(base_estimator=base)
        if mname == 'SVM':
            rng = np.random.RandomState(2025)
            idx = rng.choice(len(X_train), min(20000, len(X_train)), replace=False)
            pu.fit(X_train[idx], y_train[idx])
        else:
            pu.fit(X_train, y_train)
        score = pu.predict_proba(X_test)[:, 1]
        result_df[f'PU+{mname}_score'] = score
        auc = roc_auc_score(y_test, score)
        print(f'AUC={auc:.4f}')
    except Exception as e:
        print(f'SKIPPED ({e})')

# -------------------------------------------------------------------
# 3. Save
# -------------------------------------------------------------------
col_order = ['UniprotEntry', 'true_label'] + \
            sorted([c for c in result_df.columns if c not in ['UniprotEntry', 'true_label']])
result_df = result_df[col_order].round(6)

out = f'{COMPARE}/test_set_predictions_all_models.csv'
result_df.to_csv(out, index=False)
print(f'\nSaved: {out}')
print(f'Shape: {result_df.shape}')
print(f'Columns: {list(result_df.columns)}')
