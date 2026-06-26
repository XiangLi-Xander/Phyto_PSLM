"""Train SVM for RNA dependency using PhytoRNP 512d pooled features."""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, joblib
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score, accuracy_score

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, HERE)
from src.utils import PROJECT_ROOT

FEAT_CACHE = os.path.join(PROJECT_ROOT, "outputs", "benchmarks", "phyto_512d_features.npz")
MODEL_DIR = os.path.join(PROJECT_ROOT, "outputs", "models")

d = np.load(FEAT_CACHE)
X = np.concatenate([d['X_train'], d['X_val']]); y = np.concatenate([d['y_train'], d['y_val']])
Xt, yt = d['X_test'], d['y_test']
sc = StandardScaler(); Xs = sc.fit_transform(X).astype(np.float32); Xts = sc.transform(Xt).astype(np.float32)

m = SVC(kernel='rbf', C=10, gamma=0.01, probability=True)
m.fit(Xs, y); pr = m.predict_proba(Xts)[:, 1]; p = m.predict(Xts)
print(f"SVM: F1={f1_score(yt,p):.4f} AUC={roc_auc_score(yt,pr):.4f} "
      f"AUPR={average_precision_score(yt,pr):.4f} Acc={accuracy_score(yt,p):.4f}")
joblib.dump({"model": m, "scaler": sc}, os.path.join(MODEL_DIR, "best_rna_svm_model.pkl"))
