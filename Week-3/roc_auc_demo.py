# roc_auc_demo.py
# Quick demo of ROC and AUC with printed values and a plot.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix

rng = np.random.default_rng(42)
y_true = np.array([1]*10 + [0]*20)
pos_scores = rng.normal(loc=0.75, scale=0.12, size=10).clip(0,1)
neg_scores = rng.normal(loc=0.30, scale=0.12, size=20).clip(0,1)
y_scores = np.r_[pos_scores, neg_scores]

fpr, tpr, thresh = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

rows = []
P = (y_true == 1).sum()
N = (y_true == 0).sum()
for t in np.r_[np.inf, thresh, -np.inf]:
    y_pred = (y_scores >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    TPR = tp / P if P else 0.0
    FPR = fp / N if N else 0.0
    rows.append((t, tp, fp, tn, fn, TPR, FPR))

df = pd.DataFrame(rows, columns=["threshold", "TP", "FP", "TN", "FN", "TPR", "FPR"]).drop_duplicates(subset=["threshold"]).sort_values("threshold", ascending=False)
print("\nROC Threshold Table (head):")
print(df.round(3).head(10).to_string(index=False))

plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, marker="o", label=f"ROC (AUC={roc_auc:.3f})")
plt.plot([0,1],[0,1], linestyle="--", label="Random")
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.title("ROC Curve Demo")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

