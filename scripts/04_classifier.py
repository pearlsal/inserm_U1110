"""04 - (Optional, 'AI is a plus') L1-regularised (lasso) logistic regression: mild vs advanced fibrosis.

n=72, one cohort: this estimates separability, it is NOT a biomarker. Feature selection sits INSIDE the
pipeline so each outer fold selects genes on its training data only (no label leakage). The AUC is compared
with a permutation baseline produced by the identical pipeline.
"""
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_predict, permutation_test_score
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROC, FIG, TAB = Path("data/processed"), Path("results/figures"), Path("results/tables")
ex = pd.read_csv(PROC / "GSE49541_gene_expr.csv.gz", index_col=0)
stage = pd.read_csv(PROC / "GSE49541_stage.csv", index_col=0)["stage"].reindex(ex.columns)
X, y = ex.T.values, (stage == "advanced").astype(int).values
genes = ex.index.values

pipe = Pipeline([("sel", SelectKBest(f_classif, k=min(1000, X.shape[1]))), ("sc", StandardScaler()),
                 ("lr", LogisticRegression(penalty="l1", solver="liblinear", max_iter=2000, random_state=0))])
inner = GridSearchCV(pipe, {"lr__C": [0.05, 0.2, 1.0]}, scoring="roc_auc",
                     cv=StratifiedKFold(5, shuffle=True, random_state=1))
outer = StratifiedKFold(5, shuffle=True, random_state=0)

proba = cross_val_predict(inner, X, y, cv=outer, method="predict_proba")[:, 1]
auc = roc_auc_score(y, proba)
rng = np.random.default_rng(0)
boot = [roc_auc_score(y[i], proba[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(2000)) if len(set(y[i])) == 2]

# Gene stability across outer folds
counts = pd.Series(0, index=genes)
for tr, _ in outer.split(X, y):
    best = inner.fit(X[tr], y[tr]).best_estimator_
    keep = genes[best.named_steps["sel"].get_support()]
    counts[keep[best.named_steps["lr"].coef_[0] != 0]] += 1
counts[counts > 0].sort_values(ascending=False).rename("n_outer_folds_selected").to_csv(TAB / "04_selected_genes_stability.csv")

_, perm, p_perm = permutation_test_score(inner, X, y, cv=outer, scoring="roc_auc", n_permutations=20, n_jobs=-1, random_state=0)
summary = [f"Nested-CV AUC = {auc:.3f} (bootstrap 95% CI {np.percentile(boot, 2.5):.3f}-{np.percentile(boot, 97.5):.3f})",
           f"Permutation AUC: median {np.median(perm):.3f}, 95th pct {np.percentile(perm, 95):.3f}, p = {p_perm:.3f} (20 perms)",
           f"CLDN1 selected in {int(counts.get('CLDN1', 0))}/5 outer folds",
           f"Genes selected in >=3/5 folds: {int((counts >= 3).sum())}"]
(TAB / "04_classifier_summary.txt").write_text("\n".join(summary)); print("\n".join(summary))

fpr, tpr, _ = roc_curve(y, proba)
fig, ax = plt.subplots(figsize=(4.5, 4.5))
ax.plot(fpr, tpr, lw=2); ax.plot([0, 1], [0, 1], "k--", lw=1)
ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate"); ax.set_title(f"Nested-CV ROC, AUC = {auc:.2f}")
fig.savefig(FIG / "04_roc.png", dpi=200, bbox_inches="tight"); plt.close(fig)
print("04 done.")
