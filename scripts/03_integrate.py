"""03 - Integration: score atlas-derived cell-type signatures in bulk fibrosis biopsies.

Question: which cell-type programmes shift with fibrosis, and is the bulk CLDN1 change explained by
composition (e.g. ductular reaction / stellate expansion) rather than per-cell upregulation?
Score = mean z-score of signature genes per sample (a simple, transparent ssGSEA-like proxy).
This is NOT formal deconvolution - stated as a limitation.
"""
import json, re
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
import statsmodels.formula.api as smf
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROC, FIG, TAB = Path("data/processed"), Path("results/figures"), Path("results/tables")
sigs = json.load(open(PROC / "celltype_signatures.json"))
ex = pd.read_csv(PROC / "GSE49541_gene_expr.csv.gz", index_col=0)
stage = pd.read_csv(PROC / "GSE49541_stage.csv", index_col=0)["stage"].reindex(ex.columns)

z = ex.sub(ex.mean(1), axis=0).div(ex.std(1).replace(0, np.nan), axis=0)
scores, sizes = {}, {}
for ct, genes in sigs.items():
    g = [x for x in genes if x in z.index and x != "CLDN1"]   # exclude CLDN1 itself to avoid circularity
    sizes[ct] = len(g)
    if len(g) >= 10: scores[ct] = z.loc[g].mean()
print("Signature genes found on array:", sizes)
S = pd.DataFrame(scores)                                      # samples x cell types
S.to_csv(TAB / "03_celltype_scores.csv")

rows = []
for ct in S:
    a, m = S.loc[(stage == "advanced").values, ct], S.loc[(stage == "mild").values, ct]
    t, p = stats.ttest_ind(a, m, equal_var=False)
    rows.append({"cell_type": ct, "n_genes": sizes[ct], "delta_advanced_minus_mild": a.mean() - m.mean(), "t": t, "pval": p})
res = pd.DataFrame(rows); res["fdr"] = multipletests(res.pval, method="fdr_bh")[1]
res.sort_values("t", ascending=False).to_csv(TAB / "03_celltype_score_by_stage.csv", index=False)
print(res.round(4))

n = len(S.columns); ncol = min(4, n); nrow = int(np.ceil(n / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(3 * ncol, 2.8 * nrow), squeeze=False)
for ax, ct in zip(axes.flat, S.columns):
    ax.boxplot([S.loc[(stage == s).values, ct] for s in ("mild", "advanced")], showfliers=False)
    ax.set_xticks([1, 2], ["mild", "adv."]); ax.set_title(ct, fontsize=9)
for ax in list(axes.flat)[n:]: ax.axis("off")
fig.suptitle("Atlas-derived cell-type signature scores in NAFLD biopsies"); fig.tight_layout()
fig.savefig(FIG / "03_celltype_scores_by_stage.png", dpi=200); plt.close(fig)

if "CLDN1" in ex.index:
    c = ex.loc["CLDN1"]
    cor = pd.DataFrame([dict(zip(["cell_type", "rho", "pval"], [ct, *stats.spearmanr(S[ct], c)])) for ct in S])
    cor["fdr"] = multipletests(cor.pval, method="fdr_bh")[1]
    cor.sort_values("rho", ascending=False).to_csv(TAB / "03_CLDN1_vs_celltype_spearman.csv", index=False)
    print(cor.round(4))

    df = S.copy(); df.columns = [re.sub(r"\W", "_", x) for x in df.columns]
    df["CLDN1"] = c.values; df["advanced"] = (stage == "advanced").astype(int).values
    adj = [x for x in ("Cholangiocyte", "Stellate_Fibro") if x in df]
    m0 = smf.ols("CLDN1 ~ advanced", df).fit()
    m1 = smf.ols("CLDN1 ~ advanced" + "".join(f" + {a}" for a in adj), df).fit() if adj else m0
    out = pd.DataFrame({"model": ["unadjusted", "adjusted: " + ",".join(adj)],
                        "stage_coef": [m0.params["advanced"], m1.params["advanced"]],
                        "stage_pval": [m0.pvalues["advanced"], m1.pvalues["advanced"]],
                        "r2": [m0.rsquared, m1.rsquared]})
    out.to_csv(TAB / "03_CLDN1_composition_adjusted.csv", index=False); print(out.round(4))
    (TAB / "03_CLDN1_models_full.txt").write_text(str(m0.summary()) + "\n\n" + str(m1.summary()))

    # Collinearity of the adjusted model: covariate-covariate correlation, covariate-stage correlation, VIFs
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    col = []
    if len(adj) == 2:
        for name, fn in [("spearman", stats.spearmanr), ("pearson", stats.pearsonr)]:
            r = fn(df[adj[0]], df[adj[1]])
            col.append({"metric": f"{name} r: {adj[0]} vs {adj[1]} score", "value": r[0], "pval": r[1]})
    for a_ in adj:
        r = stats.pearsonr(df[a_], df["advanced"])
        col.append({"metric": f"point-biserial r: {a_} score vs advanced stage", "value": r[0], "pval": r[1]})
    X, names = m1.model.exog, m1.model.exog_names
    for i, nm in enumerate(names):
        if nm != "Intercept":
            col.append({"metric": f"VIF (adjusted model): {nm}", "value": variance_inflation_factor(X, i), "pval": np.nan})
    col = pd.DataFrame(col); col.to_csv(TAB / "03_collinearity.csv", index=False); print(col.round(4).to_string(index=False))
print("03 done.")
