"""02 - Advanced (F3-4) vs mild (F0-1) NAFLD fibrosis: GSE49541 (Moylan et al., Hepatology 2014).

A REPLICATION test of Roehlen et al. 2022: is CLDN1 higher in advanced fibrosis in a cohort the lab did not generate?
Positive controls (collagens, LUM) are reported first: if they fail, the comparison is not trusted.
"""
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from sklearn.decomposition import PCA
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

RAW, PROC = Path("data/raw/GSE49541"), Path("data/processed")
FIG, TAB = Path("results/figures"), Path("results/tables")
for d in (RAW, PROC, FIG, TAB): d.mkdir(parents=True, exist_ok=True)
EXPR_F, META_F = PROC / "GSE49541_gene_expr.csv.gz", PROC / "GSE49541_stage.csv"

# ---- 1. Download + probe->gene (cached) -----------------------------------
if not EXPR_F.exists():
    import GEOparse
    gse = GEOparse.get_GEO("GSE49541", destdir=str(RAW), silent=True)
    ph = gse.phenotype_data
    col = [c for c in ph.columns if ph[c].astype(str).str.contains("advanced", case=False).any()][0]
    print("Stage column:", col, "| values:", ph[col].unique())
    stage = pd.Series(np.where(ph[col].str.contains("advanced", case=False), "advanced", "mild"), index=ph.index)
    ex = gse.pivot_samples("VALUE")[ph.index].astype(float)
    if np.nanmax(ex.values) > 50:
        print("Values look linear -> log2(x+1)"); ex = np.log2(ex + 1)
    ann = list(gse.gpls.values())[0].table.set_index("ID")["Gene Symbol"].astype(str).str.split(" /// ").str[0]
    ex["gene"] = ann.reindex(ex.index).values
    ex = ex[ex.gene.notna() & (ex.gene != "nan")]
    # keep the most highly expressed probe per gene
    ex["mean"] = ex.drop(columns="gene").mean(axis=1)
    ex = ex.sort_values("mean", ascending=False).drop_duplicates("gene").set_index("gene").drop(columns="mean")
    ex.to_csv(EXPR_F); stage.rename("stage").to_csv(META_F)

ex = pd.read_csv(EXPR_F, index_col=0)
stage = pd.read_csv(META_F, index_col=0)["stage"].reindex(ex.columns)
print(stage.value_counts().to_dict(), "| genes:", ex.shape[0])

# ---- 2. QC: distributions + PCA -------------------------------------------
fig, ax = plt.subplots(figsize=(14, 4))
ax.boxplot(ex.values, showfliers=False)
ax.set_title("GSE49541 per-sample expression distributions"); ax.set_xticks([])
fig.savefig(FIG / "02_sample_boxplot.png", dpi=200, bbox_inches="tight"); plt.close(fig)

top = ex.loc[ex.var(axis=1).nlargest(2000).index]
z = ((top.T - top.T.mean()) / top.T.std()).fillna(0)
pcs = PCA(n_components=2, random_state=0).fit(z)
xy = pcs.transform(z)
scores = pd.DataFrame(xy, index=ex.columns, columns=["PC1", "PC2"]).assign(stage=stage.values)
scores.rename_axis("sample").to_csv(TAB / "02_pca_scores.csv")
pc_tests = []
for j, pc in enumerate(["PC1", "PC2"]):
    a_, m_ = scores.loc[scores.stage == "advanced", pc], scores.loc[scores.stage == "mild", pc]
    tt = stats.ttest_ind(a_, m_, equal_var=False)
    pc_tests.append({"pc": pc, "var_explained_pct": round(100 * pcs.explained_variance_ratio_[j], 1),
                     "mean_advanced": a_.mean(), "mean_mild": m_.mean(), "welch_t": tt.statistic, "pval": tt.pvalue,
                     "mannwhitney_pval": stats.mannwhitneyu(a_, m_).pvalue})
pc_tests = pd.DataFrame(pc_tests); pc_tests.to_csv(TAB / "02_pca_stage_tests.csv", index=False)
print(pc_tests.round(4).to_string(index=False))
fig, ax = plt.subplots(figsize=(5.5, 4.5))
for s, c in [("mild", "tab:blue"), ("advanced", "tab:red")]:
    m = (stage == s).values; ax.scatter(xy[m, 0], xy[m, 1], c=c, label=s, s=18)
ax.set_xlabel(f"PC1 ({pcs.explained_variance_ratio_[0]:.0%})"); ax.set_ylabel(f"PC2 ({pcs.explained_variance_ratio_[1]:.0%})")
ax.legend(); ax.set_title("GSE49541 PCA (top 2000 variable genes)")
fig.savefig(FIG / "02_pca.png", dpi=200, bbox_inches="tight"); plt.close(fig)

# ---- 3a. Independent filter (added after the first run exposed a volcano artefact; docs/DECISIONS.md B9) ---------
# Drop genes whose mean across ALL 72 samples (stage ignored) is below the 25th percentile of gene means. The filter
# never uses stage, so it is independent of the test statistic under the null (Bourgon et al., PNAS 2010). It removes
# array-floor genes (e.g. constant at the GC-RMA floor), whose Welch t is numerically unstable (catastrophic
# cancellation gave |t| = 5.57 for genes with SD 0). limma's moderated t addresses the same problem by variance shrinkage.
gene_mean = ex.mean(axis=1)
thr = gene_mean.quantile(0.25)
keep_g = gene_mean >= thr
ex_de = ex.loc[keep_g]
print(f"Independent filter: mean < {thr:.4f} (25th pct of gene means) -> removed {int((~keep_g).sum())} of "
      f"{len(ex)} genes; {len(ex_de)} tested")

# ---- 3. Differential expression (Welch t-test + BH) ------------------------
# limma's moderated t would be the R standard; with n=72 the two agree closely for ranking.
adv, mild = ex_de.loc[:, (stage == "advanced").values], ex_de.loc[:, (stage == "mild").values]
t, p = stats.ttest_ind(adv, mild, axis=1, equal_var=False)
de = pd.DataFrame({"logFC": adv.mean(1) - mild.mean(1), "t": t, "pval": p}, index=ex_de.index).dropna()
de["fdr"] = multipletests(de.pval, method="fdr_bh")[1]
de = de.sort_values("pval"); de.to_csv(TAB / "02_DE_advanced_vs_mild.csv")
spurious = de[(de.logFC.abs() < 0.02) & (de.pval < 1e-4)]
flt = pd.DataFrame([{
    "n_genes_before": len(ex), "threshold_mean_log2": thr, "n_removed": int((~keep_g).sum()), "n_tested": len(de),
    "n_constant_genes_removed": int((ex.loc[~keep_g].std(1) == 0).sum()),
    "n_constant_genes_tested": int((ex_de.std(1) == 0).sum()),
    "n_fdr_lt_0.05": int((de.fdr < 0.05).sum()), "n_up_fdr_lt_0.05": int(((de.fdr < 0.05) & (de.logFC > 0)).sum()),
    "n_down_fdr_lt_0.05": int(((de.fdr < 0.05) & (de.logFC < 0)).sum()),
    "n_spurious_absLogFC_lt_0.02_and_p_lt_1e-4": len(spurious)}])
flt.to_csv(TAB / "02_independent_filter.csv", index=False); print(flt.T.to_string(header=False))
if len(spurious): print("Remaining |logFC|<0.02 & p<1e-4:\n" + spurious.assign(sd=ex_de.loc[spurious.index].std(1)).round(5).to_string())

FOCUS = ["COL1A1", "COL1A2", "LUM", "TIMP1", "CLDN1", "EPCAM", "KRT7", "SOX9"]
de.reindex(FOCUS).to_csv(TAB / "02_focus_genes.csv")
print(de.reindex(FOCUS).round(4))

fig, ax = plt.subplots(figsize=(6.5, 5))
sig = de.fdr < 0.05
ax.scatter(de.logFC[~sig], -np.log10(de.pval[~sig]), s=3, c="grey", alpha=.5)
ax.scatter(de.logFC[sig], -np.log10(de.pval[sig]), s=3, c="firebrick", alpha=.6, label="FDR < 0.05")
for g in FOCUS:
    if g in de.index: ax.annotate(g, (de.logFC[g], -np.log10(de.pval[g])), fontsize=8)
ax.set_xlabel("log2 fold change (advanced - mild)"); ax.set_ylabel("-log10 p"); ax.legend()
ax.set_title(f"Advanced (F3-4) vs mild (F0-1) fibrosis, GSE49541\n{len(de)} genes after independent filter", fontsize=10)
fig.savefig(FIG / "02_volcano.png", dpi=200, bbox_inches="tight"); plt.close(fig)

if "CLDN1" in ex.index:
    fig, ax = plt.subplots(figsize=(3.8, 4.5))
    data = [ex.loc["CLDN1", (stage == s).values] for s in ("mild", "advanced")]
    ax.boxplot(data, showfliers=False); ax.set_xticks([1, 2], ["mild\nF0-1", "advanced\nF3-4"])
    rng = np.random.default_rng(0)                                   # seeded jitter (reproducible figure)
    for i, d in enumerate(data, 1): ax.scatter(rng.normal(i, .06, len(d)), d, s=8, c="k", alpha=.6)
    ax.set_ylabel("log2 expression (GC-RMA)")
    ax.set_title(f"CLDN1: advanced (F3-4, n={len(data[1])}) vs mild (F0-1, n={len(data[0])})\n"
                 f"log2FC={de.logFC['CLDN1']:.2f}, FDR={de.fdr['CLDN1']:.2g}", fontsize=9)
    fig.savefig(FIG / "02_CLDN1_by_stage.png", dpi=200, bbox_inches="tight"); plt.close(fig)

# ---- 3b. CLDN1 probe-level check (does NOT change the main analysis) -------------------------------------
# Main analysis keeps one probe per gene (highest mean). Here every GPL570 probe annotated to CLDN1 is tested with the
# same Welch test; FDR is BH across all probes passing the same independent filter as the gene-level DE
# (probe mean across all samples >= 25th percentile of probe means).
PROBE_F = PROC / "GSE49541_probe_expr.csv.gz"
if not PROBE_F.exists():
    import GEOparse
    gse = GEOparse.get_GEO("GSE49541", destdir=str(RAW), silent=True)
    pr = gse.pivot_samples("VALUE")[stage.index].astype(float)
    sym = list(gse.gpls.values())[0].table.set_index("ID")["Gene Symbol"].astype(str)
    pr.insert(0, "gene_symbol", sym.reindex(pr.index).values)
    pr.rename_axis("probe").to_csv(PROBE_F)
pr = pd.read_csv(PROBE_F, index_col=0)
sym_all = pr.pop("gene_symbol").astype(str)
pr = pr[stage.index]
pmean = pr.mean(axis=1); keep_p = pmean >= pmean.quantile(0.25)          # same independent filter, probe level
prf = pr.loc[keep_p]
print(f"Probe-level independent filter: removed {int((~keep_p).sum())} of {len(pr)} probes")
tp, pp = stats.ttest_ind(prf.loc[:, (stage == "advanced").values], prf.loc[:, (stage == "mild").values], axis=1, equal_var=False)
pde = pd.DataFrame({"gene_symbol": sym_all[keep_p], "mean_expr": prf.mean(1), "logFC": prf.loc[:, (stage == "advanced").values].mean(1)
                    - prf.loc[:, (stage == "mild").values].mean(1), "t": tp, "pval": pp}).dropna(subset=["pval"])
pde["fdr_filtered_probes"] = multipletests(pde.pval, method="fdr_bh")[1]
cl_pr = pde[pde.gene_symbol.str.split(" /// ").apply(lambda x: "CLDN1" in x)].copy()
first_is_cldn1 = cl_pr.gene_symbol.str.split(" /// ").str[0] == "CLDN1"
cl_pr["used_in_main_analysis"] = False
if first_is_cldn1.any():   # same rule as section 1: highest-mean probe among those whose first symbol is CLDN1
    cl_pr.loc[cl_pr[first_is_cldn1].mean_expr.idxmax(), "used_in_main_analysis"] = True
    used = cl_pr.index[cl_pr.used_in_main_analysis][0]
    assert np.allclose(pr.loc[used].values, ex.loc["CLDN1"].values), "probe check does not reproduce the main CLDN1 values"
cl_pr.sort_values("mean_expr", ascending=False).rename_axis("probe").to_csv(TAB / "02_CLDN1_probes.csv")
print(cl_pr.sort_values("mean_expr", ascending=False).round(4).to_string())

# ---- 3c. Pre-set CLDN1 probe check (main analysis stays on the pre-specified probe whatever this shows) ----------
def gpl_rows(soft_gz, ids):
    """Rows of the GPL570 platform table (as deposited in the GSE SOFT file) for the given probe IDs."""
    import gzip
    rows, hdr, on = [], None, False
    with gzip.open(soft_gz, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("!platform_table_begin"): on = True; continue
            if line.startswith("!platform_table_end"): break
            if on:
                f = line.rstrip("\n").split("\t")
                if hdr is None: hdr = f
                elif f[0] in ids: rows.append(f + [""] * (len(hdr) - len(f)))
    return pd.DataFrame(rows, columns=hdr).set_index("ID")

ids = list(cl_pr.sort_values("used_in_main_analysis", ascending=False).index)      # pre-specified probe first
gpl = gpl_rows(RAW / "GSE49541_family.soft.gz", set(ids))
all_means = pr.mean(1)
rho = stats.spearmanr(pr.loc[ids[0]], pr.loc[ids[1]]) if len(ids) == 2 else None
chk = []
for pid in ids:
    r = cl_pr.loc[pid]; sym = gpl.loc[pid, "Gene Symbol"]
    chk.append({"probe": pid, "role": "pre-specified (main analysis)" if r.used_in_main_analysis else "second probe",
                "gpl_gene_symbol": sym, "n_genes_annotated": len([x for x in sym.split(" /// ") if x]),
                "gpl_gene_title": gpl.loc[pid, "Gene Title"], "gpl_entrez_gene_id": gpl.loc[pid, "ENTREZ_GENE_ID"],
                "gpl_representative_public_id": gpl.loc[pid, "Representative Public ID"],
                "gpl_target_description": gpl.loc[pid, "Target Description"],
                "mean_expr": r.mean_expr, "mean_expr_percentile_all_probes": round(100 * (all_means <= r.mean_expr).mean(), 1),
                "logFC": r.logFC, "t": r.t, "pval": r.pval, "fdr_filtered_probes": r.fdr_filtered_probes,
                "spearman_rho_between_probes": rho.statistic if rho else np.nan,
                "spearman_p_between_probes": rho.pvalue if rho else np.nan})
# SENSITIVITY ANALYSIS (labelled; not the main result): stage test on the per-sample mean of the two probes
avg = pr.loc[ids].mean(0)
ta = stats.ttest_ind(avg[(stage == "advanced").values], avg[(stage == "mild").values], equal_var=False)
chk.append({"probe": "mean(" + ", ".join(ids) + ")", "role": "SENSITIVITY ANALYSIS: mean of the two probes (not the main result)",
            "mean_expr": avg.mean(), "logFC": avg[(stage == "advanced").values].mean() - avg[(stage == "mild").values].mean(),
            "t": ta.statistic, "pval": ta.pvalue})
chk = pd.DataFrame(chk); chk.to_csv(TAB / "02_CLDN1_probe_check.csv", index=False)
print(chk.drop(columns=["gpl_target_description"]).round(4).to_string(index=False))
for pid in ids: print(pid, "| GPL570 Target Description:", gpl.loc[pid, "Target Description"])

# ---- 4. Hallmark pre-ranked GSEA (needs internet for the gene-set library) --
try:
    import gseapy as gp
    pre = gp.prerank(rnk=de["t"].sort_values(ascending=False), gene_sets="MSigDB_Hallmark_2020",
                     permutation_num=1000, min_size=15, max_size=500, seed=1, outdir=None, threads=4, verbose=False)
    r = pre.res2d[["Term", "NES", "NOM p-val", "FDR q-val"]].sort_values("NES", ascending=False)
    r.to_csv(TAB / "02_gsea_hallmark.csv", index=False); print(r.head(10))
except Exception as e:
    print("GSEA skipped:", e)
print("02 done.")
