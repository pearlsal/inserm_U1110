"""01 - Re-analysis of the human liver cell atlas (Aizarani et al., Nature 2019; GSE124395).

Question: which liver cell types express CLDN1 (Roehlen et al., Sci Transl Med 2022)?
Cell types are annotated from canonical markers, independently of the paper, then cross-tabulated
against the authors' cluster partition as a sanity check.
"""
from pathlib import Path
import json, re
import numpy as np, pandas as pd, scipy.io, scipy.sparse as sp
import scanpy as sc
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

RAW, PROC = Path("data/raw/GSE124395"), Path("data/processed")
FIG, TAB = Path("results/figures"), Path("results/tables")
for d in (PROC, FIG, TAB): d.mkdir(parents=True, exist_ok=True)
sc.settings.figdir = FIG; sc.settings.verbosity = 1; sc.settings.set_figure_params(dpi_save=200)
SEED = 0

# ---- 1. Load ---------------------------------------------------------------
X = scipy.io.mmread(RAW / "counts.mtx").tocsr().T.tocsr()            # -> cells x genes
genes = pd.read_csv(RAW / "genes.txt", header=None)[0].astype(str).str.replace(r"__.*$", "", regex=True)
cells = pd.read_csv(RAW / "cells.txt", header=None)[0].astype(str)
print(f"Raw: {X.shape[0]} cells x {X.shape[1]} genes; example cell IDs: {list(cells[:3])}")

# RaceID gene names look like 'CLDN1__chr3'; after stripping, sum duplicated symbols.
codes, uniq = pd.factorize(genes)
G = sp.csr_matrix((np.ones(len(codes)), (np.arange(len(codes)), codes)), shape=(len(codes), len(uniq)))
adata = sc.AnnData(X=(X @ G).astype(np.float32).tocsr(), obs=pd.DataFrame(index=cells.values),
                   var=pd.DataFrame(index=pd.Index(uniq, dtype=str)))
# Drop ERCC spike-ins only ('ERCC-00002' style). The GEO matrix has none, but it does hold the real human genes
# ERCC1-ERCC8 / ERCC6L / BIVM-ERCC5, which a plain startswith("ERCC") would silently remove.
adata = adata[:, ~adata.var_names.str.match(r"^ERCC-\d+$")].copy()

# ---- Donor -----------------------------------------------------------------
# Verified on the real data: cell IDs are <plate>_<cell barcode>, and the first '_' token is often the FACS gate
# (e.g. 'ASGR1_CD45_UNB_P4_5_16', 'EPCAMTrop2high_2_17'), NOT the donor. Donor is taken from
#   (i) a leading 'P3xx' (or 'P3xx_3xx') token when the plate is named after the donor (e.g. 'P301_19_91'), else
#  (ii) the GEO sample record of that plate: GSM title 'P3xx_n' <-> plate name in the GSM supplementary file name.
# Plates with no donor on GEO (titles 'BP_1'..'BP_4' = CD45negASGR1negBio plates) are labelled 'unknown'.
def gsm_plate_to_donor(path):
    recs, cur = [], None
    for line in open(path, encoding="utf-8", errors="replace"):
        k, _, v = line.rstrip("\r\n").partition(" = ")
        if k == "^SAMPLE":
            cur = {}; recs.append(cur)
        elif cur is not None and k == "!Sample_title":
            cur["title"] = v
        elif cur is not None and k == "!Sample_supplementary_file_1":
            cur["plate"] = re.sub(r"^GSM\d+_|\.coutt\.csv\.gz$", "", v.rsplit("/", 1)[-1])
    g = pd.DataFrame(recs).dropna(subset=["title", "plate"])
    g["donor"] = g.title.str.extract(r"^(P\d{3})_\d+$")[0]
    return g

def sort_label(geo_plate):
    """FACS gate / plate type from the GEO plate name: 'P301_19_NP1_3_1_newref' -> 'NP1', 'EPCAMTrop2int_1' -> 'EPCAMTrop2int'."""
    x = re.sub(r"(_\d+)+(_newref)?$", "", geo_plate)
    return re.sub(r"^P\d{3}(_\d{3})?_\d+_", "", x)

plate = adata.obs_names.str.replace(r"_\d+$", "", regex=True)          # strip the cell-barcode number
lead = plate.str.extract(r"^(P\d{3}(?:_\d{3})?)_\d+$")[0]               # 'P301_19' -> P301, 'P311_313_5' -> P311_313
gsm = gsm_plate_to_donor(RAW / "GSE124395_gsm_brief.txt")
plate2donor = gsm.dropna(subset=["donor"]).set_index("plate")["donor"].to_dict()
# cell-ID plate key -> full GEO plate name (keys are either a GSM title, a GEO plate name, or 'P311_313_k' = plate prefix)
key2geo = {**dict(zip(gsm.title, gsm.plate)), **dict(zip(gsm.plate, gsm.plate)),
           **{m.group(1): pl for pl in gsm.plate if (m := re.match(r"^(P\d{3}_\d{3}_\d+)_", pl))}}
adata.obs["sort_label"] = [sort_label(key2geo.get(k, k)) for k in plate]
from_gsm = pd.Series(plate, index=adata.obs_names).map(plate2donor)
adata.obs["donor"] = (lead.fillna(pd.Series(from_gsm.values)).fillna("unknown")).values
adata.obs["donor_source"] = np.where(lead.notna(), "cell_id", np.where(from_gsm.notna().values, "geo_gsm", "none"))
(adata.obs.assign(plate_prefix=plate.str.replace(r"_\d+$", "", regex=True))
      .groupby(["plate_prefix", "donor", "donor_source"]).size().rename("n_cells_raw")
      .to_csv(TAB / "01_donor_mapping.csv"))
print("Donor assignment (raw cells):", adata.obs.donor.value_counts().to_dict(),
      "| source:", adata.obs.donor_source.value_counts().to_dict())

part_f = RAW / "GSE124395_clusterpartition.txt"
# The author partition is required: the main analysis is restricted to the cells the authors retained.
if not part_f.exists(): raise SystemExit(f"{part_f} missing - run scripts/00_download.sh")
# The GEO file starts with a one-token header line ('"sct@cpart"'), then '"cellID" cluster' rows.
one_token_header = len(open(part_f).readline().split()) == 1
part = pd.read_csv(part_f, sep=r"\s+", header=None, skiprows=int(one_token_header))
if not str(part.iloc[0, 1]).lstrip("-").isdigit():      # drop header row if present
    part = part.iloc[1:]
part = pd.Series(part.iloc[:, 1].values, index=part.iloc[:, 0].astype(str))
lab = part.reindex(adata.obs_names)                                       # NaN for cells not in the partition
adata.obs["author_cluster"] = np.where(lab.isna(), "nan", lab.astype("Int64").astype(str))   # "12", not "12.0"
# DECISION (docs/DECISIONS.md #D2): like-for-like with the published atlas -> keep only the cells in the author
# partition (the GEO matrix also holds cells the authors filtered out), THEN apply our own pre-declared QC.
in_part = adata.obs.author_cluster != "nan"
print(f"Author partition: {len(part)} cells listed; {int(in_part.sum())} found in matrix; "
      f"dropping {int((~in_part).sum())} matrix cells not in the partition")
adata = adata[in_part.values].copy()

# ---- 2. QC -----------------------------------------------------------------
# NOTE (verified): the GEO matrix contains no mitochondrially encoded genes (no MT-*/MTND*/MTCO*/MTATP*/MTCYB/MTRNR1-2
# symbols and no chrM annotation; MTRNR2L* are nuclear), so pct_counts_mt is 0 for every cell and the mito threshold
# below is inactive. Effective QC = the authors' own filtering (partition) + n_genes > 300. Kept for portability.
adata.var["mt"] = adata.var_names.str.startswith("MT-")
print("Mito genes found:", int(adata.var.mt.sum()))
sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True, percent_top=None, log1p=False)
sc.pl.violin(adata, ["n_genes_by_counts", "total_counts"], groupby="donor", rotation=90,   # no mito panel: all 0
             multi_panel=True, show=False, save="_01_qc.png")
n0 = adata.n_obs
before = adata.obs.donor.value_counts()
# Permissive, pre-declared thresholds: CEL-seq2 is shallow, and hepatocytes are mito-rich.
fail_genes, fail_mt = adata.obs.n_genes_by_counts <= 300, adata.obs.pct_counts_mt >= 40
keep = ~(fail_genes | fail_mt)
pd.DataFrame({"n_author_retained_cells": [n0], "removed_n_genes_le_300": [int(fail_genes.sum())],
              "removed_pct_mt_ge_40": [int(fail_mt.sum())], "removed_total": [int((~keep).sum())],
              "kept": [int(keep.sum())]}).to_csv(TAB / "01_qc_summary.csv", index=False)
(adata.obs.assign(removed=~keep).groupby("author_cluster")["removed"].agg(n_cells="size", n_removed="sum")
      .assign(pct_removed=lambda d: (100 * d.n_removed / d.n_cells).round(1))
      .sort_index(key=lambda s: pd.to_numeric(s, errors="coerce"))
      .to_csv(TAB / "01_qc_removed_by_author_cluster.csv"))
print(f"Our QC would remove {int((~keep).sum())}/{n0} author-retained cells "
      f"(n_genes<=300: {int(fail_genes.sum())}; pct_mt>=40: {int(fail_mt.sum())})")
adata = adata[keep.values].copy()
sc.pp.filter_genes(adata, min_cells=5)
qc = pd.DataFrame({"before_qc": before, "after_qc": adata.obs.donor.value_counts()}).fillna(0).astype(int)
qc["removed"] = qc.before_qc - qc.after_qc
print(qc.to_string())
qc.to_csv(TAB / "01_cells_per_donor_after_qc.csv")
print(f"QC: {n0} -> {adata.n_obs} cells")

# ---- 3. Normalise / cluster / embed ---------------------------------------
sc.pp.normalize_total(adata, target_sum=1e4); sc.pp.log1p(adata)
adata.raw = adata
sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor="seurat")
ad = adata[:, adata.var.highly_variable].copy()
sc.pp.scale(ad, max_value=10)
sc.tl.pca(ad, n_comps=30, random_state=SEED)
adata.obsm["X_pca"] = ad.obsm["X_pca"]
sc.pp.neighbors(adata, n_pcs=30, random_state=SEED)
sc.tl.leiden(adata, resolution=0.8, random_state=SEED, flavor="igraph", n_iterations=2, directed=False)
sc.tl.umap(adata, random_state=SEED)
print(f"Leiden (resolution 0.8): {adata.obs.leiden.nunique()} clusters")

# DECISION (docs/DECISIONS.md #D3): donor composition of each cluster; >80% from one donor = possibly donor-driven.
comp = pd.crosstab(adata.obs.leiden, adata.obs.donor, normalize="index").mul(100).round(1)
comp.insert(0, "n_cells", adata.obs.leiden.value_counts().reindex(comp.index).values)
comp["n_donors_present"] = (pd.crosstab(adata.obs.leiden, adata.obs.donor).drop(columns="unknown", errors="ignore") > 0).sum(1)
known = comp.drop(columns=["n_cells", "n_donors_present", "unknown"], errors="ignore")
comp["top_donor"], comp["top_donor_pct"] = known.idxmax(axis=1), known.max(axis=1)
comp["possibly_donor_driven"] = comp.top_donor_pct > 80
comp.to_csv(TAB / "01_cluster_donor_composition.csv")
print(comp[["n_cells", "n_donors_present", "top_donor", "top_donor_pct", "possibly_donor_driven"]].to_string())

# ---- 4. Marker-based annotation -------------------------------------------
MARKERS = {
    "Hepatocyte":       ["ALB", "APOA1", "CYP3A4", "TF"],
    "Cholangiocyte":    ["KRT19", "KRT7", "EPCAM", "SOX9"],
    "LSEC":             ["CLEC4G", "CLEC4M", "FCN3", "STAB2"],
    "Macrovascular_EC": ["VWF", "PECAM1", "AQP1"],
    "Kupffer_Macro":    ["CD68", "MARCO", "VSIG4", "CD163"],
    "Stellate_Fibro":   ["COL1A1", "DCN", "ACTA2", "PDGFRB", "LUM"],
    "T_NK":             ["CD3D", "NKG7", "GZMA", "IL7R"],
    "B_Plasma":         ["MS4A1", "CD79A", "JCHAIN", "MZB1"],
}
MARKERS = {k: [g for g in v if g in adata.raw.var_names] for k, v in MARKERS.items()}
MARKERS = {k: v for k, v in MARKERS.items() if v}
for ct, g in MARKERS.items():
    sc.tl.score_genes(adata, g, score_name=f"score_{ct}", random_state=SEED, use_raw=True)
score_cols = [f"score_{c}" for c in MARKERS]
cl = adata.obs.groupby("leiden", observed=True)[score_cols].mean()
srt = np.sort(cl.values, axis=1)
cl["cell_type"] = [c.replace("score_", "") for c in cl[score_cols].idxmax(axis=1)]
cl["margin_top1_vs_top2"] = srt[:, -1] - srt[:, -2]
cl["ambiguous"] = cl["margin_top1_vs_top2"] < 0.1          # flag, don't hide
cl = cl.join(comp[["n_cells", "top_donor", "top_donor_pct", "possibly_donor_driven"]])
cl.to_csv(TAB / "01_cluster_annotation_scores.csv")
adata.obs["cell_type"] = adata.obs.leiden.map(cl["cell_type"]).astype("category")
print(cl[["n_cells", "cell_type", "margin_top1_vs_top2", "ambiguous", "top_donor_pct", "possibly_donor_driven"]].round(3).to_string())

if "author_cluster" in adata.obs:
    ct_tab = pd.crosstab(adata.obs.author_cluster, adata.obs.cell_type)
    ct_tab = ct_tab.sort_index(key=lambda s: pd.to_numeric(s, errors="coerce"))      # 'nan' = cell not in author partition
    ct_tab.to_csv(TAB / "01_author_vs_our_annotation.csv"); print(ct_tab.to_string())

sc.pl.umap(adata, color=["cell_type", "donor"], wspace=0.4, show=False, save="_01_celltypes_donor.png",
           title=["Normal liver atlas: marker-based cell type", "Normal liver atlas: donor"])
sc.pl.dotplot(adata, {k: v[:3] for k, v in MARKERS.items()}, groupby="cell_type", show=False, save="01_markers.png")

# ---- 5. Target genes -------------------------------------------------------
TARGETS = [g for g in ["CLDN1", "OCLN", "CD81", "SCARB1", "EGFR", "COL1A1", "TIMP1", "TGFB1"]
           if g in adata.raw.var_names]
sc.pl.dotplot(adata, TARGETS, groupby="cell_type", show=False, save="01_targets.png")
sc.pl.umap(adata, color="CLDN1", use_raw=True, show=False, save="_01_CLDN1.png",
           title="CLDN1 expression across the normal liver atlas (log-normalised)")

rows = []
for ct, sub in adata.obs.groupby("cell_type", observed=True):
    idx = adata.obs_names.get_indexer(sub.index)
    m = adata.raw[idx, TARGETS].X
    m = m.toarray() if sp.issparse(m) else m
    row = {"cell_type": ct, "n_cells": len(idx), "n_donors": sub.donor[sub.donor != "unknown"].nunique()}   # known donors only
    for j, g in enumerate(TARGETS):
        row[f"{g}_pct"] = round(100 * (m[:, j] > 0).mean(), 1)
        row[f"{g}_mean"] = round(float(m[:, j].mean()), 3)
    rows.append(row)
pd.DataFrame(rows).to_csv(TAB / "01_target_expression_by_celltype.csv", index=False)

# Ambient / contaminating RNA check: ALB is hepatocyte-specific, so its detection in non-hepatocytes estimates the
# background that also inflates low CLDN1 percentages there (added after the first run; docs/DECISIONS.md D7f).
amb = []
for ct, sub in adata.obs.groupby("cell_type", observed=True):
    idx = adata.obs_names.get_indexer(sub.index)
    m = adata.raw[idx, ["ALB", "CLDN1"]].X; m = m.toarray() if sp.issparse(m) else m
    amb.append({"cell_type": ct, "n_cells": len(idx),
                "ALB_pct": round(100 * (m[:, 0] > 0).mean(), 1), "ALB_mean": round(float(m[:, 0].mean()), 3),
                "CLDN1_pct": round(100 * (m[:, 1] > 0).mean(), 1), "CLDN1_mean": round(float(m[:, 1].mean()), 3)})
amb = pd.DataFrame(amb); amb.to_csv(TAB / "01_ambient_ALB_by_celltype.csv", index=False); print(amb.to_string(index=False))

# ---- 6. Cell-type signatures for script 03 --------------------------------
sc.tl.rank_genes_groups(adata, "cell_type", method="wilcoxon", use_raw=True)
deg = sc.get.rank_genes_groups_df(adata, group=None)
# Full table (~14 MB) is kept out of git; results/ gets the top 100 per cell type plus every CLDN1 row (< 5 MB rule).
deg.to_csv(PROC / "01_celltype_markers_all.csv.gz", index=False)
top = deg.sort_values("scores", ascending=False).groupby("group", observed=True).head(100)
pd.concat([top, deg[deg.names == "CLDN1"]]).drop_duplicates().to_csv(TAB / "01_celltype_markers_top100.csv", index=False)
sigs = (deg.query("pvals_adj < 0.01 and logfoldchanges > 0.5")
           .sort_values("logfoldchanges", ascending=False)
           .groupby("group", observed=True).head(50)
           .groupby("group", observed=True)["names"].apply(list).to_dict())
json.dump(sigs, open(PROC / "celltype_signatures.json", "w"), indent=1)
adata.write_h5ad(PROC / "atlas.h5ad")
print("01 done:", {k: len(v) for k, v in sigs.items()})

# ---- 7. CLDN1 by Leiden cluster; hepatocyte sensitivity; ambiguous-cluster check ----------------------------
# Effect sizes reported downstream are % expressing and mean log-normalised expression, NOT scanpy's Wilcoxon
# logfoldchanges (approximate: computed from means of log values). See docs/DECISIONS.md.
cldn1 = adata.raw[:, "CLDN1"].X
adata.obs["CLDN1_lognorm"] = np.asarray(cldn1.toarray() if sp.issparse(cldn1) else cldn1).ravel()
known_n = lambda s: s[s != "unknown"].nunique()
pct_pos = lambda v: round(100 * (v > 0).mean(), 1)
g = adata.obs.groupby("leiden", observed=True)
per_cl = pd.DataFrame({"n_cells": g.size(), "n_donors": g.donor.agg(known_n),
                       "CLDN1_pct": g.CLDN1_lognorm.agg(pct_pos), "CLDN1_mean": g.CLDN1_lognorm.mean().round(3)})
per_cl.index = per_cl.index.astype(str)
ann = cl[["cell_type", "ambiguous", "top_donor", "top_donor_pct", "possibly_donor_driven"]].copy(); ann.index = ann.index.astype(str)
per_cl = ann.join(per_cl).rename_axis("leiden").sort_index(key=lambda s: s.astype(int))
per_cl.to_csv(TAB / "01_CLDN1_by_leiden_cluster.csv")
print(per_cl.to_string())

hep = adata.obs.cell_type == "Hepatocyte"
amb = [str(c) for c in cl.index[cl.ambiguous & (cl.cell_type == "Hepatocyte")]]
hep_b = hep & ~adata.obs.leiden.astype(str).isin(amb)
def summ(mask):
    v = adata.obs.loc[mask, "CLDN1_lognorm"]
    return {"n_cells": int(mask.sum()), "n_donors": known_n(adata.obs.loc[mask, "donor"]),
            "CLDN1_pct": pct_pos(v), "CLDN1_mean": round(float(v.mean()), 3)}
by_donor = pd.DataFrame({d: {**{f"a_{k}": v for k, v in summ(hep & (adata.obs.donor == d)).items()},
                             **{f"b_{k}": v for k, v in summ(hep_b & (adata.obs.donor == d)).items()}}
                         for d in sorted(adata.obs.donor[hep].unique()) if d != "unknown"}).T
by_donor = by_donor.drop(columns=["a_n_donors", "b_n_donors"]).rename_axis("donor")
by_donor[["a_n_cells", "b_n_cells"]] = by_donor[["a_n_cells", "b_n_cells"]].astype(int)
by_donor.to_csv(TAB / "01_hepatocyte_CLDN1_by_donor.csv")
pdn = by_donor.a_CLDN1_pct[by_donor.a_n_cells > 0]
pdn20 = by_donor.a_CLDN1_pct[by_donor.a_n_cells >= 20]
sens = pd.DataFrame([
    {"definition": "a) all hepatocyte clusters", **summ(hep)},
    {"definition": f"b) excluding ambiguous hepatocyte cluster(s) leiden {','.join(amb) or 'none'}", **summ(hep_b)},
    {"definition": "c) per donor, all hepatocyte clusters: median [min-max] across donors",
     "n_donors": len(pdn), "CLDN1_pct": round(pdn.median(), 1), "CLDN1_pct_min": pdn.min(), "CLDN1_pct_max": pdn.max()},
    {"definition": "c) per donor, donors with >=20 hepatocytes: median [min-max]",
     "n_donors": len(pdn20), "CLDN1_pct": round(pdn20.median(), 1), "CLDN1_pct_min": pdn20.min(), "CLDN1_pct_max": pdn20.max()},
])
sens["n_cells"] = sens["n_cells"].astype("Int64")
sens.to_csv(TAB / "01_hepatocyte_CLDN1_sensitivity.csv", index=False)
print(by_donor.to_string()); print(sens.to_string(index=False))

# Ambiguous cluster(s): author clusters, sort gates, top markers vs all other cells, lineage panel
for c in amb:
    m = adata.obs.leiden.astype(str) == c
    a_cl = adata.obs.loc[m, "author_cluster"].astype(str).value_counts().rename("n_cells")
    a_cl.to_frame().assign(pct=lambda d: (100 * d.n_cells / d.n_cells.sum()).round(1)).rename_axis("author_cluster") \
        .to_csv(TAB / f"01_leiden{c}_author_clusters.csv")
    gates = adata.obs.loc[m, "sort_label"].astype(str).value_counts().rename("n_cells")
    gates.to_frame().assign(pct=lambda d: (100 * d.n_cells / d.n_cells.sum()).round(1)).rename_axis("sort_label") \
        .to_csv(TAB / f"01_leiden{c}_sort_gates.csv")
    sc.tl.rank_genes_groups(adata, "leiden", groups=[c], reference="rest", method="wilcoxon", use_raw=True,
                            pts=True, key_added="rank_amb")
    top10 = sc.get.rank_genes_groups_df(adata, group=c, key="rank_amb").sort_values("scores", ascending=False).head(10)
    top10.to_csv(TAB / f"01_leiden{c}_top10_markers.csv", index=False)
    PANEL = [x for x in ["EPCAM", "TACSTD2", "KRT19", "KRT7", "SOX9", "ALB", "APOA1", "CYP3A4", "CLDN1"]
             if x in adata.raw.var_names]
    grp = np.select([m, hep_b, adata.obs.cell_type == "Cholangiocyte"],
                    [f"leiden{c}", "other_hepatocytes", "cholangiocytes"], "other")
    M = adata.raw[:, PANEL].X; M = M.toarray() if sp.issparse(M) else M
    panel = pd.DataFrame({k: (100 * (M[grp == k] > 0).mean(0)).round(1)
                          for k in [f"leiden{c}", "other_hepatocytes", "cholangiocytes"]}, index=PANEL)
    panel.rename_axis("gene_pct_expressing").to_csv(TAB / f"01_leiden{c}_lineage_panel.csv")
    print(f"\n== Leiden {c} ==\nauthor clusters:\n{a_cl.to_string()}\nsort gates:\n{gates.to_string()}")
    print("top 10 markers vs rest:\n" + top10[["names", "scores", "pct_nz_group", "pct_nz_reference", "pvals_adj"]].round(3).to_string(index=False))
    print("lineage panel (% expressing):\n" + panel.to_string())
print("01 section 7 done.")
