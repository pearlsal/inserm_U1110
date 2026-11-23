"""Convert the GSE124395 .RData count matrix into Matrix Market files that Python can read.

The object name/class inside the RData is not documented on GEO. Strategy:
  1) try pyreadr (works if the object is a data.frame / base matrix);
  2) if that fails (e.g. a sparse dgCMatrix), fall back to a 6-line Rscript (needs base R + Matrix).
Output: data/raw/GSE124395/{counts.mtx, genes.txt, cells.txt}  (genes x cells)
"""
import subprocess, sys
from pathlib import Path
import numpy as np, pandas as pd, scipy.io, scipy.sparse as sp

RAW = Path("data/raw/GSE124395")
rdata = RAW / "GSE124395_Normalhumanliverdata.RData"
out_mtx = RAW / "counts.mtx"
if out_mtx.exists():
    print("counts.mtx already exists - skipping"); sys.exit(0)

try:
    import pyreadr
    res = pyreadr.read_r(str(rdata))
    print("Objects in RData:", {k: (type(v).__name__, getattr(v, "shape", None)) for k, v in res.items()})
    name, df = max(res.items(), key=lambda kv: np.prod(kv[1].shape))
    print(f"Using object '{name}' with shape {df.shape} (expected genes x cells)")
    df = df.fillna(0)
    scipy.io.mmwrite(str(out_mtx), sp.csr_matrix(df.values.astype(np.float32)))
    pd.Series(df.index).to_csv(RAW / "genes.txt", index=False, header=False)
    pd.Series(df.columns).to_csv(RAW / "cells.txt", index=False, header=False)
except Exception as e:
    print(f"pyreadr could not read it ({e}); falling back to Rscript")
    r = f'''
    suppressMessages(library(Matrix)); e <- new.env(); load("{rdata}", envir = e)
    for (o in ls(e)) cat(o, class(e[[o]])[1], paste(dim(e[[o]]), collapse = "x"), "\\n")
    o <- ls(e)[which.max(sapply(ls(e), function(o) prod(dim(e[[o]]) %||% 0)))]
    m <- as(as.matrix(e[[o]]), "CsparseMatrix"); m[is.na(m)] <- 0
    writeMM(m, "{out_mtx}"); writeLines(rownames(m), "{RAW/'genes.txt'}"); writeLines(colnames(m), "{RAW/'cells.txt'}")
    '''
    r = "`%||%` <- function(a, b) if (is.null(a)) b else a\n" + r
    subprocess.run(["Rscript", "-e", r], check=True)
print("Wrote", out_mtx)
