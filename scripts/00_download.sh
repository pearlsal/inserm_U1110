#!/usr/bin/env bash
# Download public data used in this project.
#   1) Aizarani et al., Nature 2019 - human liver cell atlas (GEO: GSE124395)
#   2) Moylan et al., Hepatology 2014 - NAFLD liver biopsies, mild (F0-1) vs advanced (F3-4) fibrosis (GEO: GSE49541)
#      GSE49541 is fetched in Python with GEOparse (02_bulk.py); only GSE124395 is fetched here.
set -euo pipefail

OUT="data/raw/GSE124395"
mkdir -p "$OUT"
BASE="https://ftp.ncbi.nlm.nih.gov/geo/series/GSE124nnn/GSE124395/suppl"

# List what GEO actually holds first - file names should be verified, not assumed.
echo "Files available on GEO:"
curl -s "$BASE/" | grep -o 'GSE124395_[^"<]*' | sort -u | tee "$OUT/available_files.txt"

for f in GSE124395_Normalhumanliverdata.RData.gz GSE124395_clusterpartition.txt.gz GSE124395_File_Descriptions.txt.gz; do
  if [[ ! -s "$OUT/$f" ]]; then
    echo "Downloading $f"
    curl -fL --retry 3 -o "$OUT/$f" "$BASE/$f"
  fi
done

# Per-plate sample records (GSM title = donor, e.g. 'P325_12'; supplementary file name = plate name used in cell IDs).
# 01_atlas.py needs these to map cells to donors: the donor is NOT a fixed token of the cell ID.
GSM_META="$OUT/GSE124395_gsm_brief.txt"
if [[ ! -s "$GSM_META" ]]; then
  echo "Downloading GSM sample records"
  curl -fsL --retry 3 -o "$GSM_META" "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE124395&targ=gsm&form=text&view=brief"
fi
grep -c '^\^SAMPLE' "$GSM_META" | xargs echo "GSM records:"

gunzip -kf "$OUT"/*.gz
# md5sum is GNU coreutils; macOS ships `md5 -r` (same "hash  path" output order).
if command -v md5sum >/dev/null; then MD5="md5sum"; else MD5="md5 -r"; fi
$MD5 "$OUT"/*.gz > "$OUT/checksums.md5"
echo "Done. Contents:"; ls -lh "$OUT"
