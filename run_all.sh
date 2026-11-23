#!/usr/bin/env bash
# Reproduce the full analysis. Usage: bash run_all.sh   (Python >= 3.10; ~20-40 min incl. downloads)
set -euo pipefail
cd "$(dirname "$0")"; mkdir -p logs
python -m pip install -q -r env/requirements.txt
bash   scripts/00_download.sh         2>&1 | tee logs/00_download.log
python scripts/00b_convert_rdata.py   2>&1 | tee logs/00b_convert.log
python scripts/01_atlas.py            2>&1 | tee logs/01.log
python scripts/02_bulk.py             2>&1 | tee logs/02.log
python scripts/03_integrate.py        2>&1 | tee logs/03.log
python scripts/04_classifier.py       2>&1 | tee logs/04.log
python -m pip freeze > env/pip_freeze.txt
echo "Done. See results/ and docs/INTERPRETATION.md"
