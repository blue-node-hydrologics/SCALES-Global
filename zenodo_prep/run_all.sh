#!/usr/bin/env bash
# Build the SCALES-Global data deposit (grids + tables) from the model outputs.
# Run from the repository root:  bash zenodo_prep/run_all.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GRIDS="$ROOT/data/outputs/grids"
UNC="$ROOT/data/outputs/uncertainty_mc"
TAB="$ROOT/data/outputs"
OUT="$ROOT/data/outputs/zenodo"

cd "$ROOT/zenodo_prep"

echo "== self-tests =="
python repackage_grids.py selftest
python consolidate_tables.py selftest

echo "== 1/3 surplus grids -> full + lite time-stacked cubes =="
python repackage_grids.py surplus --grids "$GRIDS" --out "$OUT"

echo "== 2/3 uncertainty grids -> every-5-year snapshots (+2023) =="
python repackage_grids.py uncertainty --grids "$UNC/grids" --out "$OUT" \
  --years 1961 1965 1970 1975 1980 1985 1990 1995 2000 2005 2010 2015 2020 2023

echo "== 3/3 tables -> consolidated national + global CSVs =="
python consolidate_tables.py run --outputs "$TAB" --uncertainty "$UNC" --out "$OUT/tables"

# Zenodo allows at most 100 files per record, so the 63 per-year national tables go in one zip
python - "$OUT/tables" <<'PY'
import shutil, sys
from pathlib import Path
t = Path(sys.argv[1]); d = t / "per_year_national"
if d.is_dir():
    shutil.make_archive(str(t / "per_year_national_N_1961_2023"), "zip", root_dir=t, base_dir="per_year_national")
    shutil.rmtree(d)
    print("  per_year_national/ -> per_year_national_N_1961_2023.zip")
PY

echo "== documentation =="
cp "$ROOT/zenodo_prep/DEPOSIT_README.md" "$OUT/README.md"
python build_data_dictionary.py --package "$OUT"

echo "== done. package at: $OUT =="
du -sh "$OUT"/* 2>/dev/null || true
