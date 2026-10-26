#!/usr/bin/env python3
"""Download Lu & Tian (2016) half-degree gridded N & P fertilizer use, 1900-2013.

PANGAEA doi:10.1594/PANGAEA.863323 — ASCII grids, 0.5°, g N(or P) m-2 cropland yr-1,
matched to HYDE 3.2 land use. ~8 MB zip. Used for the historical FERTILIZER spatial pattern
(1961-2013); within-cell crop split comes from NPKGRIDS.

Usage:  python scripts/21_download_lu_tian.py
"""
import sys, io, zipfile
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import os; os.chdir(ROOT)
from src import config as cfgmod  # noqa: E402

URL = "https://store.pangaea.de/Publications/Lu_Tian_2016/Lu-Tian_2016.zip"


def main():
    out = cfgmod.get_path(cfgmod.load_config(), "raw") / "lu_tian_fertilizer"
    out.mkdir(parents=True, exist_ok=True)
    if list(out.rglob("*.asc")) or list(out.rglob("*.txt")):
        print(f"  already have Lu & Tian files in {out}"); return
    print(f"downloading Lu & Tian fertilizer <- {URL}")
    r = requests.get(URL, timeout=900); r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        z.extractall(out)
    print(f"  extracted {len(list(out.rglob('*')))} items -> {out}")


if __name__ == "__main__":
    main()
