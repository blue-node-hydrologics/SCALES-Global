#!/usr/bin/env python3
"""Download GLW livestock density for additional epochs (default 2010 + 2015).

We already have GLW4-2020. For the nearest-epoch historical manure pattern we add GLW3 (2010)
and GLW4 (2015). Each FAO species page links the per-year Harvard Dataverse releases; this
script discovers the DOIs, picks the one whose Dataverse metadata matches the target year, and
extracts the density GeoTIFF to data/raw/gridded_livestock/GLW<ep>_<year>_<species>_*.tif.

Usage:  python scripts/23_download_glw_epochs.py --years 2010 2015
(If a species/year fails to auto-resolve, grab it manually from the FAO page.)
"""
import sys, re, io, zipfile, argparse
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import os; os.chdir(ROOT)
from src import config as cfgmod  # noqa: E402

SPECIES = {"cattle": "CTL", "buffaloes": "BFL", "sheep": "SHP",
           "goats": "GTS", "pigs": "PGS", "chickens": "CHK"}
PAGE = "https://www.fao.org/livestock-systems/global-distributions/{}/en/"
META = "https://dataverse.harvard.edu/api/datasets/:persistentId/?persistentId=doi:{}"
ACCESS = "https://dataverse.harvard.edu/api/access/dataset/:persistentId/?persistentId=doi:{}"
HEAD = {"User-Agent": "SCALES-Global/1.0 (research)"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="+", default=[2010, 2015])
    args = ap.parse_args()
    out = cfgmod.get_path(cfgmod.load_config(), "raw") / "gridded_livestock"
    out.mkdir(parents=True, exist_ok=True)

    for sp, code in SPECIES.items():
        try:
            html = requests.get(PAGE.format(sp), headers=HEAD, timeout=120).text
            dois = list(dict.fromkeys(re.findall(r"10\.7910/DVN/[A-Z0-9]+", html)))
        except Exception as e:  # noqa: BLE001
            print(f"  [error] {sp} page: {e}"); continue
        for year in args.years:
            if list(out.glob(f"*{year}*{code}*.tif")) or list(out.glob(f"*{code}*{year}*.tif")):
                print(f"  have {sp} {year}"); continue
            doi = None
            for d in dois:                                  # match the DOI whose metadata cites the year
                try:
                    j = requests.get(META.format(d), headers=HEAD, timeout=60).json()
                    cite = j.get("data", {}).get("latestVersion", {}).get("citation", "") or str(j)
                    if str(year) in cite:
                        doi = d; break
                except Exception:  # noqa: BLE001
                    continue
            if not doi:
                print(f"  [skip] {sp} {year}: no DOI matched (grab manually from FAO page)"); continue
            try:
                r = requests.get(ACCESS.format(doi), headers=HEAD, timeout=1800); r.raise_for_status()
                with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                    tifs = [n for n in z.namelist() if n.lower().endswith((".tif", ".tiff"))]
                    pick = next((n for n in tifs if "_Da" in n or "Aw" in n), tifs[0] if tifs else None)
                    if not pick:
                        print(f"  [skip] {sp} {year}: no tif"); continue
                    dest = out / f"GLW_{year}_{code}_{Path(pick).name}"
                    dest.write_bytes(z.read(pick))
                    print(f"  saved {dest.name}")
            except Exception as e:  # noqa: BLE001
                print(f"  [error] {sp} {year}: {e}")
    print("Done.")


if __name__ == "__main__":
    main()
