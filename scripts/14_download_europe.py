#!/usr/bin/env python3
"""Download the European soil N surplus dataset (Batool et al. 2022, Sci. Data 9:612)
for validation. Zenodo record 6581441 (doi:10.5281/zenodo.6581441).

The record holds a 16-method ensemble of 5-arcmin gridded N surplus (kg N ha-1 of grid
area yr-1, 1850-2019) plus NUTS1/2/3 aggregated CSVs (mean +/- sd of the 16 methods).

By default this grabs the lightweight NUTS CSVs + the ensemble-relevant gridded files.
Use --all to pull every file (large). Files land in data/raw/europe_batool2022/.

Usage:  python scripts/14_download_europe.py            # NUTS csvs + (if present) a mean grid
        python scripts/14_download_europe.py --all      # everything in the record
"""
import sys, argparse
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import os; os.chdir(ROOT)
from src import config as cfgmod  # noqa: E402

ZENODO_API = "https://zenodo.org/api/records/6581441"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="download every file (large)")
    args = ap.parse_args()
    out = cfgmod.get_path(cfgmod.load_config(), "raw") / "europe_batool2022"
    out.mkdir(parents=True, exist_ok=True)

    print("querying Zenodo record 6581441 ...")
    meta = requests.get(ZENODO_API, timeout=120).json()
    files = meta.get("files", [])
    print(f"  record has {len(files)} files")

    def want(name):
        n = name.lower()
        if args.all:
            return True
        if n.endswith(".csv") and "nuts" in n:           # small aggregated tables
            return True
        return False

    picked = [f for f in files if want(f["key"])]
    if not picked:
        print("  nothing matched the default filter; listing files so you can choose --all:")
        for f in files:
            print(f"    {f['key']}  ({f['size']/1e6:.1f} MB)")
        return
    for f in picked:
        url = f["links"]["self"]; dest = out / f["key"]
        if dest.exists() and dest.stat().st_size == f["size"]:
            print(f"  have {f['key']}"); continue
        print(f"  downloading {f['key']} ({f['size']/1e6:.1f} MB) ...")
        with requests.get(url, stream=True, timeout=1800) as r:
            r.raise_for_status()
            with open(dest, "wb") as fh:
                for c in r.iter_content(1 << 20):
                    fh.write(c)
    print(f"Done -> {out}")
    print("For the gridded comparison, rerun with --all (the NetCDF methods are large).")


if __name__ == "__main__":
    main()
