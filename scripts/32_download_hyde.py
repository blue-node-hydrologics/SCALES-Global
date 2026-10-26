#!/usr/bin/env python3
"""Download HYDE 3.3 (baseline) NetCDFs for the historical land-base envelope.

HYDE 3.3 supplies annual gridded cropland, pasture, rangeland and population at 5 arcmin back
to 1961+ (and far earlier). We use these to make the SCALES-Global spatial patterns
time-varying through the deep historical period (cropland EXTENT, grazing land, population),
keeping FAO national magnitudes and the CropGRIDS crop mix.

Files (baseline scenario, NetCDF/ folder), saved to data/raw/hyde/:
  cropland.nc (427M), pasture.nc (206M), rangeland.nc (254M), population.nc (629M)

The HYDE host (Utrecht Yoda vault) sits behind an anti-bot wall (Anubis). A plain script may
get HTTP 403. If so, download the URLs printed below via a BROWSER into data/raw/hyde/ (or map
the vault as a network drive, as the site suggests). A real browser passes the wall.

Usage:  python scripts/32_download_hyde.py            # cropland+pasture+rangeland+population
        python scripts/32_download_hyde.py --files cropland.nc population.nc
"""
import sys, argparse
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import os; os.chdir(ROOT)
from src import config as cfgmod  # noqa: E402

BASE = ("https://geo.public.data.uu.nl/vault-hyde/HYDE%203.3%5B1710493486%5D/"
        "original/hyde33_c7_base_mrt2023/NetCDF/")
DEFAULT = ["cropland.nc", "pasture.nc", "rangeland.nc", "population.nc"]
UA = {"User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                     "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="*", default=DEFAULT)
    args = ap.parse_args()
    out = cfgmod.get_path(cfgmod.load_config(), "raw") / "hyde"
    out.mkdir(parents=True, exist_ok=True)
    print(f"HYDE 3.3 baseline -> {out}")
    blocked = False
    for name in args.files:
        dest = out / name
        if dest.exists() and dest.stat().st_size > 1_000_000:
            print(f"  have {name}"); continue
        url = BASE + name
        print(f"  downloading {name} ...")
        try:
            with requests.get(url, headers=UA, stream=True, timeout=3600) as r:
                if r.status_code == 403 or "anubis" in r.text[:2000].lower():
                    print(f"    [blocked by anti-bot wall] {name}"); blocked = True; continue
                r.raise_for_status()
                with open(dest, "wb") as fh:
                    for c in r.iter_content(1 << 20):
                        fh.write(c)
            print(f"    saved ({dest.stat().st_size/1e6:.0f} MB)")
        except Exception as e:  # noqa: BLE001
            print(f"    [error] {name}: {e}"); blocked = True
    if blocked:
        print("\nAnti-bot wall hit. Download these in a BROWSER into data/raw/hyde/ :")
        for name in args.files:
            print("   ", BASE + name)
    print("Done.")


if __name__ == "__main__":
    main()
