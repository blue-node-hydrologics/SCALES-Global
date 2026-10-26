#!/usr/bin/env python3
"""Download GHS-POP population epochs for the historical human-waste pattern.

JRC GHSL R2023A, 30 arc-sec (matches the 2020 file already on disk). Default epochs
1975/1990/2000/2015 (2020 already present). Clean download, no auth. Saved to
data/raw/population_grid/ alongside GHS_POP_E2020_...tif.

Usage:  python scripts/22_download_ghs_epochs.py
        python scripts/22_download_ghs_epochs.py --years 1975 1990 2000 2015
"""
import sys, io, zipfile, argparse
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import os; os.chdir(ROOT)
from src import config as cfgmod  # noqa: E402

BASE = ("https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_POP_GLOBE_R2023A/"
        "GHS_POP_E{y}_GLOBE_R2023A_4326_30ss/V1-0/GHS_POP_E{y}_GLOBE_R2023A_4326_30ss_V1_0.zip")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="+", default=[1975, 1990, 2000, 2015])
    args = ap.parse_args()
    out = cfgmod.get_path(cfgmod.load_config(), "raw") / "population_grid"
    out.mkdir(parents=True, exist_ok=True)
    for y in args.years:
        if list(out.glob(f"GHS_POP_E{y}_*.tif")):
            print(f"  have {y}"); continue
        url = BASE.format(y=y)
        print(f"  downloading GHS-POP {y} ...")
        try:
            r = requests.get(url, timeout=1800); r.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                for n in z.namelist():
                    if n.lower().endswith((".tif", ".tiff")):
                        (out / Path(n).name).write_bytes(z.read(n))
                        print(f"    saved {Path(n).name}")
        except Exception as e:  # noqa: BLE001
            print(f"    [error] {y}: {e}")
    print("Done.")


if __name__ == "__main__":
    main()
