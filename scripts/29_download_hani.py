#!/usr/bin/env python3
"""Download the HaNi dataset (Tian et al. 2022, PANGAEA 10.1594/PANGAEA.942069) for validation.

HaNi = 5-arcmin ANNUAL (1860-2019) anthropogenic N inputs, 10 categories:
  1 NH4-N fert->cropland, 2 NO3-N fert->cropland, 3 NH4-N fert->pasture, 4 NO3-N fert->pasture,
  5 manure N applied->cropland, 6 manure N applied->pasture, 7 manure dep->pasture,
  8 manure dep->rangeland, 9 NHx deposition, 10 NOy deposition.
For SCALES-Global validation we mainly want the CROPLAND categories (1,2,5) + deposition (9,10)
to cross-check our gridded inputs (fertilizer + applied manure + deposition).

This SELF-DISCOVERS the file links from the PANGAEA landing page (which loads fine from a normal
network), then downloads a keyword-filtered subset to data/raw/hani/. Needs internet access.

Usage:
  python scripts/29_download_hani.py                 # list available files only
  python scripts/29_download_hani.py --get crop      # download files whose name contains 'crop'
  python scripts/29_download_hani.py --get crop deposition
  python scripts/29_download_hani.py --all           # everything (large!)
"""
import sys, io, zipfile, argparse
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import os; os.chdir(ROOT)
from src import config as cfgmod  # noqa: E402

BASE = "https://download.pangaea.de/dataset/942069/files/{}"
HEAD = {"User-Agent": "SCALES-Global/1.0 (research)"}

# HaNi binary objects (each a zip of annual 1860-2019 NetCDFs at 5-arcmin, kg N ha-1 yr-1)
FILES = {
    "nfer_crop_nh4.zip":   "NH4-N fertilizer applied to cropland",
    "nfer_crop_no3.zip":   "NO3-N fertilizer applied to cropland",
    "nfer_pas_nh4.zip":    "NH4-N fertilizer applied to pasture",
    "nfer_pas_no3.zip":    "NO3-N fertilizer applied to pasture",
    "nmanure_app_crop.zip": "manure N application on cropland",
    "nmanure_app_pas.zip":  "manure N application on pasture",
    "nmanure_dep_pas.zip":  "manure N deposition on pasture",
    "nmanure_dep_range.zip": "manure N deposition on rangeland",
    "ndep_nhx.zip":        "NHx-N deposition",
    "ndep_noy.zip":        "NOy-N deposition",
}
# default validation subset: cropland inputs (fert + applied manure) + deposition
CROPLAND_SET = ["nfer_crop_nh4.zip", "nfer_crop_no3.zip", "nmanure_app_crop.zip",
                "ndep_noy.zip", "ndep_nhx.zip"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="*", default=None,
                    help="specific zip names to fetch; default = cropland-input validation subset")
    ap.add_argument("--all", action="store_true", help="download all 10 category files (large)")
    ap.add_argument("--extract", action="store_true", help="unzip after download")
    args = ap.parse_args()
    out = cfgmod.get_path(cfgmod.load_config(), "raw") / "hani"
    out.mkdir(parents=True, exist_ok=True)

    targets = list(FILES) if args.all else (args.files or CROPLAND_SET)
    print(f"HaNi -> {out}\n  targets: {', '.join(targets)}")
    for name in targets:
        if name not in FILES:
            print(f"  [skip] unknown file {name}"); continue
        dest = out / name
        if dest.exists() and dest.stat().st_size > 1_000_000:
            print(f"  have {name}")
        else:
            url = BASE.format(name)
            print(f"  downloading {name}  ({FILES[name]}) ...")
            try:
                with requests.get(url, headers=HEAD, stream=True, timeout=3600) as r:
                    r.raise_for_status()
                    with open(dest, "wb") as fh:
                        for c in r.iter_content(1 << 20):
                            fh.write(c)
                print(f"    saved ({dest.stat().st_size/1e6:.1f} MB)")
            except Exception as e:  # noqa: BLE001
                print(f"    [error] {name}: {e}"); continue
        if args.extract and dest.exists():
            with zipfile.ZipFile(dest) as z:
                z.extractall(out / name[:-4])
            print(f"    extracted -> {name[:-4]}/")
    print("Done.")


if __name__ == "__main__":
    main()
