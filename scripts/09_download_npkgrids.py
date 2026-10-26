#!/usr/bin/env python3
"""Download NPKGRIDS v1.08 (Nguyen/Tang/Maggi et al. 2024, figshare 10.6084/m9.figshare.24616050,
CC-BY-4.0) into data/raw/npkgrids/ and extract the per-crop NetCDF maps.

NPKGRIDS is the fertilizer sibling of CropGRIDS: 173 crops, 0.05deg, crop-specific
application rate of N, P2O5, K2O in kg/ha-crop (same crop names & grid as CropGRIDS).
Used as the FERT downscaling weight (crop-specific). P2O5->P x0.436, K2O->K x0.83.

Files (figshare ndownloader IDs):
  NPKGRIDSv1.08_NC.zip  (~293 MB) 173 per-crop NetCDF (N/P2O5/K2O application rate)
  CODES.zip, NPKGRIDS_v1.08_Dataset_DistributionNotes.pdf

Usage:  python scripts/09_download_npkgrids.py   (run on a networked machine)
"""
import sys, zipfile
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import os; os.chdir(ROOT)
from src import config as cfgmod  # noqa: E402

FILES = {
    "NPKGRIDSv1.08_NC.zip": "https://ndownloader.figshare.com/files/46194435",
    "CODES.zip": "https://ndownloader.figshare.com/files/48013969",
    "NPKGRIDS_DistributionNotes.pdf": "https://ndownloader.figshare.com/files/46194705",
}
HEAD = {"User-Agent": "SCALES-Global/1.0 (research)"}


def _download(url, dest: Path):
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  have {dest.name}"); return
    print(f"  downloading {dest.name} ...")
    with requests.get(url, headers=HEAD, stream=True, timeout=3600) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)


def main():
    out = cfgmod.get_path(cfgmod.load_config(), "raw") / "npkgrids"
    out.mkdir(parents=True, exist_ok=True)
    for name, url in FILES.items():
        _download(url, out / name)
    nc_zip = out / "NPKGRIDSv1.08_NC.zip"
    nc_dir = out / "nc_maps"
    if nc_zip.exists() and not nc_dir.exists():
        print("  extracting NetCDF maps ...")
        nc_dir.mkdir(exist_ok=True)
        with zipfile.ZipFile(nc_zip) as z:
            z.extractall(nc_dir)
    n = len(list(nc_dir.rglob("*.nc"))) if nc_dir.exists() else 0
    print(f"Done. {n} NPKGRIDS NetCDF maps under {nc_dir}")


if __name__ == "__main__":
    main()
