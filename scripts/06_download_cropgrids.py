#!/usr/bin/env python3
"""Download CropGRIDS v1.08 (Tang et al. 2024, figshare 10.6084/m9.figshare.22491997,
CC-BY-4.0) into data/raw/cropgrids/ and extract the per-crop NetCDF maps.

Files (figshare ndownloader IDs):
  CROPGRIDSv1.08_NC_maps.zip  (~807 MB) per-crop NetCDF: harvarea/croparea (ha) @ 0.05deg
                              + Countries_2018.nc (GAUL admin-0 country mask, same grid)
  CODES.zip                   crop codes / legend
  Table_CROPGRIDSv1.08_COU.xlsx  per-country areas for all 173 crops

Usage:  python scripts/06_download_cropgrids.py     (run on a networked machine)
"""
import sys, zipfile
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import os; os.chdir(ROOT)
from src import config as cfgmod  # noqa: E402

FILES = {
    "CROPGRIDSv1.08_NC_maps.zip": "https://ndownloader.figshare.com/files/44950942",
    "CODES.zip": "https://ndownloader.figshare.com/files/44950360",
    "Table_CROPGRIDSv1.08_COU.xlsx": "https://ndownloader.figshare.com/files/44950936",
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
    out = cfgmod.get_path(cfgmod.load_config(), "raw") / "cropgrids"
    out.mkdir(parents=True, exist_ok=True)
    for name, url in FILES.items():
        _download(url, out / name)
    # extract the NetCDF maps
    nc_zip = out / "CROPGRIDSv1.08_NC_maps.zip"
    nc_dir = out / "nc_maps"
    if nc_zip.exists() and not nc_dir.exists():
        print("  extracting NetCDF maps (this is large) ...")
        nc_dir.mkdir(exist_ok=True)
        with zipfile.ZipFile(nc_zip) as z:
            z.extractall(nc_dir)
    n = len(list(nc_dir.rglob("*.nc"))) if nc_dir.exists() else 0
    print(f"Done. {n} NetCDF crop maps under {nc_dir}")
    print("Next: scripts/07_download_glw.py, scripts/08_download_population_grid.py, then the downscale engine.")


if __name__ == "__main__":
    main()
