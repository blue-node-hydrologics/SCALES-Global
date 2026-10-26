#!/usr/bin/env python3
"""Acquire Gridded Livestock of the World (GLW) per-species rasters for the manure (MAN)
downscaling weight, into data/raw/glw/.

GLW4 (2020, FAOSTAT-aligned) is distributed via FAO's GeoNetwork catalog, which is
JavaScript-gated and can't be scripted reliably. So this script:
  1. detects GLW rasters you've already placed in data/raw/glw/ (and reports coverage), and
  2. otherwise prints the per-species download links (GLW4 2020 preferred; GLW3 2015 on
     Harvard Dataverse as a scriptable fallback for the spatial pattern).

For downscaling we only need the *spatial pattern* of each species (the national magnitude
comes from the validated budget), so GLW4-2020 is ideal but GLW3-2015 patterns are acceptable.

Usage:  python scripts/07_download_glw.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import os; os.chdir(ROOT)
from src import config as cfgmod  # noqa: E402

SPECIES = ["cattle", "buffaloes", "sheep", "goats", "pigs", "chickens", "ducks", "horses"]

# GLW4 2020 per-species FAO catalog pages (each has a Download link to the GeoTIFF):
GLW4_2020_PAGES = {s: f"https://www.fao.org/livestock-systems/global-distributions/{s}/en/"
                   for s in SPECIES}
# GLW3 2015 scriptable fallback (Harvard Dataverse persistent DOIs); cattle confirmed,
# others discoverable from the same Dataverse collection.
GLW3_2015_CATTLE = "https://doi.org/10.7910/DVN/LHBICE"


def main():
    out = cfgmod.get_path(cfgmod.load_config(), "raw") / "glw"
    out.mkdir(parents=True, exist_ok=True)
    have = {s: [p.name for p in out.glob(f"*{s[:4]}*")] for s in SPECIES}
    present = [s for s in SPECIES if have[s]]
    missing = [s for s in SPECIES if not have[s]]

    print(f"GLW raster folder: {out}")
    if present:
        print("  already present:", ", ".join(present))
    if not missing:
        print("All 8 species present. Ready for the downscale engine.")
        return

    print("\nMissing species:", ", ".join(missing))
    print("\nDownload each species' GLW4-2020 GeoTIFF (total count, 'Da' dasymetric) and save")
    print(f"into {out}  (name so the species is in the filename, e.g. GLW4_2020_cattle.tif):")
    for s in missing:
        print(f"  - {s:10s} {GLW4_2020_PAGES[s]}   (click the 2020 'Download' link)")
    print("\nFallback (scriptable, 2015 pattern): GLW3 2015 on Harvard Dataverse, e.g. cattle")
    print(f"  {GLW3_2015_CATTLE} ; the other 7 species are in the same GLW3-2015 Dataverse collection.")
    print("\nAfter placing files, re-run this script to confirm coverage.")


if __name__ == "__main__":
    main()
