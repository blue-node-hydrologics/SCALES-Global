#!/usr/bin/env python3
"""Scripted GLW download (GLW3, 2015) for the manure (MAN) spatial weight.

GLW4-2020 is only on FAO's JavaScript catalog (not scriptable), but each FAO species page
links the 2015 release on Harvard Dataverse (persistent DOI + open API). This script:
  1. reads each FAO species page to discover its 2015 Dataverse DOI, then
  2. downloads that dataset (zip) via the Dataverse access API and extracts the GeoTIFF
     into data/raw/glw/  (named with the species so 07/downscale detect it).

GLW supplies only the within-country spatial PATTERN of livestock here; the manure-N
magnitude comes from FAOSTAT EMN. So the 2015 pattern is adequate.

Usage:  python scripts/07b_download_glw3.py
"""
import sys, re, io, zipfile
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import os; os.chdir(ROOT)
from src import config as cfgmod  # noqa: E402

SPECIES = ["cattle", "buffaloes", "sheep", "goats", "pigs", "chickens", "ducks", "horses"]
PAGE = "https://www.fao.org/livestock-systems/global-distributions/{}/en/"
DV = "https://dataverse.harvard.edu/api/access/dataset/:persistentId/?persistentId=doi:{}"
HEAD = {"User-Agent": "SCALES-Global/1.0 (research)"}


def discover_doi(species):
    """Return the 2015 Harvard Dataverse DOI (10.7910/DVN/...) from the FAO species page.
    Pages list 2010 | 2015 | 2020; the 2015 link is the last 10.7910 DOI."""
    html = requests.get(PAGE.format(species), headers=HEAD, timeout=120).text
    dois = re.findall(r"10\.7910/DVN/[A-Z0-9]+", html)
    return dois[-1] if dois else None


def main():
    out = cfgmod.get_path(cfgmod.load_config(), "raw") / "glw"
    out.mkdir(parents=True, exist_ok=True)
    for sp in SPECIES:
        if list(out.glob(f"*{sp[:4]}*.tif")):
            print(f"  have {sp}"); continue
        try:
            doi = discover_doi(sp)
            if not doi:
                print(f"  [skip] no Dataverse DOI found on FAO page for {sp}"); continue
            print(f"  {sp}: doi:{doi} -> downloading dataset zip ...")
            r = requests.get(DV.format(doi), headers=HEAD, timeout=1800)
            r.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                tifs = [n for n in z.namelist() if n.lower().endswith((".tif", ".tiff"))]
                # prefer a total/areal-density count raster; else take the first tif
                pick = next((n for n in tifs if "_Da" in n or "Aw" in n or "total" in n.lower()), None) or (tifs[0] if tifs else None)
                if not pick:
                    print(f"  [skip] no GeoTIFF in {sp} dataset"); continue
                dest = out / f"GLW3_2015_{sp}_{Path(pick).name}"
                dest.write_bytes(z.read(pick))
                print(f"    saved {dest.name}")
        except Exception as e:  # noqa: BLE001
            print(f"  [error] {sp}: {e}")
    have = [sp for sp in SPECIES if list(out.glob(f"*{sp[:4]}*.tif"))]
    print(f"\nGLW present: {len(have)}/8 species -> {', '.join(have) or 'none'}")
    print("If some failed, grab those species' 2020 GeoTIFFs manually (see scripts/07_download_glw.py).")


if __name__ == "__main__":
    main()
