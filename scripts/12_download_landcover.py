#!/usr/bin/env python3
"""Download MODIS MCD12C1 (v061) global land cover, 0.05deg CMG, for the grazing-land base.

MCD12C1 carries 'Land_Cover_Type_1_Percent' = the % of each of the 17 IGBP classes within
every 0.05deg cell. We use the grassland / shrubland / savanna percentages to build the
pasture and pasture+rangeland bases for downscaling the FAOSTAT EMN "left on pasture" pool.
One small global HDF per year, native to our model grid (no resampling needed).

Uses NASA's official earthaccess library (handles Earthdata auth + on-prem/cloud redirect):
    pip install earthaccess
Earthdata login required (free): https://urs.earthdata.nasa.gov  ->  ~/.netrc:
    machine urs.earthdata.nasa.gov login YOUR_USER password YOUR_PASS   (chmod 600 ~/.netrc)

Usage:  python scripts/12_download_landcover.py            # year 2022 (latest v061)
        python scripts/12_download_landcover.py --year 2021
"""
import sys, argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import os; os.chdir(ROOT)
from src import config as cfgmod  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=None, help="single year")
    ap.add_argument("--start", type=int, default=None, help="first year of a range")
    ap.add_argument("--end", type=int, default=None, help="last year of a range")
    args = ap.parse_args()
    out = cfgmod.get_path(cfgmod.load_config(), "raw") / "landcover"
    out.mkdir(parents=True, exist_ok=True)

    if args.start and args.end:
        years = list(range(args.start, args.end + 1))
    else:
        years = [args.year or 2022]

    try:
        import earthaccess
    except ImportError:
        print("earthaccess not installed. Run:  pip install earthaccess"); sys.exit(1)

    print("logging in to Earthdata (via ~/.netrc) ...")
    earthaccess.login(strategy="netrc")

    for year in years:
        if list(out.glob(f"MCD12C1.A{year}001.*.hdf")):
            print(f"  have {year}"); continue
        results = earthaccess.search_data(
            short_name="MCD12C1", version="061",
            temporal=(f"{year}-01-01", f"{year}-12-31"))
        if not results:
            print(f"  {year}: no granule"); continue
        files = earthaccess.download(results, str(out))
        for f in files:
            p = Path(f)
            if p.exists():
                print(f"  {year}: saved {p.name} ({p.stat().st_size/1e6:.0f} MB)")
    print("Done.")


if __name__ == "__main__":
    main()
