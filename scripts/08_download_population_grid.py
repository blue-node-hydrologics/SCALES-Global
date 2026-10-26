#!/usr/bin/env python3
"""Download a gridded population raster (for downscaling HUM within each country) into
data/raw/population_grid/.

The national HUM total stays authoritative (coefficients/population_country.csv); this
grid only supplies the WITHIN-country spatial weight (people-dense cells get more
human-waste N). Any recent population grid works since it's used as a normalized weight.

Primary: WorldPop unconstrained global mosaic, 1 km, 2020 (open, no login).
Fallback: JRC GHS-POP (GHS_POP_E2020_GLOBE_R2023A, 1 km) — see note at bottom.

Usage:  python scripts/08_download_population_grid.py
"""
import sys
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import os; os.chdir(ROOT)
from src import config as cfgmod  # noqa: E402

# WorldPop Global 2000-2020, unconstrained, 1km mosaic, year 2020 (~1 GB GeoTIFF).
WORLDPOP_2020_1KM = ("https://data.worldpop.org/GIS/Population/Global_2000_2020_1km/2020/"
                     "0_Mosaicked/ppp_2020_1km_Aggregated.tif")
HEAD = {"User-Agent": "SCALES-Global/1.0 (research)"}


def main():
    out = cfgmod.get_path(cfgmod.load_config(), "raw") / "population_grid"
    out.mkdir(parents=True, exist_ok=True)
    dest = out / "worldpop_ppp_2020_1km.tif"
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  have {dest.name}"); return
    print(f"  downloading WorldPop 2020 1km mosaic (~1 GB) <- {WORLDPOP_2020_1KM}")
    try:
        with requests.get(WORLDPOP_2020_1KM, headers=HEAD, stream=True, timeout=7200) as r:
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
        print(f"  saved {dest}  ({dest.stat().st_size/1e6:.0f} MB)")
    except Exception as e:  # noqa: BLE001
        raise SystemExit(
            f"\nWorldPop download failed ({e}).\nMANUAL options (save a global population\n"
            f"count GeoTIFF into {out}):\n"
            "  - WorldPop: https://hub.worldpop.org/geodata/listing?id=64 (Global 2020 1km)\n"
            "  - GHS-POP (JRC, 1km, 2020): https://human-settlement.emergency.copernicus.eu/"
            "download.php?ds=pop  (GHS_POP_E2020_GLOBE_R2023A_4326_30ss or _1km)\n"
            "  - GPWv4 (NASA SEDAC, needs Earthdata login): population count, 2020, 2.5 arc-min")
    print("Next: the downscale engine resamples this to the 0.05deg grid and uses it as the HUM weight.")


if __name__ == "__main__":
    main()
