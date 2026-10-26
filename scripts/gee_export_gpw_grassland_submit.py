#!/usr/bin/env python3
"""Submit the Global Pasture Watch grassland -> 0.05deg export to Earth Engine from the TERMINAL
(Python API):
    GPW_grassland_frac_0p05_<YEAR>.tif  — a TWO-BAND Uint16 GeoTIFF (each band = fraction x 10000) on
    the model grid (crsTransform [0.05,0,-180,0,-0.05,90], 7200 x 3600) -> Drive folder 'GPW_export'.
      band 1  pasture    = cultivated grassland (class 1)  -> narrow base: grassland FERTILIZER
      band 2  rangeland  = natural/semi-natural (class 2)  -> broad base (pasture+rangeland): manure/forage/BNF
    The model builds: managed pasture = band1; broad grazing = band1 + band2.

-------------------------------------------------------------------------------------------------
One-time setup:
    pip install earthengine-api
    earthengine authenticate                 # opens a browser; sign in with your EE-enabled Google account

You also need an Earth-Engine-enabled Google Cloud project id (looks like 'ee-yourname' or
'my-project-123456'). If you don't have one, open https://code.earthengine.google.com once and it will
create/prompt one, or make one at https://console.cloud.google.com (enable the Earth Engine API).

RUN:
    python scripts/gee_export_gpw_grassland_submit.py --project YOUR_EE_PROJECT_ID
    # optional: --year 2022  --cultivated-only

MONITOR (task runs server-side; takes a while for a global 30-m reduction):
    earthengine task list
    # or watch the Tasks tab at https://code.earthengine.google.com

WHEN DONE: move Drive/GPW_export/GPW_grassland_frac_0p05_2022.tif to  data/raw/gpw/  then
    python scripts/12c_build_zhang_manure.py --compare-map 2010 --pool grass
    python scripts/12c_build_zhang_manure.py --mass-check
-------------------------------------------------------------------------------------------------
"""
import argparse
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=None, help="your Earth Engine Cloud project id (required unless a default is set)")
    ap.add_argument("--year", type=int, help="single year (shortcut for --start-year Y --end-year Y)")
    ap.add_argument("--start-year", type=int, default=2000, help="first GPW year (2000-2022)")
    ap.add_argument("--end-year", type=int, default=2022, help="last GPW year (2000-2022)")
    args = ap.parse_args()
    years = [args.year] if args.year else list(range(max(args.start_year, 2000), min(args.end_year, 2022) + 1))

    try:
        import ee
    except ImportError:
        sys.exit("earthengine-api not installed. Run:  pip install earthengine-api")

    try:
        ee.Initialize(project=args.project)
    except Exception as e:                                    # not authenticated yet -> run the browser flow once
        print(f"  ee.Initialize failed ({e}); launching authentication...")
        ee.Authenticate()
        ee.Initialize(project=args.project)

    coll = ee.ImageCollection("projects/global-pasture-watch/assets/ggc-30m/v1/grassland_c")
    for YEAR in years:
        cls = coll.filterDate(f"{YEAR}-01-01", f"{YEAR + 1}-01-01").first().select("dominant_class")

        # TWO bands so the model can separate the land bases it needs:
        #   pasture   = cultivated grassland (class 1)   -> narrow base: grassland FERTILIZER
        #   rangeland = natural/semi-natural (class 2)    -> broad base (pasture+rangeland): manure/forage/BNF
        two = cls.eq(1).addBands(cls.eq(2)).rename(["pasture", "rangeland"])

        # bestEffort=True + small maxPixels makes reduceResolution aggregate from a COARSE image PYRAMID of the
        # 0/1 masks (whose pyramid already IS the fraction) instead of materialising all 30 m pixels globally
        # -> avoids "Number of pixels requested from Image.eq exceeds 2^31". maxPixels=64 -> ~1 km pyramid.
        frac = (two.reduceResolution(reducer=ee.Reducer.mean(), maxPixels=64, bestEffort=True)
                .reproject(crs="EPSG:4326", crsTransform=[0.05, 0, -180, 0, -0.05, 90]))
        img = frac.unmask(0).multiply(10000).round().toUint16()  # each band = fraction x 10000 (ocean -> 0)

        name = f"GPW_grassland_frac_0p05_{YEAR}"
        task = ee.batch.Export.image.toDrive(
            image=img, description=name, folder="GPW_export", fileNamePrefix=name,
            crs="EPSG:4326", crsTransform=[0.05, 0, -180, 0, -0.05, 90],
            dimensions="7200x3600", maxPixels=int(1e13), fileFormat="GeoTIFF")
        task.start()
        print(f"  submitted '{name}'  (id {task.id})")
    print(f"  {len(years)} task(s) submitted ({years[0]}-{years[-1]}). Monitor: earthengine task list")
    print(f"  outputs -> Drive/GPW_export/GPW_grassland_frac_0p05_<year>.tif")
    print(f"  move them all to data/raw/gpw/ ; 12c auto-picks the file matching each model year.")


if __name__ == "__main__":
    main()
