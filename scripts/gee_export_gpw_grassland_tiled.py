#!/usr/bin/env python3
"""TILED Global Pasture Watch grassland -> 0.05deg export (parallel longitude strips).

Faster and more robust than the single global reduceResolution job: it submits N Earth Engine tasks,
one per longitude strip, which run concurrently — each reads only ~1/N of the native 30-m pixels, so
wall time drops roughly N-fold and no single task risks a late compute/memory timeout. You merge the
strips locally with gdal (adjacent, non-overlapping -> a clean mosaic).

Output per tile: GPW_grassland_frac_0p05_<YEAR>_tile<NN>.tif — 2-band Uint16 (fraction x 10000):
    band 1 pasture   = cultivated grassland (class 1)  -> narrow base: grassland FERTILIZER
    band 2 rangeland = natural/semi-natural (class 2)  -> broad base (pasture+rangeland): manure/forage/BNF
Each tile is on the exact model grid (crsTransform [0.05,0,<west>,0,-0.05,90]); merged = 7200 x 3600.

-------------------------------------------------------------------------------------------------
SETUP (once):  pip install earthengine-api ; earthengine authenticate

CANCEL the slow single job first (Task Manager -> Cancel, or):  earthengine task cancel <ID>

RUN:      python scripts/gee_export_gpw_grassland_tiled.py --project global-wetlands-wb --tiles 8
          # raise --tiles (12, 16) for even smaller/faster strips if 8 is still slow. Must divide 7200.
MONITOR:  earthengine task list          (wait for all tiles to reach COMPLETED)

MERGE (after downloading Drive/GPW_export/GPW_grassland_frac_0p05_2022_tile*.tif to a folder):
    mkdir -p data/raw/gpw
    gdalbuildvrt data/raw/gpw/_gpw.vrt GPW_grassland_frac_0p05_2022_tile*.tif
    gdal_translate data/raw/gpw/_gpw.vrt data/raw/gpw/GPW_grassland_frac_0p05_2022.tif
    # then: python scripts/12c_build_zhang_manure.py --compare-map 2010 --pool grass
-------------------------------------------------------------------------------------------------
"""
import argparse
import sys

NLON_TOTAL = 7200          # 0.05deg columns across the globe
NLAT = 3600                # 0.05deg rows (90 .. -90)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=None, help="your Earth Engine Cloud project id")
    ap.add_argument("--year", type=int, default=2022)
    ap.add_argument("--lon-tiles", type=int, default=4, help="longitude divisions (must divide 7200)")
    ap.add_argument("--lat-tiles", type=int, default=2, help="latitude divisions (must divide 3600)")
    args = ap.parse_args()
    if NLON_TOTAL % args.lon_tiles or NLAT % args.lat_tiles:
        sys.exit(f"--lon-tiles must divide {NLON_TOTAL} and --lat-tiles must divide {NLAT}")

    try:
        import ee
    except ImportError:
        sys.exit("earthengine-api not installed. Run:  pip install earthengine-api")
    try:
        ee.Initialize(project=args.project)
    except Exception as e:                                    # not authenticated yet -> browser flow once
        print(f"  ee.Initialize failed ({e}); launching authentication...")
        ee.Authenticate(); ee.Initialize(project=args.project)

    YEAR = args.year
    gpw = (ee.ImageCollection("projects/global-pasture-watch/assets/ggc-30m/v1/grassland_c")
           .filterDate(f"{YEAR}-01-01", f"{YEAR + 1}-01-01").first())
    cls = gpw.select("dominant_class")                       # 0 Other, 1 Cultivated, 2 Natural/Semi-natural
    two = cls.eq(1).addBands(cls.eq(2)).rename(["pasture", "rangeland"])
    # bestEffort=True + a small maxPixels is the fix for "Number of pixels requested from Image.eq
    # exceeds 2^31": it makes reduceResolution aggregate from a COARSE image PYRAMID of the 0/1 masks
    # (the pyramid of a 0/1 image already IS its fraction) instead of materialising all 30 m pixels
    # globally. maxPixels=64 -> ~1 km pyramid; the 0.05deg result is still a real 30 m-informed fraction.
    frac = (two.reduceResolution(reducer=ee.Reducer.mean(), maxPixels=64, bestEffort=True)
            .reproject(crs="EPSG:4326", crsTransform=[0.05, 0, -180, 0, -0.05, 90]))
    img = frac.unmask(0).multiply(10000).round().toUint16()  # each band = fraction x 10000 (ocean -> 0)

    ncol = NLON_TOTAL // args.lon_tiles                     # columns per tile
    nrow = NLAT // args.lat_tiles                           # rows per tile
    dlon = ncol * 0.05; dlat = nrow * 0.05                  # tile size in degrees
    n = 0
    for j in range(args.lat_tiles):                        # rows: north -> south
        north = 90.0 - j * dlat
        for i in range(args.lon_tiles):                    # cols: west -> east
            west = -180.0 + i * dlon
            name = f"GPW_grassland_frac_0p05_{YEAR}_tile{n:03d}"
            task = ee.batch.Export.image.toDrive(
                image=img, description=name, folder="GPW_export", fileNamePrefix=name,
                crs="EPSG:4326", crsTransform=[0.05, 0, west, 0, -0.05, north],
                dimensions=f"{ncol}x{nrow}", maxPixels=int(1e13), fileFormat="GeoTIFF")
            task.start()
            n += 1
    print(f"  submitted {n} tiles ({args.lon_tiles} x {args.lat_tiles}, each {dlon:.0f} x {dlat:.0f} deg).")
    print(f"  Monitor:  earthengine task list   (they run in parallel; check none FAIL in the first minute)")
    print(f"  when all COMPLETED, download Drive/GPW_export/*tile*.tif to one folder and merge:")
    print(f"    gdalbuildvrt data/raw/gpw/_gpw.vrt GPW_grassland_frac_0p05_{YEAR}_tile*.tif")
    print(f"    gdal_translate data/raw/gpw/_gpw.vrt data/raw/gpw/GPW_grassland_frac_0p05_{YEAR}.tif")
    print(f"    gdal_translate data/raw/gpw/_gpw.vrt data/raw/gpw/GPW_grassland_frac_0p05_{YEAR}.tif")


if __name__ == "__main__":
    main()
