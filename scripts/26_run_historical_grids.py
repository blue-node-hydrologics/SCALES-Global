#!/usr/bin/env python3
"""Generate the per-year gridded N surplus (1961-2023) at 0.05deg — the historical product.

For each year, run_downscale() uses year-specific FAOSTAT national magnitudes and selects the
nearest available epoch of each time-varying spatial proxy (GLW livestock, MODIS grazing land,
GHS-POP population). Cropland/crop pattern (CropGRIDS) and fertilizer intensity (NPKGRIDS) are
held at their modern pattern in this version (national magnitudes still vary by year); the
MODIS cropland-envelope and Lu&Tian fertilizer-pattern upgrades are the documented next step.

Resumable: skips years whose NetCDF already exists. Writes grids/surplus_N_<year>_0p05.nc.

Usage:  python scripts/26_run_historical_grids.py --start 1961 --end 2023
        python scripts/26_run_historical_grids.py --year 2022      # single (regression check)
"""
import sys, argparse
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import os; os.chdir(ROOT)
import functools
from src import config as cfgmod, downscale as ds, faostat_api  # noqa: E402

# speed up the per-year national-table reads: parse each FAOSTAT bulk once, cache to parquet
_PQ = ROOT / "data/raw/faostat/_parquet_cache"; _PQ.mkdir(parents=True, exist_ok=True)
_orig_read = faostat_api.read_domain


@functools.lru_cache(maxsize=None)
def _fast_read(domain, raw_dir):
    import pandas as pd
    pq = _PQ / f"{domain}.parquet"
    if pq.exists():
        return pd.read_parquet(pq)
    df = _orig_read(domain, raw_dir)
    try:
        df.to_parquet(pq)
    except Exception:  # noqa: BLE001
        pass
    return df


faostat_api.read_domain = _fast_read

MIN_CROP_FRAC = 0.01   # per-ha cropland surplus reported where cropland >= 1% of the cell (config grids.min_cropland_frac)

VARS = ["crop", "bnf", "fert", "dep", "hum_sewage", "hum_sew_water", "hum_sew_land",
        "man_crop", "man_past", "man_vol_past",
        "dep_land", "fert_grass", "fert_vol", "forage", "bnf_grass",
        "food_waste", "hum_crop", "hum_ns", "hum_bio"]   # dep_land=dep on all land; food_waste=consumer-wasted food N;
                                                 # fert_grass=(1-CF) grassland fert (NARROW managed pasture);
                                                 # fert_vol=synthetic-fert NH3 loss (cropland); man_past=pasture-
                                                 # deposited manure (BROAD pasture+rangeland); man_vol_past=its NH3
                                                 # loss; forage=grassland forage removal (BROAD); bnf_grass=pasture-
                                                 # legume BNF (NARROW managed pasture)


def write_year(cfg, year, out_dir, cache):
    import xarray as xr, time
    _t0 = time.time()
    r = ds.run_downscale_fast(cfg, year, cache, "N")
    print(f"  {year}: downscale done in {time.time()-_t0:.0f}s; writing NetCDF ...", flush=True)
    _tw = time.time()
    coords = {"lat": r["lat"], "lon": r["lon"]}
    dvars = {f"{k}_kg": (("lat", "lon"), r[k].astype("float32")) for k in VARS}
    dvars["cropland_ha"] = (("lat", "lon"), r["cropland_ha"].astype("float32"))
    # Per-hectare cropland surplus is reported only where cropland >= MIN_CROP_FRAC of the cell area
    # : below that the denominator is a trace of cropland and the per-ha value is not
    # meaningful. Mass variables (*_kg) are written for every cell and conserve national totals.
    lat = np.asarray(r["lat"], "float64"); lon = np.asarray(r["lon"], "float64")
    cell_ha = (ds._cell_area_ha(lat, abs(lat[1] - lat[0]), abs(lon[1] - lon[0])) / 1e4)[:, None] * np.ones((1, len(lon)))
    min_frac = float((cfg.get("grids", {}) or {}).get("min_cropland_frac", MIN_CROP_FRAC))
    crop_frac = np.asarray(r["cropland_ha"], "float64") / cell_ha
    ok = crop_frac >= min_frac
    dvars["cropland_frac"] = (("lat", "lon"), crop_frac.astype("float32"))
    for v in ("surplus_cropland_kgha", "surplus_cropland_faocomp_kgha"):
        dvars[v] = (("lat", "lon"), np.where(ok, r[v], np.nan).astype("float32"))
    attrs = {"cropland_frac": {"units": "1", "long_name": "cropland area / cell area (HYDE 3.3 cropland)"},
             "surplus_cropland_kgha": {"units": "kg N ha-1 cropland yr-1",
                                       "comment": f"missing where cropland_frac < {min_frac:g}"},
             "surplus_cropland_faocomp_kgha": {"units": "kg N ha-1 cropland yr-1",
                                               "comment": f"FAO-comparable (no hum_crop); missing where "
                                                          f"cropland_frac < {min_frac:g}"}}
    dvars["surplus_landscape_kgha"] = (("lat", "lon"), r["surplus_landscape_kgha"].astype("float32"))        # land surface (no human)
    dvars["surplus_full_kgha"] = (("lat", "lon"), r["surplus_full_kgha"].astype("float32"))        # incl. human waste
    ncf = out_dir / f"surplus_N_{year}_0p05.nc"
    dset = xr.Dataset(dvars, coords=coords)
    for v, a in attrs.items():
        dset[v].attrs.update(a)
    dset.attrs["min_cropland_frac"] = min_frac
    dset.to_netcdf(
        ncf, encoding={v: {"zlib": True, "complevel": 1} for v in dvars})  # complevel 1: faster write
    vc = np.isfinite(r["surplus_cropland_kgha"])
    print(f"  {year}: wrote {ncf.name} in {time.time()-_tw:.0f}s  (cropland surplus median "
          f"{np.nanmedian(r['surplus_cropland_kgha'][vc]):.1f} kg/ha)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=1961)
    ap.add_argument("--end", type=int, default=2023)
    ap.add_argument("--year", type=int, default=None)
    ap.add_argument("--force", action="store_true", help="regenerate years even if the NetCDF exists")
    args = ap.parse_args()
    cfg = cfgmod.load_config()
    out_dir = cfgmod.get_path(cfg, "outputs") / "grids"
    out_dir.mkdir(parents=True, exist_ok=True)

    # build the fixed-weight cache once (CropGRIDS/NPKGRIDS/mask), then reuse every year.
    # rebuild if missing OR too few crops (guards against a partial/test cache).
    wp = cfgmod.get_path(cfg, "processed") / "downscale_cache" / "weights.npz"
    ok = False
    if wp.exists():
        try:
            ok = len(np.load(wp, allow_pickle=False)["codes"]) >= 100
        except Exception:  # noqa: BLE001
            ok = False
    if not ok:
        print("building weight cache (one-time; reads CropGRIDS/NPKGRIDS once) ...")
        ds.build_weight_cache(cfg, "N")
    cache = ds.load_weight_cache(cfg)

    years = ([args.year] if args.year else list(range(args.start, args.end + 1)))
    for year in years:
        if (out_dir / f"surplus_N_{year}_0p05.nc").exists() and not args.force:
            print(f"  {year}: exists, skip"); continue
        try:
            write_year(cfg, year, out_dir, cache)
        except Exception as e:  # noqa: BLE001
            print(f"  {year}: ERROR {e}")
    print("Done.")


if __name__ == "__main__":
    main()
