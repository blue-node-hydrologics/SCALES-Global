#!/usr/bin/env python3
"""
Repackage SCALES-Global annual grids into compact, CF-compliant, per-variable
"time-stacked" NetCDF cubes for public deposit (Zenodo).

What it does
------------
Reads the per-year grids produced by the model
    <grids>/surplus_N_<year>_0p05.nc            (24 variables, float32, zlib-1)
and rewrites them, grouped by VARIABLE with a TIME dimension, as
    <out>/full/<var>_<Y0>_<Y1>.nc               (time, lat, lon) cubes
    <out>/lite/surplus_*_<Y0>_<Y1>.nc           (the 4 surplus rates + areas)

Compression / precision
-----------------------
  * rate variables  (*_kgha) : packed to int16 with per-variable scale_factor /
                               add_offset (range set to the true data min/max, so
                               nothing is clipped). Step is range/65534 -- for a
                               surplus field spanning a few thousand kg/ha this is
                               ~0.1 kg/ha, far below the model's real uncertainty.
  * mass variables  (*_kg)   : kept float32 (LOSSLESS) so per-cell masses still
    and areas (cropland_ha)    sum exactly to the national budget. Compressed with
                               the shuffle filter + deflate level 5.
  * ocean/NaN (~70% of cells) compresses to almost nothing either way.

Also builds the "lite" product: the four surplus-rate cubes (int16) plus
cropland_ha and a geodesic cell_area_ha grid, so users can map surpluses and
area-aggregate without downloading the full component set.

Usage (from the repository root)
--------------------------------
    python repackage_grids.py surplus \
        --grids "data/outputs/grids" \
        --out   "data/outputs/zenodo"

    # every-5-year uncertainty snapshots (+ 2023):
    python repackage_grids.py uncertainty \
        --grids "data/outputs/uncertainty_mc/grids" \
        --out   "data/outputs/zenodo" \
        --years 1961 1965 1970 1975 1980 1985 1990 1995 2000 2005 2010 2015 2020 2023

    # tiny self-test on synthetic data (no real files needed):
    python repackage_grids.py selftest
"""
from __future__ import annotations
import argparse
import datetime as dt
import glob
import os
import sys
import time
from pathlib import Path

import warnings
import numpy as np

# cosmetic: numpy warns when netCDF4 casts a NaN-bearing array to a packed
# int type; the masked write handles NaN correctly, so silence just this one.
warnings.filterwarnings("ignore", message="invalid value encountered in cast")

try:
    import netCDF4 as nc
except Exception as e:  # noqa: BLE001
    sys.stderr.write("ERROR: netCDF4 is required (conda install -c conda-forge netcdf4)\n")
    raise

# ---------------------------------------------------------------------------
# metadata: units + human-readable descriptions for every known variable
# ---------------------------------------------------------------------------
MASS_UNITS = "kg N cell-1 yr-1"
RATE_UNITS = "kg N ha-1 yr-1"

LONG_NAMES = {
    "fert_kg": "Synthetic fertilizer N applied to cropland",
    "bnf_kg": "Biological N fixation (cropland legumes + free-living)",
    "dep_kg": "Atmospheric N deposition over cropland (NOy + NHx)",
    "man_crop_kg": "Manure N applied to cropland soils",
    "crop_kg": "Crop N removal in harvest",
    "fert_vol_kg": "Synthetic-fertilizer NH3 volatilization loss (cropland)",
    "man_vol_past_kg": "Pasture-manure NH3 volatilization loss",
    "hum_sewage_kg": "Human-waste N via sewage (net of treatment & recycling)",
    "hum_sew_water_kg": "Human-waste N routed to water (sewage sub-flux)",
    "hum_sew_land_kg": "Human-waste N routed to land (sewage sub-flux)",
    "dep_land_kg": "Atmospheric N deposition over all land (NOy + NHx)",
    "fert_grass_kg": "Grassland (non-cropland) synthetic fertilizer N",
    "man_past_kg": "Manure N deposited on pasture/rangeland by grazing animals",
    "forage_kg": "Grassland forage N removal",
    "bnf_grass_kg": "Biological N fixation by pasture legumes",
    "food_waste_kg": "Consumer-wasted food N (to landfill/compost)",
    "hum_crop_kg": "Human-excreta N recycled to cropland (night soil / biosolids)",
    "hum_ns_kg": "Human-excreta N, night-soil pathway",
    "hum_bio_kg": "Human-excreta N, biosolids pathway",
    "cropland_ha": "Cropland area in cell",
    "surplus_cropland_kgha": "Cropland N surplus (View 1; FAO-comparable scope + recycled excreta)",
    "surplus_cropland_faocomp_kgha": "Cropland N surplus (strict FAO-comparable; no recycled excreta)",
    "surplus_landscape_kgha": "Total anthropogenic N surplus on land, excl. human sewage (View 2)",
    "surplus_full_kgha": "Total anthropogenic N surplus on land, incl. human sewage (View 3)",
    "cropland_frac": "Cropland area as a fraction of cell area",
}
_VIEW_NAMES = {"cropland": "cropland N surplus", "landscape": "landscape N surplus",
               "full": "total anthropogenic N surplus"}
_UNC_NAMES = {"mean": "Monte Carlo mean of the {}",
              "sd": "standard deviation of the {}",
              "coef_sd": "standard deviation of the {} from coefficient uncertainty only",
              "act_sd": "standard deviation of the {} from activity-data uncertainty only",
              "ci95": "half-width of the 95% interval of the {} (1.96 x standard deviation)"}
for _v, _vn in _VIEW_NAMES.items():
    for _k, _txt in _UNC_NAMES.items():
        LONG_NAMES[f"surplus_{_v}_{_k}_kgha"] = _txt.format(_vn)
# not included in the deposit (FAO-comparable cropland surplus used only for validation)
EXCLUDE_VARS = {"surplus_cropland_faocomp_kgha"}


def unit_for(varname: str) -> str:
    if varname == "cropland_ha" or varname.endswith("_ha") and "kgha" not in varname:
        return "ha"
    if varname.endswith("_kgha"):
        return RATE_UNITS
    if varname.endswith("_kg"):
        return MASS_UNITS
    return "1"            # fractions (e.g. cropland_frac) are dimensionless


def is_rate(varname: str) -> bool:
    return varname.endswith("_kgha")


# ---------------------------------------------------------------------------
# Cropland-surplus RATE masking.
# The per-hectare cropland surplus divides a downscaled mass by the cropland area
# in the cell. Where a cell holds a negligible share of cropland the rate is not
# meaningful. Such cells are set to _FillValue (MASKED,
# not clipped) in the two cropland-rate cubes ONLY. Component MASS cubes and the
# landscape/full rate cubes are untouched, so national budgets and all other
# products are unchanged.
CROPLAND_RATE_VARS = ("surplus_cropland_kgha", "surplus_cropland_faocomp_kgha")
MASK_MIN_CROPLAND_FRAC = 0.001    # mask cells with < 0.1% of their area in cropland
MASK_ABS_RATE_CAP = 30000.0       # kg N ha-1 yr-1


def _cell_area_ha(lat, lon):
    """Geodesic cell area (ha), 2D (lat, lon); matches add_cell_area()."""
    R = 6371000.0
    dlat = abs(float(lat[1] - lat[0])); dlon = abs(float(lon[1] - lon[0]))
    row = (R**2) * np.deg2rad(dlon) * (
        np.sin(np.deg2rad(lat) + np.deg2rad(dlat) / 2)
        - np.sin(np.deg2rad(lat) - np.deg2rad(dlat) / 2))
    return np.abs(np.repeat((row / 1e4)[:, None], lon.size, axis=1))


def _mask_cropland_rate(arr, cropland_ha, cell_area):
    """Return a copy with non-physical cropland-rate cells set to NaN."""
    a = np.array(arr, dtype="float64")
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.asarray(cropland_ha, dtype="float64") / cell_area
        bad = (~np.isfinite(frac)) | (frac < MASK_MIN_CROPLAND_FRAC) | (np.abs(a) > MASK_ABS_RATE_CAP)
    a[bad] = np.nan
    return a


def _is_cropland_rate(v):
    """Any per-hectare CROPLAND surplus variable (mean or uncertainty)."""
    return v.startswith("surplus_cropland") and v.endswith("_kgha")


def _cropland_ha_for(ds, grids_dir, year):
    """cropland_ha for a year: from the open file, else from surplus_N_<year>."""
    if "cropland_ha" in ds.variables:
        return np.asarray(ds.variables["cropland_ha"][:], dtype="float64")
    sp = grids_dir / f"surplus_N_{year}_0p05.nc"
    if sp.exists():
        with nc.Dataset(sp) as d2:
            if "cropland_ha" in d2.variables:
                return np.asarray(d2.variables["cropland_ha"][:], dtype="float64")
    return None


# ---------------------------------------------------------------------------
# core: restack a set of per-year files into per-variable time cubes
# ---------------------------------------------------------------------------
def _pack_params(vmin: float, vmax: float):
    """int16 scale/offset covering [vmin, vmax] with -32768 reserved for _FillValue."""
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        # degenerate/constant field -> tiny nonzero scale, offset at the value
        return 1.0, float(vmin if np.isfinite(vmin) else 0.0)
    n = 65534.0  # usable levels across [-32767, 32767]
    scale = (vmax - vmin) / n
    offset = (vmax + vmin) / 2.0
    return float(scale), float(offset)


def _time_units_and_values(years):
    """CF time: days since 1900-01-01, mid-year (Jul 1) per annual step."""
    epoch = dt.date(1900, 1, 1)
    vals = [float((dt.date(int(y), 7, 1) - epoch).days) for y in years]
    return "days since 1900-01-01 00:00:00", vals


def discover_years(grids_dir: Path, prefix: str):
    pat = str(grids_dir / f"{prefix}_*_0p05.nc")
    out = []
    for f in sorted(glob.glob(pat)):
        stem = Path(f).stem  # e.g. surplus_N_1961_0p05
        parts = stem.split("_")
        # find the 4-digit year token
        for p in parts:
            if len(p) == 4 and p.isdigit():
                out.append(int(p))
                break
    return sorted(set(out))


def restack(grids_dir: Path, out_dir: Path, prefix: str, years, which_vars=None,
            product="full", global_attrs=None, quiet=False):
    """
    grids_dir/  <prefix>_<year>_0p05.nc   ->   out_dir/<product>/<var>_<y0>_<y1>.nc
    Returns dict: var -> (bytes_in_sum, bytes_out) size summary.
    """
    years = sorted(years)
    if not years:
        raise SystemExit(f"no years found for prefix '{prefix}' in {grids_dir}")
    files = {y: grids_dir / f"{prefix}_{y}_0p05.nc" for y in years}
    for y, f in files.items():
        if not f.exists():
            raise SystemExit(f"missing input grid: {f}")

    outp = out_dir / product
    outp.mkdir(parents=True, exist_ok=True)

    # --- read coords + variable list + per-variable data range (pass 1) -----
    with nc.Dataset(files[years[0]]) as ds0:
        lat = ds0.variables["lat"][:].astype("float64")
        lon = ds0.variables["lon"][:].astype("float64")
        all_vars = [v for v in ds0.variables if v not in ("lat", "lon")]
    varlist = [v for v in all_vars if (which_vars is None or v in which_vars) and v not in EXCLUDE_VARS]
    ny, nx = lat.size, lon.size
    cell_area = _cell_area_ha(lat, lon)

    # min/max per rate variable across all years (only rates need packing range)
    rng = {v: [np.inf, -np.inf] for v in varlist if is_rate(v)}
    if rng:
        for y in years:
            with nc.Dataset(files[y]) as ds:
                cha_year = None
                for v in rng:
                    if v in ds.variables:
                        a = np.asarray(ds.variables[v][:], dtype="float64")
                        if _is_cropland_rate(v):
                            if cha_year is None:
                                cha_year = _cropland_ha_for(ds, grids_dir, y)
                            if cha_year is not None:
                                a = _mask_cropland_rate(a, cha_year, cell_area)
                        m = np.isfinite(a)
                        if m.any():
                            rng[v][0] = min(rng[v][0], float(a[m].min()))
                            rng[v][1] = max(rng[v][1], float(a[m].max()))

    tunits, tvals = _time_units_and_values(years)
    y0, y1 = years[0], years[-1]
    sizes = {}

    # --- write one cube per variable (pass 2, streams year by year) ---------
    for v in varlist:
        dest = outp / f"{v}_{y0}_{y1}.nc"
        rate = is_rate(v)
        with nc.Dataset(dest, "w", format="NETCDF4") as out:
            out.createDimension("time", len(years))
            out.createDimension("lat", ny)
            out.createDimension("lon", nx)

            tv = out.createVariable("time", "f8", ("time",))
            tv.units = tunits
            tv.calendar = "standard"
            tv.standard_name = "time"
            tv.long_name = "time (mid-year)"
            tv[:] = tvals
            yv = out.createVariable("year", "i4", ("time",))
            yv.long_name = "calendar year"
            yv[:] = np.asarray(years, dtype="int32")

            latv = out.createVariable("lat", "f8", ("lat",))
            latv.units = "degrees_north"; latv.standard_name = "latitude"
            latv.long_name = "latitude"; latv.axis = "Y"
            latv[:] = lat
            lonv = out.createVariable("lon", "f8", ("lon",))
            lonv.units = "degrees_east"; lonv.standard_name = "longitude"
            lonv.long_name = "longitude"; lonv.axis = "X"
            lonv[:] = lon

            crs = out.createVariable("crs", "i4")
            crs.grid_mapping_name = "latitude_longitude"
            crs.longitude_of_prime_meridian = 0.0
            crs.semi_major_axis = 6378137.0
            crs.inverse_flattening = 298.257223563
            crs.spatial_ref = ('GEOGCS["WGS 84",DATUM["WGS_1984",'
                               'SPHEROID["WGS 84",6378137,298.257223563]],'
                               'PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]]')
            crs.epsg_code = "EPSG:4326"

            chunk = (1, min(512, ny), min(512, nx))
            if rate:
                scale, offset = _pack_params(*rng[v])
                dv = out.createVariable(
                    v, "i2", ("time", "lat", "lon"),
                    zlib=True, complevel=5, shuffle=True,
                    chunksizes=chunk, fill_value=np.int16(-32768))
                dv.scale_factor = np.float64(scale)
                dv.add_offset = np.float64(offset)
            else:
                dv = out.createVariable(
                    v, "f4", ("time", "lat", "lon"),
                    zlib=True, complevel=5, shuffle=True,
                    chunksizes=chunk, fill_value=np.float32(np.nan))
            dv.units = unit_for(v)
            dv.long_name = LONG_NAMES.get(v, v)
            dv.grid_mapping = "crs"
            dv.coordinates = "year"

            # stream each year's 2D slice
            for ti, y in enumerate(years):
                with nc.Dataset(files[y]) as ds:
                    if v not in ds.variables:
                        arr = np.full((ny, nx), np.nan, dtype="float64")
                    else:
                        arr = np.asarray(ds.variables[v][:], dtype="float64")
                        if _is_cropland_rate(v):
                            cha_y = _cropland_ha_for(ds, grids_dir, y)
                            if cha_y is not None:
                                arr = _mask_cropland_rate(arr, cha_y, cell_area)
                # netCDF4 auto-packs on write for int16 vars carrying
                # scale_factor/add_offset. Assign a masked array so NaN cells
                # map cleanly to _FillValue (int16) / NaN (float32).
                dv[ti, :, :] = np.ma.masked_invalid(arr)

            if global_attrs:
                for k, val in global_attrs.items():
                    setattr(out, k, val)
            out.Conventions = "CF-1.10"

        bytes_in = sum(files[y].stat().st_size for y in years)  # whole-file (all vars)
        sizes[v] = (bytes_in, dest.stat().st_size)
        if not quiet:
            print(f"  {v:32s} -> {dest.name:40s} "
                  f"{dest.stat().st_size/1e6:8.1f} MB"
                  + (f"  [int16 step {(_pack_params(*rng[v])[0]):.4g} {RATE_UNITS}]" if rate else "  [float32]"))
    return sizes


def add_cell_area(out_dir: Path, sample_grid: Path, product="lite"):
    """Write a geodesic cell_area_ha grid (time-invariant) into the product dir."""
    with nc.Dataset(sample_grid) as ds:
        lat = ds.variables["lat"][:].astype("float64")
        lon = ds.variables["lon"][:].astype("float64")
    R = 6371000.0
    dlat = abs(float(lat[1] - lat[0])); dlon = abs(float(lon[1] - lon[0]))
    row = (R**2) * np.deg2rad(dlon) * (
        np.sin(np.deg2rad(lat) + np.deg2rad(dlat) / 2)
        - np.sin(np.deg2rad(lat) - np.deg2rad(dlat) / 2))
    area_ha = np.abs(np.repeat((row / 1e4)[:, None], lon.size, axis=1)).astype("float32")
    dest = out_dir / product / "cell_area_ha.nc"
    with nc.Dataset(dest, "w", format="NETCDF4") as out:
        out.createDimension("lat", lat.size); out.createDimension("lon", lon.size)
        latv = out.createVariable("lat", "f8", ("lat",)); latv.units = "degrees_north"
        latv.standard_name = "latitude"; latv[:] = lat
        lonv = out.createVariable("lon", "f8", ("lon",)); lonv.units = "degrees_east"
        lonv.standard_name = "longitude"; lonv[:] = lon
        av = out.createVariable("cell_area_ha", "f4", ("lat", "lon"),
                                zlib=True, complevel=5, shuffle=True)
        av.units = "ha"; av.long_name = "Geodesic cell land+water area"
        av[:] = area_ha
        out.Conventions = "CF-1.10"
    print(f"  wrote {dest.name} ({dest.stat().st_size/1e6:.2f} MB)")


# ---------------------------------------------------------------------------
# self-test on synthetic data (verifies packing round-trip + sizes)
# ---------------------------------------------------------------------------
def selftest():
    import tempfile
    tmp = Path(tempfile.mkdtemp())
    g = tmp / "grids"; g.mkdir()
    # small but realistic: 360x720, ~70% NaN, 3 years, a few vars
    ny, nx = 360, 720
    lat = np.linspace(-89.75, 89.75, ny)
    lon = np.linspace(-179.75, 179.75, nx)
    rng = np.random.default_rng(42)
    land = rng.random((ny, nx)) > 0.7  # ~30% land
    truth = {}
    vlist = ["fert_kg", "crop_kg", "cropland_ha",
             "surplus_cropland_kgha", "surplus_full_kgha"]
    for y in (2000, 2001, 2002):
        with nc.Dataset(g / f"surplus_N_{y}_0p05.nc", "w") as ds:
            ds.createDimension("lat", ny); ds.createDimension("lon", nx)
            ds.createVariable("lat", "f8", ("lat",))[:] = lat
            ds.createVariable("lon", "f8", ("lon",))[:] = lon
            for v in vlist:
                a = np.full((ny, nx), np.nan)
                if v.endswith("_kgha"):
                    a[land] = rng.normal(60, 200, land.sum())  # can be negative
                else:
                    a[land] = np.abs(rng.normal(1e5, 3e5, land.sum()))
                ds.createVariable(v, "f4", ("lat", "lon"),
                                  zlib=True, complevel=1)[:] = a.astype("float32")
                truth[(v, y)] = a.copy()

    out = tmp / "zenodo"
    print("[selftest] restacking full ...")
    sizes = restack(g, out, "surplus_N", [2000, 2001, 2002], product="full",
                    global_attrs={"title": "selftest"})
    add_cell_area(out, g / "surplus_N_2000_0p05.nc", product="full")

    # verify round-trip against the SOURCE float32 values (what was actually stored)
    print("[selftest] verifying round-trip ...")
    ok = True
    for v in vlist:
        f = out / "full" / f"{v}_2000_2002.nc"
        with nc.Dataset(f) as ds:
            ds.set_auto_maskandscale(True)
            got = np.ma.filled(ds.variables[v][0].astype("float64"), np.nan)  # yr 2000
            sf = float(getattr(ds.variables[v], "scale_factor", 0.0) or 0.0)
        with nc.Dataset(g / "surplus_N_2000_0p05.nc") as dss:
            ref = np.asarray(dss.variables[v][:], dtype="float64")
            # cropland-rate cubes are masked on write; expect the same mask in `ref`
            if _is_cropland_rate(v) and "cropland_ha" in dss.variables:
                ref = _mask_cropland_rate(
                    ref, np.asarray(dss.variables["cropland_ha"][:], dtype="float64"),
                    _cell_area_ha(lat, lon))
        m = np.isfinite(ref)
        # NaN pattern preserved (source-NaN plus any masked cropland-rate cells)?
        nan_ok = np.array_equal(np.isfinite(got), m)
        if v.endswith("_kgha"):
            step = sf if sf > 0 else (np.nanmax(ref) - np.nanmin(ref)) / 65534.0
            err = np.nanmax(np.abs(got[m] - ref[m])) if m.any() else 0.0
            good = err <= step * 1.01
            print(f"    {v:30s} int16  max|err|={err:.4g}  step={step:.4g}  "
                  f"{'OK' if good and nan_ok else 'FAIL'}")
            ok &= good and nan_ok
        else:
            err = np.nanmax(np.abs(got[m] - ref[m]))
            good = err == 0.0
            print(f"    {v:30s} float32 max|err|={err:.4g}  {'OK' if good and nan_ok else 'FAIL'}")
            ok &= good and nan_ok

    tot_in = sum(f.stat().st_size for f in (g).glob("*.nc"))
    tot_out = sum(f.stat().st_size for f in (out / "full").glob("*.nc"))
    print(f"[selftest] source {tot_in/1e6:.2f} MB (per-year, zlib1) -> "
          f"cubes {tot_out/1e6:.2f} MB")
    print("[selftest]", "ALL CHECKS PASSED" if ok else "*** CHECKS FAILED ***")
    return ok


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("surplus", help="repackage the annual surplus grids (full + lite)")
    sp.add_argument("--grids", required=True)
    sp.add_argument("--out", required=True)
    sp.add_argument("--start", type=int, default=None)
    sp.add_argument("--end", type=int, default=None)
    sp.add_argument("--vars", nargs="+", default=None,
                    help="rebuild only these variables (subset re-pack, e.g. after a mask change)")

    up = sub.add_parser("uncertainty", help="repackage uncertainty grids for snapshot years")
    up.add_argument("--grids", required=True)
    up.add_argument("--out", required=True)
    up.add_argument("--years", type=int, nargs="+", required=True)

    sub.add_parser("selftest", help="synthetic round-trip test (no real files)")

    args = ap.parse_args()

    if args.cmd == "selftest":
        sys.exit(0 if selftest() else 1)

    grids = Path(args.grids); out = Path(args.out)
    gattrs = {
        "title": "SCALES-Global: annual 0.05-degree global nitrogen-surplus dataset (1961-2023)",
        "institution": "The Pennsylvania State University; University of Waterloo",
        "source": "SCALES framework (FAOSTAT national budgets downscaled to 0.05deg)",
        "references": "Van Meter & Basu, SCALES-Global data descriptor (Scientific Data, in prep)",
        "license": "CC-BY-4.0",
    }

    if args.cmd == "surplus":
        years = discover_years(grids, "surplus_N")
        if args.start: years = [y for y in years if y >= args.start]
        if args.end:   years = [y for y in years if y <= args.end]
        print(f"[surplus] {len(years)} years: {years[0]}-{years[-1]}")
        t0 = time.time()
        # FULL: every variable
        sel = set(args.vars) if args.vars else None
        print("[surplus] FULL product" + (f" (subset: {sorted(sel)})" if sel else " (all component + surplus cubes):"))
        restack(grids, out, "surplus_N", years, which_vars=sel, product="full",
                global_attrs=gattrs)
        # LITE: the 3 surplus rates + cropland_ha + cell area
        lite_vars = ["surplus_cropland_kgha", "surplus_landscape_kgha", "surplus_full_kgha", "cropland_ha"]
        if sel:
            lite_vars = [v for v in lite_vars if v in sel]
        if lite_vars:
            print("[surplus] LITE product (surplus rates + areas):")
            restack(grids, out, "surplus_N", years, which_vars=lite_vars, product="lite",
                    global_attrs=gattrs)
        if not args.vars:
            add_cell_area(out, grids / f"surplus_N_{years[0]}_0p05.nc", product="lite")
        print(f"[surplus] done in {time.time()-t0:.0f}s -> {out}")

    elif args.cmd == "uncertainty":
        years = sorted(args.years)
        print(f"[uncertainty] snapshot years: {years}")
        restack(grids, out, "uncertainty_N", years, which_vars=None,
                product="uncertainty", global_attrs=gattrs)
        print(f"[uncertainty] done -> {out}")


if __name__ == "__main__":
    main()
