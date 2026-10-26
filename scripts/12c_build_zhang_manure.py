#!/usr/bin/env python3
"""Build gridded manure-N fields for SCALES-Global from the Tian-group products.

  CROPLAND manure  ->  Zhang et al. 2017 (ESSD 9:667; PANGAEA 871980) "ManNitProCrpRd" = manure N
                       APPLIED TO CROPLAND, 5 arcmin, 1860-2014, kg N km-2 yr-1. Net of
                       management/housing loss and non-soil uses; comparable to FAOSTAT EMN "applied".
  GRASSLAND manure ->  Zhang RESIDUAL = total production ("ManNitPro") minus cropland application, i.e.
                       manure NOT applied to cropland (rangeland/pasture-deposited PRP + managed-grass
                       application + management losses), 5 arcmin, kg N km-2 yr-1.

Both are FAOSTAT-livestock-anchored, so we use them for the within-country DISTRIBUTION only and ENFORCE
FAOSTAT EMN national totals downstream (allocate() renormalises to applied 72381 / pasture 72380).

CROPLAND POOL — PLACEMENT BASIS
--------------------------------------------------------------------------------------------------
Default (`--placement hyde`): within each Zhang 5-arcmin cell, cropland manure is placed on the cropland
extent of that year used by every other cropland term in the model, HYDE 3.3 cropland H(m,t)
(`downscale.cropgrids_cropland_extent`). Zhang built its field on HYDE 3.2 cropland, so the mass and the
area it is placed on now share (nearly) one cropland basis; the per-hectare denominator is the same H(m,t).
Manure stays LOCAL: each Zhang cell's mass is redistributed only among the 0.05deg cells it overlaps.
A Zhang cell with no HYDE 3.3 cropland of year t places nothing (no CropGRIDS or cell-area fallback, which
would put manure where the model has no cropland); that mass is reported by --mass-check and returned to
the country by allocate() in the engine. An alternative placement (CropGRIDS croparea as the primary
weight, described below) is available with `--placement cropgrids`. The basis is written to the NetCDF attribute `placement`.

Alternative placement (CropGRIDS basis):
Manure is not crop-specific in Zhang, so it is placed by TOTAL physical cropland area (CropGRIDS
`croparea`, native 0.05deg) rather than a per-crop layer. Each 5-arcmin Zhang cropland-manure MASS
M(k,t) = density(k,t)*AREA5(k) is redistributed onto the 0.05deg cells it overlaps, weighted by
CropGRIDS croparea, using AREA-CONSERVATIVE OVERLAP (a 0.05deg cell straddling two Zhang cells takes an
area-weighted share of each). This refines the manure field from 5 arcmin to 0.05deg, concentrating it
onto the cropland within each Zhang cell, exactly parallel to the Coello->CropGRIDS fertilizer scheme,
and is mass-conserving per Zhang cell (then renormalised to FAOSTAT). PHYSICAL area (croparea), not
harvested area, is used: manure is spread on the field once, independent of multi-cropping (fertilizer
uses harvested area, per cropping event). Where a Zhang cell has no CropGRIDS croparea, its mass falls
back to the cell-area share. CropGRIDS is static ~2020, so the sub-5-arcmin placement is fixed while
Zhang supplies the 5-arcmin annual variation — parallel to Coello (annual) + CropGRIDS (static).
  The cropland pool is emitted ALREADY PLACED at 0.05deg (var `man_weight`); the engine uses it
directly (NO extra x cropland_ha). The grassland pool is unchanged: a 5-arcmin density (var
`man_density`) the engine multiplies by the HYDE grazing base (CropGRIDS is cropland-only and cannot
refine grassland).

GAP YEARS (Zhang ends 2014) are handled by the LOADER clamp to 2014; FAOSTAT sets the gap-year magnitude.

Output: <paths.raw>/zhang_manure/derived/zhang_man_crop_<year>_0p05.nc  (var man_weight; placed, kg N/cell)
        <paths.raw>/zhang_manure/derived/zhang_man_grass_<year>_0p05.nc (var man_density; kg N km-2 yr-1)

Usage:
    python scripts/12c_build_zhang_manure.py                 # all years 1961-2014, both pools
    python scripts/12c_build_zhang_manure.py --pool crop --mass-check
    python scripts/12c_build_zhang_manure.py --compare-map 2010   # old (5') vs new (CropGRIDS) map, no write
    python scripts/12c_build_zhang_manure.py --selftest
"""
import argparse
import glob
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import os
os.chdir(ROOT)
from src import config as cfgmod          # noqa: E402
from src import downscale as ds           # noqa: E402

ZH_DIR = ROOT / "data/raw/zhang_manure"
DERIVED = ZH_DIR / "derived"
CG_DIR = ROOT / "data/raw" / ds.CG_SUBDIR          # CropGRIDS NC maps (native 0.05deg) — cropland pool
GPW_DIR = ROOT / "data/raw/gpw"                    # Global Pasture Watch export — grassland pool


def set_raw_root(raw):
    """Point the input/output folders at the configured raw tree (`config.yaml` -> `paths.raw`).

    The defaults above assume `<repository>/data/raw`; set `paths.raw` in config/config.yaml if the
    input tree lives elsewhere."""
    global ZH_DIR, DERIVED, CG_DIR, GPW_DIR
    raw = Path(raw)
    ZH_DIR = raw / "zhang_manure"
    DERIVED = ZH_DIR / "derived"
    CG_DIR = raw / ds.CG_SUBDIR
    GPW_DIR = raw / "gpw"
GPW_PATTERN = "GPW_grassland_frac_0p05_*.tif"      # per-year exports GPW_grassland_frac_0p05_<YEAR>.tif
GPW_Y0, GPW_Y1 = 2000, 2022                        # GPW coverage; model years are clamped to this range.
# TIME-VARYING: each model year uses the GPW map of that year, clamped to [2000, 2022] (pre-2000 -> 2000,
# post-2022 -> 2022) — parallel to Coello/CropGRIDS feeding fertilizer. If only ONE GPW tif is present,
# every year falls back to it (a single static year, e.g. 2010). Drop the year tif(s) in GPW_DIR.
YEAR0, YEAR1 = 1961, 2023
CROP_LAST = 2014
GRASS_LAST = 2014

CROP_SUBDIR = "zhang_2017/ManNitProCrpRd"
PROD_SUBDIR = "zhang_2017/ManNitPro"
PATTERN = "*{year}*"

# Native Zhang 5-arcmin geometry (PANGAEA 871980 headerless CRLF matrices): 2124 rows x 4320 cols,
# lat [-88.5, 88.5], lon [-180, 180]. NOTE lat origin -88.5 (NOT -90) and 2124 rows (NOT 2160).
ZH_NCOLS, ZH_NROWS = 4320, 2124
ZH_CELLSIZE = 1.0 / 12.0
ZH_XLL, ZH_YLL = -180.0, -88.5
ZH_NODATA = -9999.0


# ---------- Zhang ASCII I/O (unchanged) ----------
def _read_ascii_grid(path):
    """Read a Zhang manure ASCII grid -> (values north-up 2d, header dict). NODATA/neg -> 0."""
    import pandas as pd
    with open(path) as fh:
        first = fh.readline().split()
    has_header = bool(first) and first[0].lower() in (
        "ncols", "nrows", "xllcorner", "yllcorner", "cellsize", "nodata_value")
    if has_header:
        hdr = {}
        with open(path) as fh:
            for _ in range(6):
                k, v = fh.readline().split()
                hdr[k.lower()] = float(v)
        arr = pd.read_csv(path, header=None, skiprows=6, sep=r"\s+", dtype="float64").to_numpy()
    else:
        hdr = {"ncols": ZH_NCOLS, "nrows": ZH_NROWS, "cellsize": ZH_CELLSIZE,
               "xllcorner": ZH_XLL, "yllcorner": ZH_YLL, "nodata_value": ZH_NODATA}
        arr = pd.read_csv(path, header=None, sep=r"\s+", dtype="float64").to_numpy()
    nod = hdr.get("nodata_value", ZH_NODATA)
    arr = np.where(np.isfinite(arr) & (arr != nod) & (arr > 0), arr, 0.0)
    return arr, hdr


def _find_year_file(subdir, pattern, year):
    d = ZH_DIR / subdir
    hits = [p for p in glob.glob(str(d / pattern.format(year=year)))
            if not Path(p).name.startswith("._")]
    hits = [p for p in hits if p.lower().endswith((".asc", ".txt", ".dat")) or "." not in Path(p).name]
    return sorted(hits)[0] if hits else None


def _rate_to_model_grid(rate_northup, hdr, lat, lon):
    """NEAREST-resample a rate density onto the model grid; return lat-ascending density (grassland)."""
    from rasterio.warp import reproject, Resampling
    from rasterio.transform import from_origin
    ncols = int(hdr["ncols"]); nrows = int(hdr["nrows"])
    cs = float(hdr["cellsize"]); xll = float(hdr["xllcorner"]); yll = float(hdr["yllcorner"])
    src_tr = from_origin(xll, yll + nrows * cs, cs, cs)
    dlon = float(abs(lon[1] - lon[0])); dlat = float(abs(lat[1] - lat[0]))
    west = float(min(lon)) - dlon / 2.0; north = float(max(lat)) + dlat / 2.0
    dst = np.zeros((len(lat), len(lon)), dtype="float64")
    reproject(source=rate_northup.astype("float64"), destination=dst,
              src_transform=src_tr, src_crs="EPSG:4326",
              dst_transform=from_origin(west, north, dlon, dlat), dst_crs="EPSG:4326",
              resampling=Resampling.nearest)
    return dst[::-1, :]


# ---------- area-conservative overlap: 0.05deg model cell -> Zhang 5-arcmin cells ----------
def _axis_overlap(centers, mres, cres, cn, origin):
    """Up-to-2 coarse cells each model cell overlaps, with the FRACTION OF THE MODEL CELL in each."""
    lo = centers - mres / 2.0
    hi = centers + mres / 2.0
    iA = np.clip(np.floor((lo - origin) / cres).astype("int64"), 0, cn - 1)
    iH = np.clip(np.floor((hi - origin) / cres).astype("int64"), 0, cn - 1)
    bnd = origin + (iA + 1) * cres
    fA = np.clip((np.minimum(hi, bnd) - lo) / mres, 0.0, 1.0)
    same = iH == iA
    iB = np.minimum(iA + 1, cn - 1)
    fA = np.where(same, 1.0, fA)
    fB = np.where(same, 0.0, 1.0 - fA)
    return iA, fA, iB, fB


def build_overlap(lat, lon):
    """4 (pidx int32, w float32) combos: overlap of each 0.05deg cell with Zhang 5-arcmin cells.
    Model rows outside Zhang's lat coverage [-88.5, 88.5] are zeroed (no manure at the poles)."""
    mres_lat = float(abs(lat[1] - lat[0])); mres_lon = float(abs(lon[1] - lon[0]))
    rA, fA, rB, fB = _axis_overlap(lat, mres_lat, ZH_CELLSIZE, ZH_NROWS, ZH_YLL)
    cA, gA, cB, gB = _axis_overlap(lon, mres_lon, ZH_CELLSIZE, ZH_NCOLS, ZH_XLL)
    top = ZH_YLL + ZH_NROWS * ZH_CELLSIZE
    inlat = ((lat >= ZH_YLL) & (lat <= top)).astype("float64")     # zero polar rows off the Zhang grid
    fA *= inlat; fB *= inlat
    combos = []
    for r, f in [(rA, fA), (rB, fB)]:
        for c, g in [(cA, gA), (cB, gB)]:
            combos.append(((r[:, None] * ZH_NCOLS + c[None, :]).astype("int32"),
                           (f[:, None] * g[None, :]).astype("float32")))
    return combos


def _scatter(field2d, combos, npar):
    z = np.zeros(npar, "float64")
    for pidx, w in combos:
        z += np.bincount(pidx.ravel(), weights=(w * field2d).ravel().astype("float64"), minlength=npar)
    return z


def _gather(coarse_flat, combos, shape):
    out = np.zeros(shape, "float64")
    for pidx, w in combos:
        out += w * coarse_flat[pidx]
    return out


def cropgrids_croparea(shape):
    """Total physical cropland area (ha) per 0.05deg cell = Σ CropGRIDS croparea over all crops.
    Only the per-crop CROPGRIDS layers carry 'croparea'; skip the Countries_2018 mask and anything else."""
    import xarray as xr
    acc = np.zeros(shape, "float32")
    files = sorted(glob.glob(str(CG_DIR / "CROPGRIDS*.nc")))
    for f in files:
        try:
            v = xr.open_dataset(f)
            if "croparea" not in v.data_vars:
                continue
            a = v["croparea"].values.astype("float32")
        except Exception:  # noqa: BLE001
            continue
        a = np.where(np.isfinite(a) & (a > 0), a, 0.0)
        acc += a
    if acc.sum() <= 0:
        sys.exit(f"ERROR: no CropGRIDS croparea summed from {CG_DIR} (found {len(files)} CROPGRIDS files).")
    return acc


def gpw_grassland_area(shape, cellarea_ha, base="broad", year=None):
    """Global Pasture Watch land AREA (ha) per 0.05deg cell = class fraction x cell area, from the
    2-band GEE export (band1 pasture=cultivated, band2 rangeland=natural/semi; each fraction x 10000).
      base='broad'     -> pasture + rangeland (manure PRP, forage, BNF, and the grassland-manure pool)
      base='pasture'   -> cultivated only (grassland FERTILIZER, managed pasture)
      base='rangeland' -> natural/semi-natural only
    year: use GPW of that year clamped to [GPW_Y0, GPW_Y1]; if that file is absent (or year is None),
    fall back to the newest GPW tif present (supports a single static-year export). None if none present."""
    import rasterio
    f = None
    if year is not None:
        cand = GPW_DIR / f"GPW_grassland_frac_0p05_{min(max(int(year), GPW_Y0), GPW_Y1)}.tif"
        f = str(cand) if cand.exists() else None
    if f is None:
        hits = glob.glob(str(GPW_DIR / GPW_PATTERN))
        if not hits:
            return None
        f = max(hits, key=lambda p: Path(p).stat().st_mtime)          # newest GPW export present
    print(f"  [GPW] grassland base from {Path(f).name} ({base})")
    with rasterio.open(f) as s:
        bands = [s.read(i).astype("float64") / 1e4 for i in range(1, s.count + 1)]   # x10000 -> fraction
    past = bands[0]
    rang = bands[1] if len(bands) > 1 else np.zeros_like(past)         # tolerate a 1-band (broad) export
    frac = {"broad": past + rang, "pasture": past, "rangeland": rang}[base]
    frac = np.clip(np.where(np.isfinite(frac), frac, 0.0), 0.0, 1.0)[::-1, :]        # -> lat ascending
    if frac.shape != shape:
        print(f"  [GPW][warn] {Path(f).name} shape {frac.shape} != model {shape}; grass uses 5' density fallback")
        return None
    return (frac * cellarea_ha).astype("float32")                     # land area (ha) per cell


def zhang_area5_km2():
    """5-arcmin Zhang cell area (km2) per latitude row, ascending."""
    lat5 = ZH_YLL + (np.arange(ZH_NROWS) + 0.5) * ZH_CELLSIZE
    return ds._cell_area_ha(lat5, ZH_CELLSIZE, ZH_CELLSIZE) / 1e6      # m2 -> km2


def zhang_density_crop_asc(year):
    """Zhang cropland-manure density (kg N km-2), lat-ascending (2124x4320). None if absent."""
    f = _find_year_file(CROP_SUBDIR, PATTERN, year)
    if not f:
        return None
    rate, _ = _read_ascii_grid(f)
    return rate[::-1, :]                                               # north-up -> ascending


def zhang_density_grass_asc(year):
    """Zhang grassland residual density (ManNitPro - ManNitProCrpRd), kg N km-2, lat-ascending, at the
    native 5-arcmin resolution. None if the production file is absent."""
    fp = _find_year_file(PROD_SUBDIR, PATTERN, year)
    if not fp:
        return None
    prod, _ = _read_ascii_grid(fp)
    fc = _find_year_file(CROP_SUBDIR, PATTERN, year)
    if fc:
        crop, _ = _read_ascii_grid(fc)
        resid = np.clip(prod - crop, 0.0, None) if crop.shape == prod.shape else prod
    else:
        resid = prod
    return resid[::-1, :]                                              # north-up -> ascending


def build_placed(dens_asc, land_area, combos, cellarea_ha, area5_km2, shape, fallback_area=None,
                 area_fallback=True):
    """Place a Zhang 5' manure DENSITY onto the 0.05deg grid weighted by `land_area` — CropGRIDS croparea
    for the cropland pool, GPW grassland area for the grassland pool — via area-conservative overlap.
    Fallback ladder per Zhang cell: land_area -> fallback_area (optional; e.g. NON-cropland for grassland
    so residual manure stays off cropland where GPW finds no grassland) -> cell area. Mass-conserving per
    Zhang cell (each cell's mass goes entirely to whichever weight is available). `area_fallback=False`
    drops the cell-area step (mass of a Zhang cell with no land weight is left unplaced). None if dens_asc
    is None."""
    if dens_asc is None:
        return None
    npar = ZH_NROWS * ZH_NCOLS
    M = (dens_asc * area5_km2[:, None]).ravel()                       # kg per Zhang 5' cell
    Zc = _scatter(land_area, combos, npar)                           # Σ overlap*land_area per Zhang cell
    Zf = _scatter(fallback_area, combos, npar) if fallback_area is not None else np.zeros(npar)
    Za = _scatter(cellarea_ha, combos, npar)                         # Σ overlap*cellarea (last resort)
    useC = Zc > 0
    useF = (~useC) & (Zf > 0)                                        # GPW empty -> non-cropland
    useA = (~useC) & (~useF) & (Za > 0) & bool(area_fallback)        # no non-cropland either -> uniform
    with np.errstate(divide="ignore", invalid="ignore"):
        Rc = np.where(useC, M / np.where(useC, Zc, 1.0), 0.0)
        Rf = np.where(useF, M / np.where(useF, Zf, 1.0), 0.0)
        Ra = np.where(useA, M / np.where(useA, Za, 1.0), 0.0)
    out = land_area * _gather(Rc, combos, shape) + cellarea_ha * _gather(Ra, combos, shape)
    if fallback_area is not None:
        out = out + fallback_area * _gather(Rf, combos, shape)
    return out


# ---------- grassland 5' density fallback (used only if the GPW layer is absent) ----------
def build_grass_density(year, lat, lon):
    fp = _find_year_file(PROD_SUBDIR, PATTERN, year)
    if not fp:
        return None
    prod, hdr = _read_ascii_grid(fp)
    fc = _find_year_file(CROP_SUBDIR, PATTERN, year)
    if fc:
        crop, _ = _read_ascii_grid(fc)
        resid = np.clip(prod - crop, 0.0, None) if crop.shape == prod.shape else prod
        if crop.shape != prod.shape:
            print(f"    [warn] prod/crop shape mismatch {prod.shape} vs {crop.shape}; using production")
    else:
        print(f"    [warn] no cropland file for {year}; grassland = full production")
        resid = prod
    return _rate_to_model_grid(resid, hdr, lat, lon)


PLACEMENT_ATTR = {"hyde": "hyde_cropland_overlap", "cropgrids": "cropgrids_croparea_overlap"}
PLACEMENT_LABEL = {"hyde": "year-t cropland, HYDE 3.3 H(m,t)",
                   "cropgrids": "CropGRIDS croparea"}


def save_crop(year, weight, lat, lon, placement="hyde", placement_tag=None):
    import xarray as xr
    DERIVED.mkdir(parents=True, exist_ok=True)
    da = xr.DataArray(weight.astype("float32"), coords={"lat": lat, "lon": lon}, dims=("lat", "lon"),
                      name="man_weight",
                      attrs={"units": "kg N cell-1 yr-1 (relative weight; magnitude set by FAOSTAT)",
                             "long_name": f"cropland manure placed at 0.05deg (Zhang 5' mass x "
                                          f"{PLACEMENT_LABEL[placement]}, area-conservative overlap)",
                             "placement": placement_tag or PLACEMENT_ATTR[placement],   # engine: use directly
                             "placement_fallback": ("none (unplaced; returned to country by allocate)"
                                                    if placement == "hyde" else "cell area"),
                             "placement_local": "mass conserved within each Zhang 5-arcmin cell",
                             "source": f"Zhang et al. 2017 ESSD 9:667 (ManNitProCrpRd) + {PLACEMENT_LABEL[placement]}",
                             "year": year})
    out = DERIVED / f"zhang_man_crop_{year}_0p05.nc"
    da.to_netcdf(out, encoding={"man_weight": {"zlib": True, "complevel": 4, "dtype": "float32"}})
    return out


def save_grass_placed(year, weight, lat, lon):
    """Grassland pool PLACED at 0.05deg via GPW grassland area (engine uses it directly, like cropland)."""
    import xarray as xr
    DERIVED.mkdir(parents=True, exist_ok=True)
    da = xr.DataArray(weight.astype("float32"), coords={"lat": lat, "lon": lon}, dims=("lat", "lon"),
                      name="man_weight",
                      attrs={"units": "kg N cell-1 yr-1 (relative weight; magnitude set by FAOSTAT)",
                             "long_name": "grassland manure placed at 0.05deg (Zhang 5' residual mass x GPW "
                                          "grassland area, area-conservative overlap)",
                             "placement": "gpw_grassland_overlap",       # engine: use directly, no x land base
                             "source": "Zhang et al. 2017 ESSD 9:667 (ManNitPro-ManNitProCrpRd) + "
                                       "Global Pasture Watch grassland (Parente et al. 2024)",
                             "year": year})
    out = DERIVED / f"zhang_man_grass_{year}_0p05.nc"
    da.to_netcdf(out, encoding={"man_weight": {"zlib": True, "complevel": 4, "dtype": "float32"}})
    return out


def save_grass(year, field, lat, lon):
    """Grassland 5' DENSITY fallback (only if GPW absent); engine multiplies by the grazing base."""
    import xarray as xr
    DERIVED.mkdir(parents=True, exist_ok=True)
    da = xr.DataArray(field.astype("float32"), coords={"lat": lat, "lon": lon}, dims=("lat", "lon"),
                      name="man_density", attrs={"units": "kg N km-2 yr-1",
                      "long_name": "gridded manure N density (grass); engine x grazing base",
                      "source": "Zhang et al. 2017 ESSD 9:667 (ManNitPro - ManNitProCrpRd residual)",
                      "year": year})
    out = DERIVED / f"zhang_man_grass_{year}_0p05.nc"
    da.to_netcdf(out, encoding={"man_density": {"zlib": True, "complevel": 4, "dtype": "float32"}})
    return out


def engine_cropland(cfg, croparea):
    """CropGRIDS cropland exactly as the model engine uses it (sum of croparea over the crops in
    `cropgrids_item_crosswalk.csv`, from the downscale weight cache), so applied manure sits on the same
    cropland as every other cropland term. Fodder crops outside that crop set (alfalfa, clover, forage
    maize, grasses) are therefore excluded, as they are from removal and fertilizer. Falls back to
    `croparea` (all CropGRIDS croparea files) with a warning if the cache is absent."""
    try:
        c = ds.load_weight_cache(cfg)["cropland_ha"].astype("float32")
        if c.shape == croparea.shape and c.sum() > 0:
            print(f"  cropland-pool basis: engine CropGRIDS cropland (weight cache; {c.sum()/1e6:.1f} Mha)")
            return c
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] weight cache unavailable ({e})")
    print("  [warn] cropland-pool basis falls back to ALL CropGRIDS croparea (incl. fodder crops)")
    return croparea


def placement_attr(cfg, placement):
    """NetCDF `placement` attribute matching what was actually used (depends on hyde_envelope.mode)."""
    if placement == "cropgrids":
        return PLACEMENT_ATTR["cropgrids"]
    return PLACEMENT_ATTR["hyde"]                 # both envelope modes place on HYDE cropland H(m,t)


def crop_placement_layers(cfg, placement, year, croparea, shape, lat, lon):
    """Return (primary_weight, fallback_weight) for placing the cropland pool within each Zhang cell.

    - `hyde` (default): primary = the year-t cropland extent used by every other cropland term,
      HYDE 3.3 H(m,t) (`ds.cropgrids_cropland_extent`); no fallback (see module docstring).
      Exits if HYDE cannot be loaded, rather than silently changing basis.
    - `cropgrids`: primary = CropGRIDS croparea, no intermediate fallback."""
    if placement == "cropgrids":
        return croparea, None
    ext, _ = ds.cropgrids_cropland_extent(cfg, croparea.astype("float64"), shape, lat, lon, year)
    if ext is None:
        sys.exit(f"ERROR: HYDE cropland for {year} could not be loaded (<paths.raw>/hyde/cropland.nc). "
                 f"Fix the HYDE input or rerun with --placement cropgrids.")
    ext = np.where(np.isfinite(ext) & (ext > 0), ext, 0.0).astype("float32")
    return ext, None


def mass_check(weight, dens_asc, area5_km2, label):
    M = (dens_asc * area5_km2[:, None]).ravel()
    print(f"  [mass-check {label}] GLOBAL placed={weight.sum()/1e9:.3f} Tg vs Zhang {label}={M.sum()/1e9:.3f} Tg "
          f"(diff {100*(weight.sum()-M.sum())/max(M.sum(),1):+.3f}%, = mass on cells with no land layer)")


def selftest():
    """Synthetic overlap test at the real 1/12 vs 1/20 ratio: uniform cropland -> smooth output (no grid
    artifact); mass conserved per Zhang cell."""
    global ZH_NROWS, ZH_NCOLS, ZH_YLL, ZH_XLL
    ZH_NROWS, ZH_NCOLS, ZH_YLL, ZH_XLL = 12, 12, 0.0, 0.0
    lat = np.arange(0.025, 1.0, 0.05); lon = np.arange(0.025, 1.0, 0.05)
    shape = (len(lat), len(lon)); npar = ZH_NROWS * ZH_NCOLS
    rA, fA, rB, fB = _axis_overlap(lat, 0.05, ZH_CELLSIZE, ZH_NROWS, ZH_YLL)
    cA, gA, cB, gB = _axis_overlap(lon, 0.05, ZH_CELLSIZE, ZH_NCOLS, ZH_XLL)
    combos = [((r[:, None] * ZH_NCOLS + c[None, :]).astype("int32"), (f[:, None] * g[None, :]).astype("float32"))
              for r, f in [(rA, fA), (rB, fB)] for c, g in [(cA, gA), (cB, gB)]]
    croparea = np.ones(shape, "float32"); cellarea = np.ones(shape, "float64")
    area5 = np.ones(ZH_NROWS) * 100.0
    d = np.full((ZH_NROWS, ZH_NCOLS), 3.0)
    M = (d * area5[:, None]).ravel()
    Zc = _scatter(croparea, combos, npar)
    with np.errstate(divide="ignore", invalid="ignore"):
        Rc = np.where(Zc > 0, M / np.where(Zc > 0, Zc, 1.0), 0.0)
    out = croparea * _gather(Rc, combos, shape)
    cv = out.std() / out.mean()
    ok = cv < 1e-3 and np.isclose(out.sum(), M[Zc > 0].sum())
    print(f"  [selftest] uniform cropland: output CV={cv:.2e} (old 5' nearest would step ~5e-1)  "
          f"mass ok={np.isclose(out.sum(), M[Zc>0].sum())}")
    assert ok, "SELFTEST FAILED"
    print("  [selftest] PASSED")


def compare_map(pool, year, dens_asc, land_area, combos, cellarea_ha, area5_km2, lat, lon, shape,
                fallback_area=None, crop_label="CropGRIDS croparea"):
    """OLD (Zhang 5' nearest density) vs NEW (dasymetric placement by land_area) for one pool/year."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from rasterio.warp import reproject, Resampling
    from rasterio.transform import from_origin
    new = build_placed(dens_asc, land_area, combos, cellarea_ha, area5_km2, shape, fallback_area=fallback_area)
    old = np.zeros(shape)
    reproject(source=dens_asc[::-1, :].astype("float64"), destination=old,
              src_transform=from_origin(ZH_XLL, ZH_YLL + ZH_NROWS * ZH_CELLSIZE, ZH_CELLSIZE, ZH_CELLSIZE),
              src_crs="EPSG:4326",
              dst_transform=from_origin(lon.min() - 0.025, lat.max() + 0.025, 0.05, 0.05),
              dst_crs="EPSG:4326", resampling=Resampling.nearest)
    old = old[::-1, :]
    lab = f"cropland ({crop_label})" if pool == "crop" else "grassland (GPW grassland area)"
    la = (lat > 38) & (lat < 46); lo = (lon > -100) & (lon < -84)
    sub = np.ix_(la, lo); ext = [lon[lo][0], lon[lo][-1], lat[la][0], lat[la][-1]]
    fig, ax = plt.subplots(1, 2, figsize=(12, 5.2))
    for a_, img, ttl in [(ax[0], old[sub], "OLD  (Zhang 5-arcmin, nearest)"),
                         (ax[1], new[sub], f"NEW  (area-conservative, {lab})")]:
        vmax = np.nanpercentile(img[img > 0], 99) if (img > 0).any() else 1
        im = a_.imshow(img, origin="lower", extent=ext, vmin=0, vmax=vmax, cmap="YlOrBr", aspect="auto")
        a_.set_title(ttl, fontsize=11.5, fontweight="bold"); fig.colorbar(im, ax=a_, shrink=0.8)
    fig.suptitle(f"Zhang {pool} manure downscaling — {year} (US corn belt)", fontweight="bold")
    fig.tight_layout()
    out = ROOT / "data/outputs/figures" / f"zhang_manure_{pool}_dasymetric_compare_{year}.png"
    out.parent.mkdir(parents=True, exist_ok=True); fig.savefig(out, dpi=170, facecolor="white")
    print(f"  wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", choices=["crop", "grass", "both"], default="both")
    ap.add_argument("--year", type=int)
    ap.add_argument("--start", type=int, default=YEAR0)
    ap.add_argument("--end", type=int, default=YEAR1)
    ap.add_argument("--mass-check", action="store_true")
    ap.add_argument("--compare-map", type=int, metavar="YEAR")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--placement", choices=["hyde", "cropgrids"], default="hyde",
                    help="cropland-pool weight within each Zhang cell: the year-t cropland, s(m)*H(m,t) with "
                         "s = CropGRIDS/HYDE 2020 (default, same extent as the other cropland terms), "
                         "or static CropGRIDS croparea ")
    args = ap.parse_args()

    if args.selftest:
        selftest(); return

    cfg = cfgmod.load_config()
    set_raw_root(cfgmod.get_path(cfg, "raw"))
    print(f"  raw inputs: {ZH_DIR.parent}  |  cropland placement: {args.placement}")
    mask, idx2iso, lat, lon = ds.load_country_grid(cfg)
    shape = (len(lat), len(lon))

    # static inputs for the dasymetric placement (both pools share the overlap + cell/Zhang areas)
    combos = build_overlap(lat, lon)
    dlat = float(abs(lat[1] - lat[0])); dlon = float(abs(lon[1] - lon[0]))
    cellarea_ha = (ds._cell_area_ha(lat, dlat, dlon) / 1e4)[:, None] * np.ones((1, len(lon)))
    area5_km2 = zhang_area5_km2()

    want_crop = args.pool in ("crop", "both")
    want_grass = args.pool in ("grass", "both")
    # croparea is needed by BOTH pools now: cropland uses it as the placement weight, grassland uses the
    # NON-cropland residual (cell area - croparea) as its fallback where GPW finds no grassland.
    croparea = noncrop_area = None
    if want_crop or want_grass or args.compare_map is not None:
        croparea = cropgrids_croparea(shape)
        noncrop_area = np.clip(cellarea_ha - croparea, 0.0, None).astype("float32")   # cell minus cropland
        print(f"  CropGRIDS croparea loaded ({np.count_nonzero(croparea):,} cropland cells); non-cropland fallback ready")
    crop_basis = engine_cropland(cfg, croparea) if (want_crop or args.compare_map is not None) else None
    gpw_present = bool(glob.glob(str(GPW_DIR / GPW_PATTERN)))          # GPW time-varying: loaded per year below
    if want_grass and not gpw_present:
        print(f"  GPW absent ({GPW_DIR / GPW_PATTERN}) -> grass uses 5' density fallback")

    if args.compare_map is not None:
        if args.pool == "crop":
            land_w, fb_w = crop_placement_layers(cfg, args.placement, args.compare_map, crop_basis, shape, lat, lon)
            compare_map("crop", args.compare_map, zhang_density_crop_asc(args.compare_map), land_w,
                        combos, cellarea_ha, area5_km2, lat, lon, shape, fallback_area=fb_w,
                        crop_label=PLACEMENT_LABEL[args.placement])
        elif gpw_present:
            ga = gpw_grassland_area(shape, cellarea_ha, "broad", year=args.compare_map)
            compare_map("grass", args.compare_map, zhang_density_grass_asc(args.compare_map), ga,
                        combos, cellarea_ha, area5_km2, lat, lon, shape, fallback_area=noncrop_area)
        else:
            print("  [compare-map] grass needs the GPW layer; export it first (gee_export_gpw_grassland_submit.py)")
        return

    pools = ["crop", "grass"] if args.pool == "both" else [args.pool]
    print(f"model grid {len(lat)}x{len(lon)}; pools {pools}")
    for pool in pools:
        last = CROP_LAST if pool == "crop" else GRASS_LAST
        yrs = [args.year] if args.year else list(range(max(args.start, YEAR0), min(args.end, last) + 1))
        for year in yrs:
            if pool == "crop":
                land_w, fb_w = crop_placement_layers(cfg, args.placement, year, crop_basis, shape, lat, lon)
                w = build_placed(zhang_density_crop_asc(year), land_w, combos, cellarea_ha, area5_km2, shape,
                                 fallback_area=fb_w, area_fallback=(args.placement != "hyde"))
                if w is None:
                    print(f"  [skip] crop {year}: source not found"); continue
                out = save_crop(year, w, lat, lon, placement=args.placement,
                                placement_tag=placement_attr(cfg, args.placement))
                print(f"  wrote {out.name}  (placed; global {w.sum()/1e9:.2f} Tg)")
                if args.mass_check:
                    mass_check(w, zhang_density_crop_asc(year), area5_km2, "cropland")
            elif gpw_present:                                          # grassland: GPW dasymetric, year-matched
                ga = gpw_grassland_area(shape, cellarea_ha, "broad", year=year)
                w = build_placed(zhang_density_grass_asc(year), ga, combos, cellarea_ha, area5_km2, shape,
                                 fallback_area=noncrop_area)
                if w is None:
                    print(f"  [skip] grass {year}: source not found"); continue
                out = save_grass_placed(year, w, lat, lon)
                print(f"  wrote {out.name}  (placed via GPW; global {w.sum()/1e9:.2f} Tg)")
                if args.mass_check:
                    mass_check(w, zhang_density_grass_asc(year), area5_km2, "grassland")
            else:                                                     # grassland fallback: 5' density
                f = build_grass_density(year, lat, lon)
                if f is None:
                    print(f"  [skip] grass {year}: source not found"); continue
                out = save_grass(year, f, lat, lon)
                print(f"  wrote {out.name}  (5' density fallback; max {f.max():.1f} kg N km-2)")


if __name__ == "__main__":
    main()
