#!/usr/bin/env python3
"""Build the Coello crop-specific fertilizer WEIGHT field for SCALES-Global
(CROP-SPECIFIC DASYMETRIC downscaling via CropGRIDS harvested area, AREA-CONSERVATIVE overlap).

Coello et al. (2025, Sci Data 12:44) give crop-specific fertilizer applied on a 5-arcmin grid,
annually 1961-2019, for 13 crop groups x 3 nutrients (N, P2O5, K2O):
    data/raw/coello_fertilization/.../Cropland_Maps/<Crop>_<Nut>_<year>.tiff
    (2160 x 4320, EPSG:4326, north-up, 1/12 deg)

UNITS (verified in-session): each pixel is **kg nutrient per hectare of GRID CELL** -- an areal
mass DENSITY that already embeds Coello's own cropland map, so the mass in a 5-arcmin cell k is
d_fert(c,k,t) * AREA5(k).

METHOD (Methods Eqs 8-11) -- crop-specific dasymetric, mass-conserving within each 5-arcmin cell:
    M_fert(c,k,t) = d_fert(c,k,t) * AREA5(k)                                 (Eq. 8)
    w(c,m)        = A_harv(c,m) / Σ_{m'⊂k} A_harv(c,m')                      (Eq. 9)
    FERT_crop(m,t)= Σ_c  M_fert(c,k(m),t) · w(c,m)                           (Eqs 10, 12)
Each Coello crop group's mass in a 5-arcmin cell is redistributed onto the 0.05deg cells IN
PROPORTION TO WHERE THAT CROP IS GROWN, using CropGRIDS **harvested-area** maps (native 0.05deg,
static ~2020). Harvested (not physical) area weights multiply-cropped land by its cropping events.

GRID ASSIGNMENT -- AREA-CONSERVATIVE OVERLAP (not centre assignment).
The 5-arcmin (1/12 deg) and 0.05deg (1/20 deg) grids do NOT nest (ratio 1.667 per axis). If each
0.05deg cell were handed WHOLLY to the 5-arcmin cell containing its centre, the variable child count
per parent (1, 2 or 4) beats against the fertilizer field and prints a regular grid artifact. Instead
each 0.05deg cell is split between the 5-arcmin cells it overlaps, weighted by overlap AREA (separable
per axis). This preserves each 5-arcmin cell's Coello total (mass conserving) AND is continuous, so a
uniformly cropped landscape downscales to a smooth field with no lattice artifact. (For a uniform crop
and density, acc(m) reduces exactly to density * area(m) -- see selftest.)

FALLBACKS per 5-arcmin cell (Eq. 11 + text), applied to the mass that has no crop-specific target:
  crop harvarea (Eq. 9)  ->  total-cropland harvarea (13 groups)  ->  CropGRIDS croparea.
Any residual with no cropland anywhere is left to allocate() (national FAOSTAT re-normalization).

STATIC INPUTS: CropGRIDS is a single circa-2020 harvested-area map (no time axis), so the overlap
weights and the per-5-arcmin harvarea sums Z(c,k) are computed ONCE and reused for every year; Coello
supplies the time-varying crop-specific magnitudes. After 2019 the caller holds the 2019 field fixed.

The national MAGNITUDE stays FAOSTAT-RFN x our CF -- allocate() re-normalizes this field within each
country, so it only sets WITHIN-country geography. Global total is identical to the crop-summed build.

Output (unchanged contract -- load_coello_fert() needs no change):
    data/raw/coello_fertilization/derived/coello_Nfert_<year>_0p05.nc     (var nfert_applied_kg)

Usage:
    python scripts/12b_build_coello_fert.py                    # N, all years 1961-2019, + validation
    python scripts/12b_build_coello_fert.py --nutrient P2O5
    python scripts/12b_build_coello_fert.py --year 2010 --mass-check
    python scripts/12b_build_coello_fert.py --compare-map 2015 # before/after map vs existing file (no write)
    python scripts/12b_build_coello_fert.py --selftest         # synthetic overlap unit test (no data files)
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import os
os.chdir(ROOT)
from src import config as cfgmod          # noqa: E402
from src import downscale as ds           # noqa: E402

COE_DIR = ROOT / "data/raw/coello_fertilization"
DERIVED = COE_DIR / "derived"
CG_DIR = ROOT / "data/raw" / ds.CG_SUBDIR                 # CropGRIDS NC maps (same 0.05deg model grid)
XWALK = ROOT / "coefficients/coello_cropgrids_crosswalk.csv"
YEAR0, YEAR1 = 1961, 2019                 # Coello coverage
COE_RES = 1.0 / 12.0                      # 5-arcmin
COE_NLAT, COE_NLON = 2160, 4320

CROPS = ["Fiber crops", "Fruits", "Maize", "Other Cereals", "Other Oilseeds", "Other crops",
         "Palm Oil fruit", "Rice", "Roots and tubers", "Soybean", "Sugar crops", "Vegetables",
         "Wheat"]


def maps_dir() -> Path:
    hits = sorted(COE_DIR.glob("**/Cropland_Maps"))
    if not hits:
        sys.exit(f"ERROR: no Cropland_Maps/ under {COE_DIR}. Unzip the Figshare deposit there.")
    return hits[0]


def crosswalk_groups() -> dict:
    """{coello_group -> [CropGRIDS crop slugs]}, EXCLUDE_forage dropped."""
    x = pd.read_csv(XWALK)
    x = x[x["coello_group"] != "EXCLUDE_forage"]
    g = {grp: sorted(sub["cropgrids_crop"].tolist()) for grp, sub in x.groupby("coello_group")}
    missing = set(CROPS) - set(g)
    if missing:
        sys.exit(f"ERROR: crosswalk missing Coello groups: {sorted(missing)}")
    return g


def coello_area5_ha(lat_asc_5min):
    """5-arcmin cell area (ha) per latitude row, ascending."""
    return (ds._cell_area_ha(lat_asc_5min, COE_RES, COE_RES) / 1e4)   # m2 -> ha, len 2160


# ---------- area-conservative overlap: 0.05deg cell -> up-to-2 5-arcmin cells per axis ----------
def axis_overlap(centers, mres, cres, cn, origin):
    """For each model cell (centre, width mres), the up-to-2 coarse cells (width cres) it overlaps,
    with the FRACTION OF THE MODEL CELL area in each. Returns iA, fA, iB, fB (fB=0 if no straddle).
    Valid because mres < cres (a model cell spans at most two coarse cells)."""
    lo = centers - mres / 2.0
    hi = centers + mres / 2.0
    iA = np.clip(np.floor((lo - origin) / cres).astype("int64"), 0, cn - 1)
    iH = np.clip(np.floor((hi - origin) / cres).astype("int64"), 0, cn - 1)
    bnd = origin + (iA + 1) * cres                                   # right edge of coarse cell iA
    fA = np.clip((np.minimum(hi, bnd) - lo) / mres, 0.0, 1.0)
    same = iH == iA
    iB = np.minimum(iA + 1, cn - 1)
    fA = np.where(same, 1.0, fA)
    fB = np.where(same, 0.0, 1.0 - fA)
    return iA, fA, iB, fB


def build_overlap(lat, lon):
    """Up to 4 (pidx 2d int32, w 2d float32) combos: area-conservative overlap of each 0.05deg cell
    with the 5-arcmin cells. Σ over combos of w == 1 per model cell (a partition of its area)."""
    mres_lat = float(abs(lat[1] - lat[0]))
    mres_lon = float(abs(lon[1] - lon[0]))
    rA, fA, rB, fB = axis_overlap(lat, mres_lat, COE_RES, COE_NLAT, -90.0)
    cA, gA, cB, gB = axis_overlap(lon, mres_lon, COE_RES, COE_NLON, -180.0)
    combos = []
    for r, f in [(rA, fA), (rB, fB)]:
        for c, g in [(cA, gA), (cB, gB)]:
            pidx = (r[:, None] * COE_NLON + c[None, :]).astype("int32")
            w = (f[:, None] * g[None, :]).astype("float32")
            combos.append((pidx, w))
    return combos


def _scatter(field2d, combos, npar):
    """Z(p) = Σ_m w_{m,p} · field(m)  (0.05deg field -> per 5-arcmin-cell sum)."""
    z = np.zeros(npar, "float64")
    for pidx, w in combos:
        z += np.bincount(pidx.ravel(), weights=(w * field2d).ravel().astype("float64"), minlength=npar)
    return z


def _gather(coarse_flat, combos, shape):
    """g(m) = Σ_p w_{m,p} · coarse(p)  (per 5-arcmin-cell quantity -> 0.05deg, area-weighted)."""
    out = np.zeros(shape, "float64")
    for pidx, w in combos:
        out += w * coarse_flat[pidx]
    return out


# ---------- static per-crop harvarea + per-5-arcmin-cell sums (computed once) ----------
def build_static(groups, combos, shape):
    """Per-crop 0.05deg harvarea G[grp] and its per-5-arcmin-cell overlap sum Z[grp]=Σ_m w G; plus
    total-cropland (13 groups) and CropGRIDS-croparea fallbacks. All static (CropGRIDS ~2020)."""
    npar = COE_NLAT * COE_NLON
    G, Z = {}, {}
    G_tot = np.zeros(shape, "float32")
    for grp in CROPS:
        acc = np.zeros(shape, "float32")
        for slug in groups[grp]:
            a = ds._layer(CG_DIR, slug, "harvarea")
            if a is not None:
                acc += a
        G[grp] = acc
        Z[grp] = _scatter(acc, combos, npar)
        G_tot += acc
    Z_tot = _scatter(G_tot, combos, npar)
    G_crp = np.zeros(shape, "float32")
    for slug in pd.read_csv(XWALK)["cropgrids_crop"]:
        a = ds._layer(CG_DIR, slug, "croparea")
        if a is not None:
            G_crp += a
    Z_crp = _scatter(G_crp, combos, npar)
    return dict(G=G, Z=Z, G_tot=G_tot, Z_tot=Z_tot, G_crp=G_crp, Z_crp=Z_crp)


# ---------- per-year assembly ----------
def coello_density_crop(mdir, crop, nutrient, year):
    """Single-crop 5-arcmin density (kg/ha of cell), flipped to lat-ascending. None if absent."""
    import rasterio
    p = mdir / f"{crop}_{nutrient}_{year}.tiff"
    if not p.exists():
        return None
    with rasterio.open(p) as s:
        a = s.read(1).astype("float64")
    a = np.where(np.isfinite(a) & (a > 0), a, 0.0)
    return a[::-1, :]                                    # north-up -> ascending


def build_year(mdir, nutrient, year, S, combos, area5_ha, shape):
    """FERT_crop(m,t) = Σ_c FERT_c(m); FERT_c distributes each 5-arcmin cell's Coello mass M5_c over
    the overlapping 0.05deg cells weighted by that crop's harvested area (Eqs 8-12), mass-conserving."""
    npar = COE_NLAT * COE_NLON
    acc = np.zeros(shape, "float64")
    any_crop = False
    for grp in CROPS:
        d5 = coello_density_crop(mdir, grp, nutrient, year)             # 2160x4320 ascending density
        if d5 is None:
            continue
        any_crop = True
        M5 = (d5 * area5_ha[:, None]).ravel()                          # mass per 5-arcmin cell (Eq. 8)
        Zc, Zt, Zp = S["Z"][grp], S["Z_tot"], S["Z_crp"]
        useC = Zc > 0
        useT = (~useC) & (Zt > 0)
        useP = (~useC) & (~useT) & (Zp > 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            Rc = np.where(useC, M5 / np.where(useC, Zc, 1.0), 0.0)     # crop-specific stream
            Rt = np.where(useT, M5 / np.where(useT, Zt, 1.0), 0.0)     # total-cropland fallback
            Rp = np.where(useP, M5 / np.where(useP, Zp, 1.0), 0.0)     # croparea fallback
        acc += (S["G"][grp] * _gather(Rc, combos, shape)
                + S["G_tot"] * _gather(Rt, combos, shape)
                + S["G_crp"] * _gather(Rp, combos, shape))
    return acc if any_crop else None


def save_year(nutrient, year, total, lat, lon):
    import xarray as xr
    DERIVED.mkdir(parents=True, exist_ok=True)
    da = xr.DataArray(total.astype("float32"), coords={"lat": lat, "lon": lon}, dims=("lat", "lon"),
                      name="nfert_applied_kg",
                      attrs={"units": f"kg {nutrient} cell-1 yr-1",
                             "long_name": f"Coello crop-specific {nutrient} applied per cell "
                                          "(crop-specific dasymetric, area-conservative overlap)",
                             "source": "Coello et al. 2025 Sci Data 12:44 (5-arcmin, 13 crop groups) "
                                       "dasymetrically downscaled to 0.05deg on CropGRIDS harvarea",
                             "method": "M_fert=density*AREA5 redistributed to 0.05deg by crop harvarea "
                                       "via area-conservative overlap; fallback total-cropland->croparea; "
                                       "mass-conserving per 5-arcmin cell",
                             "year": year})
    out = DERIVED / f"coello_{nutrient}fert_{year}_0p05.nc"
    da.to_netcdf(out, encoding={"nfert_applied_kg": {"zlib": True, "complevel": 4, "dtype": "float32"}})
    return out


def validate(cfg, nutrient, year, total, mask, idx2iso):
    """Sanity: national sums vs FAOSTAT-RFN (magnitudes must not move vs the crop-summed build)."""
    if nutrient != "N":
        print(f"  [validate {year}] global Coello {nutrient} = {total.sum()/1e9:.1f} Tg")
        return
    from src import fertilizer
    nidx = int(mask.max()) + 1
    nat = np.bincount(mask.ravel().clip(min=0), weights=total.ravel(), minlength=nidx)
    coe = pd.DataFrame({"ISO3": [idx2iso.get(i) for i in range(len(nat))], "coe_kg": nat}).dropna()
    rfn = fertilizer.compute(cfg, cfgmod.get_path(cfg, "raw"), year, "N").rename(columns={"fert_kg": "rfn_kg"})
    m = coe.merge(rfn, on="ISO3", how="inner")
    m = m[(m["coe_kg"] > 0) & (m["rfn_kg"] > 0)]
    r = np.corrcoef(np.log10(m["coe_kg"]), np.log10(m["rfn_kg"]))[0, 1] if len(m) > 2 else np.nan
    print(f"  [validate {year}] global Coello N = {total.sum()/1e9:.1f} Tg  |  "
          f"FAOSTAT-RFN cropland = {rfn['rfn_kg'].sum()/1e9:.1f} Tg  |  "
          f"ratio = {total.sum()/max(rfn['rfn_kg'].sum(), 1):.3f}  |  "
          f"national log-log r = {r:.3f} (n={len(m)})")


def mass_check(total, mdir, nutrient, year, area5_ha, combos):
    """GLOBAL conservation is exact by construction (mass assigned to each 5-arcmin cell = M5). This
    also reports a spatial-fidelity number: area-reaggregating the 0.05deg field back to 5-arcmin and
    comparing to Coello M5 (nonzero only at cell boundaries, where straddling cells share mass)."""
    npar = COE_NLAT * COE_NLON
    recov = _scatter(total, combos, npar)                              # area re-binned to 5-arcmin cells
    M5tot = np.zeros(npar)
    for grp in CROPS:
        d5 = coello_density_crop(mdir, grp, nutrient, year)
        if d5 is not None:
            M5tot += (d5 * area5_ha[:, None]).ravel()
    m = M5tot > 0
    rel = np.abs(recov[m] - M5tot[m]) / M5tot[m]
    print(f"  [mass-check {year}] GLOBAL new={total.sum()/1e9:.3f} Tg vs Coello={M5tot.sum()/1e9:.3f} Tg "
          f"(diff {100*(total.sum()-M5tot.sum())/M5tot.sum():+.3f}%, = mass on no-cropland cells)  |  "
          f"5-arcmin spatial fidelity: median rel err={np.median(rel):.2e} p99={np.percentile(rel,99):.2e}")


# ---------- diagnostics ----------
def selftest():
    """Synthetic overlap test at the REAL 1/12 vs 1/20 grid ratio: (a) uniform density+harvarea ->
    uniform output (no grid artifact) & mass conserved; (b) crop confined to half the cells -> output
    tracks the crop and mass still conserved."""
    global COE_NLAT, COE_NLON, COE_RES
    COE_NLAT, COE_NLON, COE_RES = 12, 12, 1.0 / 12.0                   # 1x1 deg patch of 5-arcmin cells
    lat = np.arange(0 + 0.025, 1.0, 0.05)                             # 20 model cells (0.05deg) in [0,1]
    lon = np.arange(0 + 0.025, 1.0, 0.05)
    # override origins to 0 for the patch by shifting axis_overlap via temporary -90/-180 offset:
    combos = []
    rA, fA, rB, fB = axis_overlap(lat, 0.05, COE_RES, COE_NLAT, 0.0)
    cA, gA, cB, gB = axis_overlap(lon, 0.05, COE_RES, COE_NLON, 0.0)
    for r, f in [(rA, fA), (rB, fB)]:
        for c, g in [(cA, gA), (cB, gB)]:
            combos.append(((r[:, None] * COE_NLON + c[None, :]).astype("int32"),
                           (f[:, None] * g[None, :]).astype("float32")))
    shape = (len(lat), len(lon)); npar = COE_NLAT * COE_NLON
    area5 = coello_area5_ha(np.arange(COE_NLAT) * COE_RES + COE_RES / 2)

    def run(G, dens):
        Z = _scatter(G, combos, npar)
        M5 = (dens * area5[:, None]).ravel()
        with np.errstate(divide="ignore", invalid="ignore"):
            R = np.where(Z > 0, M5 / np.where(Z > 0, Z, 1.0), 0.0)
        out = G * _gather(R, combos, shape)
        return out, M5, Z

    # (a) uniform density + harvarea -> smooth output (no grid artifact) and mass conserved.
    # Residual CV is the real latitudinal cell-area gradient across the patch, not a lattice artifact.
    G = np.ones(shape, "float32"); dens = np.full((COE_NLAT, COE_NLON), 3.0)
    out, M5, Z = run(G, dens)
    cv = out.std() / out.mean()
    ok_a = cv < 1e-3 and np.isclose(out.sum(), M5[Z > 0].sum())
    print(f"  [selftest a] uniform: output CV={cv:.2e} (old stripe would be ~5e-1)  "
          f"global mass ok={np.isclose(out.sum(), M5[Z>0].sum())}")
    # (b) crop confined to a checkerboard -> all mass lands on cropped cells; the mass of every
    # 5-arcmin cell that overlaps crop (Z>0) is conserved (Σ_children of p's provenance == M5(p)).
    G = np.zeros(shape, "float32"); G[::2, ::2] = 1.0
    out, M5, Z = run(G, dens)
    conc = out[G > 0].sum() / out.sum()
    ok_b = np.isclose(conc, 1.0) and np.isclose(out.sum(), M5[Z > 0].sum())
    print(f"  [selftest b] crop-confined: {100*conc:.1f}% of mass on cropped cells  "
          f"global mass ok={np.isclose(out.sum(), M5[Z>0].sum())}")
    assert ok_a and ok_b, "SELFTEST FAILED"
    print("  [selftest] PASSED")


def compare_map(cfg, nutrient, year, S, combos, area5_ha, lat, lon, shape):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    import xarray as xr
    new = build_year(maps_dir(), nutrient, year, S, combos, area5_ha, shape)
    old_f = DERIVED / f"coello_{nutrient}fert_{year}_0p05.nc"
    old = xr.open_dataset(old_f)["nfert_applied_kg"].values if old_f.exists() else np.full_like(new, np.nan)
    la = (lat > 38) & (lat < 46); lo = (lon > -100) & (lon < -84)
    sub = np.ix_(la, lo); ext = [lon[lo][0], lon[lo][-1], lat[la][0], lat[la][-1]]
    vmax = np.nanpercentile(new[sub], 99)
    fig, ax = plt.subplots(1, 2, figsize=(12, 5.2))
    for a_, img, ttl in [(ax[0], old[sub], "OLD  (nearest, crop-summed)"),
                         (ax[1], new[sub], "NEW  (crop-specific, area-conservative)")]:
        im = a_.imshow(img, origin="lower", extent=ext, vmin=0, vmax=vmax, cmap="YlGnBu", aspect="auto")
        a_.set_title(ttl, fontsize=12, fontweight="bold"); fig.colorbar(im, ax=a_, shrink=0.8, label="kg N/cell")
    fig.suptitle(f"Coello {nutrient} fertilizer downscaling — {year} (US corn belt)", fontweight="bold")
    fig.tight_layout()
    out = ROOT / "data/outputs/figures" / f"coello_dasymetric_compare_{year}.png"
    out.parent.mkdir(parents=True, exist_ok=True); fig.savefig(out, dpi=170, facecolor="white")
    print(f"  wrote {out}")
    print(f"  whole-grid global this year: old={np.nansum(old)/1e9:.3f} Tg  new={np.nansum(new)/1e9:.3f} Tg")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nutrient", default="N", choices=["N", "P2O5", "K2O"])
    ap.add_argument("--year", type=int)
    ap.add_argument("--start", type=int, default=YEAR0)
    ap.add_argument("--end", type=int, default=YEAR1)
    ap.add_argument("--no-validate", action="store_true")
    ap.add_argument("--mass-check", action="store_true")
    ap.add_argument("--compare-map", type=int, metavar="YEAR")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        selftest(); return

    cfg = cfgmod.load_config()
    mask, idx2iso, lat, lon = ds.load_country_grid(cfg)
    shape = (len(lat), len(lon))
    combos = build_overlap(lat, lon)
    lat5 = -90.0 + (np.arange(COE_NLAT) + 0.5) * COE_RES
    area5_ha = coello_area5_ha(lat5)
    groups = crosswalk_groups()
    print(f"model grid {shape[0]}x{shape[1]}; building CropGRIDS harvarea + overlap sums ...")
    S = build_static(groups, combos, shape)
    print(f"  built static harvarea/overlap for {len(S['G'])} crop groups")

    if args.compare_map is not None:
        compare_map(cfg, args.nutrient, args.compare_map, S, combos, area5_ha, lat, lon, shape); return

    mdir = maps_dir()
    print(f"Coello maps: {mdir}")
    years = [args.year] if args.year else list(range(max(args.start, YEAR0), min(args.end, YEAR1) + 1))
    for year in years:
        total = build_year(mdir, args.nutrient, year, S, combos, area5_ha, shape)
        if total is None:
            print(f"  [skip] no Coello tiffs for {args.nutrient} {year}")
            continue
        out = save_year(args.nutrient, year, total, lat, lon)
        print(f"wrote {out.name}  (global {total.sum()/1e9:.1f} Tg applied)")
        if args.mass_check:
            mass_check(total, mdir, args.nutrient, year, area5_ha, combos)
        if not args.no_validate:
            validate(cfg, args.nutrient, year, total, mask, idx2iso)


if __name__ == "__main__":
    main()
