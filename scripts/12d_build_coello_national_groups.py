#!/usr/bin/env python3
"""
# Build national Coello fertilizer totals per crop group (for `grid_weight: coello_cropgrids`)

## Purpose
The `coello_cropgrids` fertilizer-downscaling mode (``src/downscale.py``) uses
Coello et al. (2025, *Sci Data* 12:44) **only** for each country's split of cropland fertilizer
among its 13 crop groups. Each group is then placed on the same area basis as crop removal
(CropGRIDS harvested area x the HYDE envelope factor ``s_env``), so the fertilizer and removal maps
share one crop map. This script produces the national group totals that mode needs:

    F_C(g, k, t) = sum over model cells m in country k of the Coello mass of group g in year t.

The national fertilizer MAGNITUDE in the model is unchanged (FAOSTAT RFN x CF); these totals only
set the within-country split among crop groups.

## Method
1. **Read** each Coello tiff ``Cropland_Maps/<Group>_<N|P2O5>_<year>.tiff`` (2160 x 4320, EPSG:4326,
   north-up, 1/12 deg). Following ``scripts/12b_build_coello_fert.py`` (units verified there), each
   pixel is **kg nutrient per hectare of GRID CELL**, i.e. an areal mass density. The tiff is read
   with 12b's own reader (``coello_density_crop``: rasterio, non-finite/negative -> 0, flipped to
   lat-ascending).
2. **Mass per 5-arcmin cell**: ``M5(p) = d(p) * AREA5(p)`` with ``AREA5`` from 12b's
   ``coello_area5_ha`` (Methods Eq. 8). Its global sum is the *raw Coello global total*.
3. **Area-conservative transfer to the 0.05 deg model grid** with 12b's overlap machinery
   (``build_overlap``): each model cell m is partitioned among the (up to 4) 5-arcmin cells p it
   overlaps with area fractions ``w(m,p)`` (sum_p w = 1). Because the density is uniform within p,
   the mass landing on m is ``mass(m) = A(m) * sum_p w(m,p) d(p) = A(m) * _gather(d)(m)``, where
   ``A(m)`` is the model-cell area (ha). This is exactly 12b's ``_gather`` applied to the density.
4. **National sums** with the CropGRIDS country mask (``ds.load_country_grid``):
   ``F_C(g,k) = sum_{m in k} mass(m)``. Steps 3-4 are linear in d, so they are precomputed once as a
   sparse (country, 5-arcmin cell) operator ``C(k,p) = sum_{m in k} w(m,p) A(m)`` and each
   group-year is then ``F_C(g,.) = bincount(k, d[p] * C)``. ``--verify-gather`` recomputes one
   group-year via the explicit 0.05 deg ``_gather`` field to confirm the two agree.
5. **Mass check** per year: global total transferred to the model grid vs the raw Coello global
   sum (should agree within ~0.1%; the residual is the latitude-band approximation of the overlap
   fractions), and the mass that falls on ocean/disputed mask cells (index <= 0), which is not
   attributed to any country.

## Inputs
- ``<paths.raw>/coello_fertilization/**/Cropland_Maps/*.tiff`` (Coello et al. 2025, Figshare)
- ``<paths.raw>/cropgrids/.../Countries_2018.nc`` and ``coefficients/cropgrids_country_crosswalk.csv``
  (via ``ds.load_country_grid``)

Paths are resolved from config (``cfgmod.get_path(cfg, "raw")``), the same root
``src/downscale.py`` uses for ``coello_fertilization/derived``. (12b's module-level ``COE_DIR`` is
``ROOT/data/raw/...``; this script reads ``paths.raw`` from the config instead.)

## Output
``<paths.raw>/coello_fertilization/derived/coello_group_national_<N|P2O5>.csv``, long format:

| column | meaning |
|---|---|
| ISO3 | country (CropGRIDS mask crosswalk) |
| year | 1961-2019 |
| coello_group | one of Coello's 13 crop groups |
| fert_kg | applied fertilizer, **kg N** (N file) or **kg P2O5** (P2O5 file; not elemental P) |
| unit | ``kg N`` or ``kg P2O5`` (explicit label) |

Re-running a subset of years replaces only those years' rows in an existing file.

## Usage (from the repository root)
    python scripts/12d_build_coello_national_groups.py --nutrient N            # all years 1961-2019
    python scripts/12d_build_coello_national_groups.py --nutrient P2O5
    python scripts/12d_build_coello_national_groups.py --nutrient N --year 2000
    python scripts/12d_build_coello_national_groups.py --nutrient N --start 1990 --end 1995
    python scripts/12d_build_coello_national_groups.py --nutrient N --year 2000 --verify-gather
"""
import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import os  # noqa: E402
os.chdir(ROOT)
from src import config as cfgmod          # noqa: E402
from src import downscale as ds           # noqa: E402

# reuse 12b's tiff reader, 5-arcmin cell areas and area-conservative overlap (behaviour unchanged)
_spec = importlib.util.spec_from_file_location("coello12b", ROOT / "scripts/12b_build_coello_fert.py")
c12b = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(c12b)

YEAR0, YEAR1 = c12b.YEAR0, c12b.YEAR1          # Coello coverage 1961-2019
GROUPS = list(c12b.CROPS)                      # the 13 Coello crop groups
UNIT = {"N": "kg N", "P2O5": "kg P2O5"}


def coello_dir(cfg) -> Path:
    return Path(cfgmod.get_path(cfg, "raw")) / "coello_fertilization"


def maps_dir(cfg) -> Path:
    hits = sorted(coello_dir(cfg).glob("**/Cropland_Maps"))
    if not hits:
        sys.exit(f"ERROR: no Cropland_Maps/ under {coello_dir(cfg)}. Unzip the Figshare deposit there.")
    return hits[0]


def out_path(cfg, nutrient) -> Path:
    return coello_dir(cfg) / "derived" / f"coello_group_national_{nutrient}.csv"


def model_cell_area_ha(lat, lon):
    """Model-cell area (ha) per latitude row (same spherical formula as 12b/downscale)."""
    dlat = float(abs(lat[1] - lat[0])); dlon = float(abs(lon[1] - lon[0]))
    return ds._cell_area_ha(np.asarray(lat, "float64"), dlat, dlon) / 1e4      # m2 -> ha


def build_operator(mask, lat, lon, combos):
    """Sparse national operator for the area-conservative transfer.

    Returns (kk, pp, cc, T): for every (country k >= 1, 5-arcmin cell p) pair with overlap,
    cc = C(k,p) = sum_{m in k} w(m,p) * A(m)  [ha], so national(k) = sum_p d(p) * C(k,p).
    T(p) = sum_{all m} w(m,p) * A(m) (dense, all mask values) gives the total mass transferred to
    the model grid, so ocean/disputed mass = sum_p d T - sum national."""
    npar = c12b.COE_NLAT * c12b.COE_NLON
    area_m = np.broadcast_to(model_cell_area_ha(lat, lon)[:, None], mask.shape)
    flat_mask = mask.ravel()
    land = flat_mask >= 1
    nidx = int(mask.max()) + 1
    keys, vals = [], []
    T = np.zeros(npar, "float64")
    for pidx, w in combos:
        wa = (w.astype("float64") * area_m).ravel()
        pf = pidx.ravel().astype("int64")
        T += np.bincount(pf, weights=wa, minlength=npar)
        sel = land & (wa > 0)
        keys.append(flat_mask[sel].astype("int64") * npar + pf[sel])
        vals.append(wa[sel])
    keys = np.concatenate(keys); vals = np.concatenate(vals)
    uk, inv = np.unique(keys, return_inverse=True)
    cc = np.bincount(inv, weights=vals)
    kk = (uk // npar).astype("int64"); pp = (uk % npar).astype("int64")
    return kk, pp, cc, T, nidx


def national_for_year(mdir, nutrient, year, op, area5_ha, idx2iso):
    """Rows [ISO3, year, coello_group, fert_kg] plus mass diagnostics for one year."""
    kk, pp, cc, T, nidx = op
    rows = []
    raw_tot = grid_tot = nat_tot = 0.0
    n_groups = 0
    for grp in GROUPS:
        d5 = c12b.coello_density_crop(mdir, grp, nutrient, year)        # kg/ha of cell, lat-ascending
        if d5 is None:
            print(f"  [warn] missing tiff: {grp}_{nutrient}_{year}.tiff")
            continue
        n_groups += 1
        d = d5.ravel()
        raw_tot += float((d5 * area5_ha[:, None]).sum())                # raw Coello mass (Eq. 8)
        grid_tot += float(np.dot(d, T))                                 # mass transferred to model grid
        nat = np.bincount(kk, weights=d[pp] * cc, minlength=nidx)       # per mask index
        nat_tot += float(nat[1:].sum())
        for i in np.nonzero(nat > 0)[0]:
            iso = idx2iso.get(int(i))
            if iso is not None and i >= 1:
                rows.append((iso, year, grp, float(nat[i])))
    diag = dict(raw=raw_tot, grid=grid_tot, nat=nat_tot, n_groups=n_groups)
    return rows, diag


def verify_gather(mdir, nutrient, year, op, combos, mask, lat, lon, idx2iso):
    """Explicit 0.05 deg path for one group (12b _gather of the density x model-cell area, then
    country bincount) vs the sparse operator. Confirms the operator is the same transfer."""
    kk, pp, cc, T, nidx = op
    grp = "Maize"
    d5 = c12b.coello_density_crop(mdir, grp, nutrient, year)
    if d5 is None:
        print("  [verify] no Maize tiff for this year"); return
    area_m = model_cell_area_ha(lat, lon)[:, None]
    field = c12b._gather(d5.ravel(), combos, mask.shape) * area_m        # kg per 0.05 deg cell
    flat = mask.ravel()
    nat_g = np.bincount(flat.clip(min=0), weights=np.where(flat >= 1, field.ravel(), 0.0), minlength=nidx)
    nat_o = np.bincount(kk, weights=d5.ravel()[pp] * cc, minlength=nidx)
    rel = np.abs(nat_g - nat_o).sum() / max(nat_o.sum(), 1e-30)
    print(f"  [verify {year} {grp}] explicit _gather field = {field.sum()/1e9:.4f} Tg; national sums "
          f"gather vs operator: total {nat_g[1:].sum()/1e9:.4f} vs {nat_o[1:].sum()/1e9:.4f} Tg, "
          f"sum|diff|/total = {rel:.2e}")


def write_table(cfg, nutrient, df_new, years):
    f = out_path(cfg, nutrient)
    f.parent.mkdir(parents=True, exist_ok=True)
    if f.exists():
        old = pd.read_csv(f)
        old = old[~old["year"].isin(years)]
        df_new = pd.concat([old, df_new], ignore_index=True)
    df_new = df_new.sort_values(["year", "ISO3", "coello_group"]).reset_index(drop=True)
    df_new.to_csv(f, index=False)                    # open-write (no delete needed on the mounts)
    return f, len(df_new)


def main():
    ap = argparse.ArgumentParser(description="National Coello fertilizer per crop group (12d)")
    ap.add_argument("--nutrient", default="N", choices=["N", "P2O5"])
    ap.add_argument("--year", type=int)
    ap.add_argument("--start", type=int, default=YEAR0)
    ap.add_argument("--end", type=int, default=YEAR1)
    ap.add_argument("--verify-gather", action="store_true",
                    help="also check the sparse operator against the explicit 0.05 deg _gather field "
                         "(Maize, first requested year)")
    args = ap.parse_args()

    cfg = cfgmod.load_config()
    mdir = maps_dir(cfg)
    mask, idx2iso, lat, lon = ds.load_country_grid(cfg)
    combos = c12b.build_overlap(lat, lon)
    lat5 = -90.0 + (np.arange(c12b.COE_NLAT) + 0.5) * c12b.COE_RES
    area5_ha = c12b.coello_area5_ha(lat5)
    print(f"Coello maps: {mdir}")
    print(f"model grid {mask.shape[0]}x{mask.shape[1]}; building country x 5-arcmin overlap operator ...")
    t0 = time.time()
    op = build_operator(mask, lat, lon, combos)
    print(f"  operator: {op[0].size:,} (country, 5-arcmin cell) pairs [{time.time()-t0:.0f}s]")

    years = [args.year] if args.year else list(range(max(args.start, YEAR0), min(args.end, YEAR1) + 1))
    if args.verify_gather and years:
        verify_gather(mdir, args.nutrient, years[0], op, combos, mask, lat, lon, idx2iso)

    rows = []
    done = []
    for year in years:
        r, dg = national_for_year(mdir, args.nutrient, year, op, area5_ha, idx2iso)
        if dg["n_groups"] == 0:
            print(f"  [skip] no Coello tiffs for {args.nutrient} {year}"); continue
        rows += r; done.append(year)
        ocean = dg["grid"] - dg["nat"]
        print(f"  [{year}] {args.nutrient}: raw Coello global = {dg['raw']/1e9:.4f} Tg | on model grid = "
              f"{dg['grid']/1e9:.4f} Tg ({100*(dg['grid']-dg['raw'])/max(dg['raw'],1e-30):+.3f}%) | "
              f"national sum = {dg['nat']/1e9:.4f} Tg | ocean/disputed mask = {ocean/1e9:.4f} Tg "
              f"({100*ocean/max(dg['raw'],1e-30):.3f}%) | groups read = {dg['n_groups']}")
    if not rows:
        sys.exit("ERROR: nothing built")
    df = pd.DataFrame(rows, columns=["ISO3", "year", "coello_group", "fert_kg"])
    df["unit"] = UNIT[args.nutrient]
    f, n = write_table(cfg, args.nutrient, df, done)
    print(f"wrote {f}  ({n:,} rows total; {len(done)} year(s) rebuilt; fert_kg in {UNIT[args.nutrient]})")


if __name__ == "__main__":
    main()
