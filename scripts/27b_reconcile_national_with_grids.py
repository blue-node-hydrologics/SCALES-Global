#!/usr/bin/env python3
"""
# Make the national budget tables consistent with the gridded budget

The national pipeline (`17_run_timeseries.py` -> `pipeline.compute_balance`) computes two terms in a
different way from the grids:

- **forage removal**: national Eq. 29 (legacy) instead of the grid-scale Eq. 29 applied cell by cell;
- **deposition on all land**: national cropland and non-cropland areas from FAOSTAT, instead of the
  HYDE cropland and grid-cell areas of the grids.

This script replaces `forage_kg` and `dep_land_kg` in every `data/outputs/mass_balance_N_<year>.csv` by
the country sums of the same variables in `grids/surplus_N_<year>_0p05.nc`,
using the country mask of the downscale weight cache. All per-hectare columns and the three surpluses
are then recomputed with `src/mass_balance.assemble`, so the surplus formulas are exactly those of the
pipeline. Countries absent from the grid mask (a few small islands) keep their national values. Cropland
deposition keeps its national value: it is the WL rate per ha of cropland, applied to FAOSTAT cropland
nationally and to HYDE cropland in the grids, so the two totals differ through the cropland areas only.

`data/outputs/timeseries_N_global.csv` is then rebuilt from the reconciled tables with the same
global-row formulas as `17_run_timeseries.py`.

The original files are copied once to `data/outputs/national_pre_grid_reconcile/` before anything is
overwritten; rerunning the script reconciles again from the current grids (it is idempotent).

Run it after `26_run_historical_grids.py` and `17_run_timeseries.py`; rerun it whenever either is rerun.

Usage (from the model root): `python scripts/27b_reconcile_national_with_grids.py --start 1961 --end 2023`
"""
import argparse
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
from src import config as cfgmod, downscale as ds, mass_balance  # noqa: E402

# national value := country sum of the grid. Cropland deposition (dep) is NOT replaced : it
# is a rate (Vishwakarma WL, kg N per ha of cropland) times cropland area, so the national table keeps
# WL x FAOSTAT cropland and the grid WL x HYDE cropland; the per-ha rate is identical in both.
GRID_TERMS = ["forage", "dep_land"]
COMPONENTS = (mass_balance.N_INPUTS + mass_balance.N_OUTPUTS + mass_balance.N_LOSSES + mass_balance.N_FULL)


def country_sums(ncf, mask, idx2iso):
    """Country sums (kg) of the GRID_TERMS from one annual grid -> DataFrame [ISO3, <term>_kg]."""
    import xarray as xr
    flat = mask.ravel()
    ok = flat >= 0
    k = np.where(ok, flat, 0)
    n = int(mask.max()) + 1
    out = {"ISO3": [idx2iso.get(i) for i in range(n)]}
    with xr.open_dataset(ncf) as d:
        for t in GRID_TERMS:
            v = np.nan_to_num(d[f"{t}_kg"].values.astype("float64")).ravel()
            out[f"{t}_kg"] = np.bincount(k, weights=np.where(ok, v, 0.0), minlength=n)
    df = pd.DataFrame(out).dropna(subset=["ISO3"])
    return df.groupby("ISO3", as_index=False).sum()


def reconcile_table(tab, sums):
    """Replace the GRID_TERMS for countries in the grid mask and re-assemble the national table."""
    tab = tab.copy()
    s = sums.set_index("ISO3")
    in_grid = tab["ISO3"].isin(s.index)
    for t in GRID_TERMS:
        col = f"{t}_kg"
        tab.loc[in_grid, col] = tab.loc[in_grid, "ISO3"].map(s[col]).values
    areas = tab[[c for c in ("ISO3", "AREA_HA", "LAND_HA") if c in tab.columns]]
    comps = {n: tab[["ISO3", f"{n}_kg"]] for n in COMPONENTS if f"{n}_kg" in tab.columns}
    new = mass_balance.assemble(comps, areas, "N")
    cols = list(tab.columns) + [c for c in new.columns if c not in tab.columns]
    return new.reindex(columns=cols), int(in_grid.sum()), int((~in_grid).sum())


COMPS_TS = ["fert_kg", "bnf_kg", "dep_kg", "man_crop_kg", "hum_sewage_kg", "crop_kg",
            "fert_vol_kg", "man_vol_kg",
            "fert_grass_kg", "dep_land_kg", "man_past_kg", "fert_vol_full_kg", "man_vol_full_kg",
            "forage_kg", "bnf_grass_kg", "food_waste_kg", "hum_crop_kg", "hum_ns_kg", "hum_bio_kg",
            "hum_landfill_kg", "hum_sew_water_kg", "hum_sew_land_kg"]


def global_row(year, t):
    """One row of timeseries_N_global.csv from a national table (same formulas as 17_run_timeseries)."""
    g = {c: float(t[c].sum()) / 1e9 for c in COMPS_TS if c in t}
    crop_ha = float(t["AREA_HA"].sum())
    land_ha = float(t["LAND_HA"].sum()) if "LAND_HA" in t else float("nan")
    fvol = g.get("fert_vol_kg", 0); mvol = g.get("man_vol_kg", 0)
    hum_crop_Tg = g.get("hum_crop_kg", 0)
    agron_faocomp = (g["fert_kg"] + g["bnf_kg"] + g.get("dep_kg", 0) + g["man_crop_kg"]
                     - g["crop_kg"] - fvol - mvol)
    agron = agron_faocomp + hum_crop_Tg
    fvol_full = g.get("fert_vol_full_kg", fvol); mvol_full = g.get("man_vol_full_kg", mvol)
    forage = g.get("forage_kg", 0)
    hum_sew_land_Tg = g.get("hum_sew_land_kg", 0)
    hum_sew_water_Tg = g.get("hum_sew_water_kg", g.get("hum_sewage_kg", 0))
    hum_landfill_Tg = g.get("hum_landfill_kg", 0)
    landscape = (g["fert_kg"] + g.get("fert_grass_kg", 0) + g["bnf_kg"] + g.get("bnf_grass_kg", 0)
                 + g.get("dep_land_kg", g.get("dep_kg", 0)) + g["man_crop_kg"]
                 + g.get("man_past_kg", 0) + g.get("food_waste_kg", 0) + hum_crop_Tg + hum_sew_land_Tg
                 + hum_landfill_Tg - g["crop_kg"] - forage - fvol_full - mvol_full)
    full = landscape + hum_sew_water_Tg
    ok_land = land_ha == land_ha and land_ha > 0
    return {"year": year, "fert_Tg": g["fert_kg"], "fert_grass_Tg": g.get("fert_grass_kg", 0),
            "bnf_Tg": g["bnf_kg"], "dep_Tg": g.get("dep_kg", 0), "dep_land_Tg": g.get("dep_land_kg", 0),
            "man_crop_Tg": g["man_crop_kg"], "man_past_Tg": g.get("man_past_kg", 0),
            "hum_sewage_Tg": g.get("hum_sewage_kg", 0), "hum_sew_water_Tg": hum_sew_water_Tg,
            "hum_sew_land_Tg": hum_sew_land_Tg, "crop_Tg": g["crop_kg"], "forage_Tg": forage,
            "bnf_grass_Tg": g.get("bnf_grass_kg", 0),
            "food_waste_Tg": g.get("food_waste_kg", 0), "hum_crop_Tg": hum_crop_Tg,
            "hum_ns_Tg": g.get("hum_ns_kg", 0), "hum_bio_Tg": g.get("hum_bio_kg", 0),
            "hum_landfill_Tg": hum_landfill_Tg,
            "fert_vol_Tg": fvol, "man_vol_Tg": mvol,
            "fert_vol_full_Tg": fvol_full, "man_vol_full_Tg": mvol_full,
            "surplus_cropland_Tg": agron, "surplus_cropland_faocomp_Tg": agron_faocomp,
            "surplus_landscape_Tg": landscape, "surplus_full_Tg": full,
            "cropland_Mha": crop_ha / 1e6, "land_Mha": land_ha / 1e6,
            "surplus_cropland_kgha": agron * 1e9 / crop_ha,
            "surplus_cropland_faocomp_kgha": agron_faocomp * 1e9 / crop_ha,
            "surplus_landscape_kgha": landscape * 1e9 / land_ha if ok_land else float("nan"),
            "surplus_full_kgha": full * 1e9 / land_ha if ok_land else float("nan")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=1961)
    ap.add_argument("--end", type=int, default=2023)
    args = ap.parse_args()
    cfg = cfgmod.load_config()
    out_dir = cfgmod.get_path(cfg, "outputs")
    grid_dir = out_dir / "grids"
    bak = out_dir / "national_pre_grid_reconcile"
    bak.mkdir(exist_ok=True)
    cache = ds.load_weight_cache(cfg)
    mask, idx2iso = cache["mask"], cache["idx2iso"]
    ts_path = out_dir / "timeseries_N_global.csv"
    if ts_path.exists() and not (bak / ts_path.name).exists():
        shutil.copy2(ts_path, bak / ts_path.name)
    rows = []
    for year in range(args.start, args.end + 1):
        tp = out_dir / f"mass_balance_N_{year}.csv"
        ncf = grid_dir / f"surplus_N_{year}_0p05.nc"
        if not tp.exists() or not ncf.exists():
            print(f"  {year}: missing {'table' if not tp.exists() else 'grid'}, skipped"); continue
        if not (bak / tp.name).exists():
            shutil.copy2(tp, bak / tp.name)
        old = pd.read_csv(bak / tp.name)                    # always reconcile from the pipeline output
        new, n_in, n_out = reconcile_table(old, country_sums(ncf, mask, idx2iso))
        new.to_csv(tp, index=False)
        r_old, r_new = global_row(year, old), global_row(year, new)
        rows.append(r_new)
        print(f"  {year}: forage {r_old['forage_Tg']:.1f} -> {r_new['forage_Tg']:.1f} Tg; "
              f"dep {r_old['dep_Tg']:.2f} -> {r_new['dep_Tg']:.2f}; landscape surplus "
              f"{r_old['surplus_landscape_Tg']:.1f} -> {r_new['surplus_landscape_Tg']:.1f}; cropland surplus "
              f"{r_old['surplus_cropland_Tg']:.2f} -> {r_new['surplus_cropland_Tg']:.2f} Tg "
              f"[{n_in} countries from grid, {n_out} kept national]", flush=True)
    if rows:
        ts = pd.DataFrame(rows)
        if ts_path.exists():                                # keep years outside --start/--end
            prev = pd.read_csv(ts_path)
            ts = pd.concat([prev[~prev["year"].isin(ts["year"])], ts]).sort_values("year")
        ts.to_csv(ts_path, index=False)
        print(f"wrote {ts_path} ({len(ts)} years); originals in {bak}")


if __name__ == "__main__":
    main()
