#!/usr/bin/env python3
"""
# Summary statistics for the annual SCALES-Global grids

Reads `data/outputs/grids/surplus_N_<year>_0p05.nc` for each year and writes one row per year to
`data/outputs/grid_summary_N.csv`, for checking a full rerun. Columns:

- `cropland_Mha`: total cropland (HYDE 3.3 extent);
- `<term>_Tg`: global total of each mass variable (crop removal, fertilizer, manure, BNF, deposition, ...);
- `cells_cropland`, `cells_reported`, `area_reported_pct`: cells with any cropland, cells that pass the
  minimum-cropland mask (`cropland_frac` >= 0.01), and the share of cropland area in reported cells;
- `surplus_reported_Tg`: cropland surplus summed over reported cells;
- `neg_cells`, `neg_pct`, `deficit_Tg`: negative cropland-surplus cells among reported cells;
- `gt1000_cells`, `gt1000_pct`: reported cells above 1,000 kg N ha-1 cropland;
- `p01`, `p50`, `p99`: percentiles of the reported cropland surplus (kg N ha-1);
- `landscape_neg_pct`: share of land cells with a negative landscape surplus;
- `masked`: whether the file carries the `min_cropland_frac` attribute;
- `zhang_placement`: `placement` attribute of the Zhang cropland-manure field used for the year
  (2014 field for 2015-2023).

Usage (from the model root): `python scripts/27_grid_summary.py --start 1961 --end 2023`
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
from src import config as cfgmod  # noqa: E402

MASS = ["crop", "fert", "man_crop", "bnf", "dep", "hum_crop", "fert_grass", "man_past", "forage",
        "bnf_grass", "dep_land"]


def summarise(ncf, zhang_dir, year):
    import xarray as xr
    with xr.open_dataset(ncf) as d:
        row = {"year": year, "masked": "min_cropland_frac" in d.attrs}
        A = d["cropland_ha"].values.astype("float64")
        sc = d["surplus_cropland_kgha"].values.astype("float64")
        row["cropland_Mha"] = A.sum() / 1e6
        for v in MASS:
            if f"{v}_kg" in d:
                row[f"{v}_Tg"] = float(np.nansum(d[f"{v}_kg"].values.astype("float64"))) / 1e9
        has = A > 0
        rep = has & np.isfinite(sc)
        S = sc[rep] * A[rep]
        row.update({
            "cells_cropland": int(has.sum()), "cells_reported": int(rep.sum()),
            "area_reported_pct": 100 * A[rep].sum() / max(A[has].sum(), 1),
            "surplus_reported_Tg": S.sum() / 1e9,
            "neg_cells": int((S < 0).sum()), "neg_pct": 100 * float((S < 0).mean()) if S.size else np.nan,
            "deficit_Tg": S[S < 0].sum() / 1e9,
            "gt1000_cells": int((sc[rep] > 1000).sum()),
            "gt1000_pct": 100 * float((sc[rep] > 1000).mean()) if S.size else np.nan,
        })
        for q in (1, 50, 99):
            row[f"p{q:02d}"] = float(np.percentile(sc[rep], q)) if S.size else np.nan
        ls = d["surplus_landscape_kgha"].values
        lm = np.isfinite(ls)
        row["landscape_neg_pct"] = 100 * float((ls[lm] < 0).mean())
    zf = zhang_dir / f"zhang_man_crop_{min(year, 2014)}_0p05.nc"
    try:
        import xarray as xr
        with xr.open_dataset(zf) as z:
            row["zhang_placement"] = z["man_weight"].attrs.get("placement", "unset")
    except Exception as e:  # noqa: BLE001
        row["zhang_placement"] = f"unreadable ({type(e).__name__})"
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=1961)
    ap.add_argument("--end", type=int, default=2023)
    args = ap.parse_args()
    cfg = cfgmod.load_config()
    grid_dir = cfgmod.get_path(cfg, "outputs") / "grids"
    zhang_dir = Path(cfgmod.get_path(cfg, "raw")) / "zhang_manure" / "derived"
    rows = []
    for year in range(args.start, args.end + 1):
        ncf = grid_dir / f"surplus_N_{year}_0p05.nc"
        if not ncf.exists():
            print(f"  {year}: no grid"); continue
        try:
            r = summarise(ncf, zhang_dir, year)
        except Exception as e:  # noqa: BLE001
            print(f"  {year}: ERROR {e}"); continue
        rows.append(r)
        print(f"  {year}: cropland {r['cropland_Mha']:.0f} Mha, neg {r['neg_pct']:.1f}%, "
              f">1000 {r['gt1000_pct']:.2f}%, masked={r['masked']}, zhang={r['zhang_placement']}", flush=True)
    out = cfgmod.get_path(cfg, "outputs") / "grid_summary_N.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
