#!/usr/bin/env python3
"""
# Build the data dictionary for the deposit

Reads the packaged files in `data/outputs/zenodo/` (after `run_all.sh`) and writes
`DATA_DICTIONARY.md` and `data_dictionary.json` beside them, listing every NetCDF variable
(long name, units, storage type, packing step) and every table column.

Usage (from the repository root):

    python zenodo_prep/build_data_dictionary.py --package data/outputs/zenodo
"""
import argparse
import datetime as dt
import json
import zipfile
from pathlib import Path

import netCDF4 as nc
import pandas as pd

HEADER = """# SCALES-Global — Data Dictionary

*Generated {date} from the deposited files.*

## Grid conventions

- Spatial: 0.05 degree global grid, 3600 lat x 7200 lon, EPSG:4326, CF-1.10.
- Longitude -180 to 180, latitude -90 to 90; cell centres on the 0.05 degree grid.
- Time axis: annual, CF `days since 1900-01-01`, stamped at mid-year (1 July), with an integer `year`
  coordinate.
- Every data variable carries `grid_mapping = crs` and CF `units` and `long_name`.
- Ocean and non-applicable cells are `_FillValue` (NaN for float32; -32768 for packed int16).

## Precision and packing

- Mass variables (`*_kg`, kg N cell-1 yr-1) and `cropland_ha` are stored as float32 without loss, so
  per-cell masses sum to the national budget.
- Rate variables (`*_kgha`, kg N ha-1 yr-1) are packed to int16 with a per-variable `scale_factor` and
  `add_offset`, which CF readers apply automatically. The packing step is listed below.
- All variables use zlib level-5 compression with the shuffle filter.

## Cropland surplus rate mask

The per-hectare cropland surplus is the cropland surplus mass divided by the cropland area of the cell.
In `surplus_cropland_kgha` and the cropland uncertainty fields, cells where cropland covers less than
0.1% of the cell, or where the absolute rate exceeds 30,000 kg N ha-1 yr-1, are set to `_FillValue`.
Mass fields, the landscape and total surplus fields and the national and global tables are not masked.
"""

SECTIONS = {
    "full": "## `full/` — all grid variables (time x lat x lon, 1961-2023)",
    "lite": "## `lite/` — mapping subset\n\nThe surplus-rate fields with `cropland_ha` and a time-invariant "
            "geodesic `cell_area_ha`.",
    "uncertainty": "## `uncertainty/` — gridded uncertainty (1961, 1965, ..., 2020 and 2023)\n\n"
                   "For each surplus view: Monte Carlo mean (`mean`), standard deviation (`sd`), its parts "
                   "from coefficient uncertainty only (`coef_sd`) and from activity-data uncertainty only "
                   "(`act_sd`), and the half-width of the 95% interval (`ci95` = 1.96 x sd).",
}


def nc_rows(folder: Path):
    rows = []
    for f in sorted(folder.glob("*.nc")):
        with nc.Dataset(f) as d:
            for v, var in d.variables.items():
                if v in ("lat", "lon", "time", "year", "crs"):
                    continue
                var.set_auto_maskandscale(False)
                step = getattr(var, "scale_factor", None)
                rows.append(dict(file=f.name, variable=v, long_name=getattr(var, "long_name", v),
                                 units=getattr(var, "units", ""), dtype=str(var.dtype),
                                 int16_step=float(step) if step is not None else None))
    return rows


def table_rows(folder: Path):
    out = []
    for f in sorted(folder.glob("*.csv")):
        df = pd.read_csv(f, nrows=5)
        n = sum(1 for _ in open(f)) - 1
        out.append(dict(file=f.name, rows=n, columns=list(df.columns)))
    for z in sorted(folder.glob("*.zip")):
        with zipfile.ZipFile(z) as zf:
            out.append(dict(file=z.name, rows=None, columns=[], members=len(zf.namelist())))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", default="data/outputs/zenodo")
    a = ap.parse_args()
    pkg = Path(a.package)
    md = [HEADER.format(date=dt.date.today().isoformat())]
    js = {"generated": dt.date.today().isoformat(), "grids": {}, "tables": []}
    for sec, title in SECTIONS.items():
        rows = nc_rows(pkg / sec) if (pkg / sec).is_dir() else []
        if not rows:
            continue
        js["grids"][sec] = rows
        md += [title, "", "| file | variable | long name | units | dtype | int16 step |", "|---|---|---|---|---|---|"]
        for r in rows:
            step = f"{r['int16_step']:.3g}" if r["int16_step"] else "— (float32)"
            md.append(f"| `{r['file']}` | `{r['variable']}` | {r['long_name']} | {r['units']} | {r['dtype']} | {step} |")
        md.append("")
    if (pkg / "tables").is_dir():
        md += ["## `tables/` — national and global budgets", ""]
        for t in table_rows(pkg / "tables"):
            js["tables"].append(t)
            if t["file"].endswith(".zip"):
                md += [f"### `{t['file']}`", "", f"The per-year national tables ({t['members']} files), "
                       "one row per country, same columns as the national budget below without the uncertainty.", ""]
            else:
                md += [f"### `{t['file']}`  ({t['rows']} rows)", "",
                       "Columns: " + ", ".join(f"`{c}`" for c in t["columns"]), ""]
        md += ["### Column conventions", "",
               "- `*_kg`: national totals (kg N yr-1); `*_kgha`: per-hectare rates; `*_Tg`, `*_TgN`: Tg N.",
               "- Surplus views: `surplus_cropland_kgha` (cropland), `surplus_landscape_kgha` (landscape), "
               "`surplus_full_kgha` (total anthropogenic, including wastewater N discharged to surface water).",
               "- Uncertainty per view: `_central_TgN` (deterministic value), `_mean_TgN`, `_sd_TgN`, "
               "`_p2.5_TgN`, `_p97.5_TgN`, `_cv`; in the global series also `_p25_Tg`, `_p50_Tg`, `_p75_Tg`.",
               ""]
    (pkg / "DATA_DICTIONARY.md").write_text("\n".join(md))
    (pkg / "data_dictionary.json").write_text(json.dumps(js, indent=1))
    print(f"wrote {pkg / 'DATA_DICTIONARY.md'} and data_dictionary.json")


if __name__ == "__main__":
    main()
