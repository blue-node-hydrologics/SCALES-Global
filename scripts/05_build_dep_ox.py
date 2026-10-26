#!/usr/bin/env python3
"""Aggregate ISIMIP NOy deposition raster -> coefficients/dep_ox_country.csv (kg N/ha).

ISIMIP3a ndep-noy_histsoc is monthly, units g N m-2 month-1, time = 'months since 1850'.
Steps: pick the total-NOy recent file, sum the 12 months of the proxy year -> g N m-2 yr-1,
x10 -> kg N ha-1 yr-1, then cos-latitude area-weighted mean within each country (Natural
Earth via regionmask). NHx (reduced N) excluded from the N surplus per the gTREND convention.
2015-2021 are held at the 2014 value in ISIMIP3a, so the 2021 field == 2014 == our 2022 proxy.

Usage:  pip install xarray netCDF4 regionmask ;  python scripts/05_build_dep_ox.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import os; os.chdir(ROOT)
from src import config as cfgmod  # noqa: E402

PROXY_YEAR = 2021         # latest in file; == 2014 held-constant == 2022 proxy
G_M2_TO_KG_HA = 10.0      # 1 g/m2 = 10 kg/ha


def _find_noy(dep_dir: Path) -> Path:
    # total NOy (not dry/wet), most recent period
    cands = [p for p in dep_dir.glob("*noy*.nc")
             if "dry" not in p.name and "wet" not in p.name]
    cands = sorted(cands, key=lambda p: ("1901" not in p.name, p.name))
    if not cands:
        raise SystemExit(f"No total-NOy NetCDF in {dep_dir} (need ndep-noy_histsoc_*.nc).")
    return cands[0]


def main():
    import xarray as xr, regionmask
    cfg = cfgmod.load_config()
    dep_dir = cfgmod.get_path(cfg, "raw") / "deposition"
    out_csv = cfgmod.get_path(cfg, "coefficients") / "dep_ox_country.csv"

    path = _find_noy(dep_dir)
    ds = xr.open_dataset(path, decode_times=False)
    var = [v for v in ds.data_vars if v != "time_bnds"][0]
    print(f"  file={path.name} var={var} units={ds[var].attrs.get('units')}")

    t = ds["time"].values                       # months since 1850-01
    sel = np.where((t >= (PROXY_YEAR - 1850) * 12) & (t < (PROXY_YEAR - 1849) * 12))[0]
    annual = ds[var].isel(time=sel).sum("time") * G_M2_TO_KG_HA   # g/m2/yr -> kg/ha/yr
    annual = annual.rename({c: "lon" for c in annual.coords if c.lower() in ("longitude", "x")})
    annual = annual.rename({c: "lat" for c in annual.coords if c.lower() in ("latitude", "y")})

    countries = regionmask.defined_regions.natural_earth_v5_0_0.countries_110
    mask = countries.mask(annual["lon"], annual["lat"])
    w = np.cos(np.deg2rad(annual["lat"])).broadcast_like(annual)

    # regionmask abbrevs are NOT ISO3 ("US","F","INDO") -> derive ISO3 from country NAMES
    import logging, country_converter as cc
    logging.getLogger("country_converter").setLevel(logging.CRITICAL)
    iso3 = cc.CountryConverter().convert(list(countries.names), to="ISO3", not_found=None)

    rows = []
    for i, code in enumerate(iso3):
        if not code or code == "not found":
            continue
        rm = mask == i
        if not bool(rm.any()):
            continue
        sub, ww = annual.where(rm), w.where(rm)
        denom = float(ww.sum())
        if denom <= 0 or np.isnan(denom):
            continue
        rows.append({"ISO3": str(code), "dep_ox_kgha": float((sub * ww).sum()) / denom,
                     "year_basis": f"{PROXY_YEAR}(=2014 held)=2022 proxy", "status": "ACTIVE",
                     "source": "ISIMIP3a NOy histsoc (DOI 10.48364/ISIMIP.759077)"})
    out = pd.DataFrame(rows).drop_duplicates("ISO3")
    out.to_csv(out_csv, index=False)
    print(f"  wrote {len(out)} countries -> {out_csv}")
    print(f"  global mean DEP_ox = {out.dep_ox_kgha.mean():.1f} kg N/ha  "
          f"(min {out.dep_ox_kgha.min():.1f}, max {out.dep_ox_kgha.max():.1f})")


if __name__ == "__main__":
    main()
