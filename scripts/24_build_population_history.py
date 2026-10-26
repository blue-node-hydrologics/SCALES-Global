#!/usr/bin/env python3
"""Build coefficients/population_country.csv (ISO3, year, population) for 1961-2023 from
FAOSTAT OA (Annual population): element 511 'Total Population - Both sexes' (1000 persons),
item 3010. Replaces the 2020/2022-only table so the human-waste term is annualized.

Usage:  python scripts/24_build_population_history.py
"""
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import os; os.chdir(ROOT)
from src import config as cfgmod, faostat_api, faostat_prep as prep  # noqa: E402


def main():
    cfg = cfgmod.load_config()
    raw = cfgmod.get_path(cfg, "raw")
    df = faostat_api.read_domain("OA", raw)
    df = faostat_api.filter_rows(df, element=511, item=3010)
    df = df[(df["Year"] >= 1961) & (df["Year"] <= 2023)]
    df = prep.to_iso3(df, cfg)
    out = df[["ISO3", "Year", "Value"]].rename(columns={"Year": "year"})
    out["population"] = out["Value"].astype(float) * 1000.0   # 1000 persons -> persons
    out = out.groupby(["ISO3", "year"], as_index=False)["population"].sum()
    out["status"] = "ACTIVE"
    out["source"] = "FAOSTAT OA element 511 item 3010 (Total population, both sexes)"
    dest = cfgmod.coef(cfg, "population_country.csv")
    out.to_csv(dest, index=False)
    print(f"wrote {dest}: {len(out)} rows, {out.ISO3.nunique()} countries, "
          f"years {int(out.year.min())}-{int(out.year.max())}")
    print(f"  global pop 1961 {out[out.year==1961].population.sum()/1e9:.2f}B  "
          f"2022 {out[out.year==2022].population.sum()/1e9:.2f}B")


if __name__ == "__main__":
    main()
