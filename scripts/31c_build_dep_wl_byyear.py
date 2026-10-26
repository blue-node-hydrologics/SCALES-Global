#!/usr/bin/env python3
"""Build the WL (Wang x LUH2) national cropland N-deposition table for the HYBRID deposition basis.

Hybrid deposition (config.deposition.basis = 'wl_hybrid'): the national deposition MAGNITUDE is
taken from Vishwakarma et al. (2023) product WL — the obs-better-validated, FAO-CNB deposition
term — while the within-country SPATIAL PATTERN in the gridded product remains the native ISIMIP3a
gradient (engine distributes the national total by the ISIMIP weight). WL is total N (NOy+NHx).

Reads data/raw/vishwakarma2023/4_WL*.xlsx (country x year, kg N/ha cropland; covers 1961-2022) and
writes coefficients/dep_wl_country_byyear.csv [ISO3, year, dep_kgha]. The engine clamps to coverage
(holds 2022 for 2023).
"""
import glob, logging
from pathlib import Path
import pandas as pd, country_converter as coco

ROOT = Path(__file__).resolve().parents[1]
logging.getLogger("country_converter").setLevel(logging.CRITICAL)


def main():
    hits = glob.glob(str(ROOT / "data/raw/vishwakarma2023/*WL*.xlsx"))
    if not hits:
        raise SystemExit("WL file not found in data/raw/vishwakarma2023/ (need 4_WL*.xlsx)")
    df = pd.read_excel(sorted(hits)[0])
    ycols = [c for c in df.columns if str(c).strip().isdigit() and 1900 < int(str(c)) < 2100]
    ccol = "Countries" if "Countries" in df.columns else next(c for c in df.columns if c not in ycols)
    long = df.melt(id_vars=[ccol], value_vars=ycols, var_name="year", value_name="dep_kgha")
    long["year"] = long["year"].astype(int)
    long["ISO3"] = coco.convert(long[ccol].astype(str).tolist(), to="ISO3", not_found=None)
    long = long[long["ISO3"].astype(str).str.fullmatch(r"[A-Z]{3}")].dropna(subset=["dep_kgha"])
    out = long.groupby(["ISO3", "year"], as_index=False)["dep_kgha"].mean()
    out["source"] = "Vishwakarma et al. 2023 WL (Wang x LUH2) cropland total-N deposition; Dryad doi:10.5061/dryad.msbcc2g1x"
    dest = ROOT / "coefficients/dep_wl_country_byyear.csv"
    out.to_csv(dest, index=False)
    g = out.groupby("year")["dep_kgha"].mean()
    print(f"wrote {dest}: {len(out)} rows, {out.ISO3.nunique()} countries, {out.year.min()}-{out.year.max()}")
    print(f"  global unweighted country-mean WL dep: {g.loc[g.index.min()]:.2f} ({g.index.min()}) -> {g.loc[g.index.max()]:.2f} ({g.index.max()}) kg N/ha")


if __name__ == "__main__":
    main()
