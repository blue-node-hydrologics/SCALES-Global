"""DEP — atmospheric N deposition onto cropland.

RELEASE BASIS = TOTAL N (oxidized NOy + reduced NHx), per the soil-surface convention
(net manure paired with total deposition). The
per-year country table is dep_total_country_byyear.csv (NOy+NHx), built from ISIMIP3a histsoc
(DOI 10.48364/ISIMIP.759077) by scripts/31b. config.deposition.basis='total' selects it;
basis='oxidized' selects the NOy-only table (dep_ox_country_byyear.csv) for NANI/gTREND-style
diagnostics. (Historical note: an earlier configuration used NOy-only with NHx excluded; that
is no longer the release basis.)

NB: the returned per-ha column is still named 'dep_ox_kgha' for backward compatibility, but
under basis='total' it carries TOTAL N (NOy+NHx), not oxidized-only — to be renamed dep_kgha
in the deposited dataset.

Output here is kg N for the country = dep_ox_kgha * cropland_ha (cropland scope);
the country table stores the per-ha rate and country_areas supplies the hectares.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
from . import utils
from .config import coef


def dep_kgha_for_year(cfg, year: int):
    """Per-country N deposition (kg N/ha) for `year`, returned as DataFrame[ISO3, dep_ox_kgha]
    (column name kept for downstream compatibility; under basis 'total'/'wl_hybrid' it carries
    TOTAL N, not oxidized-only). Bases:
      'wl_hybrid' (release): national MAGNITUDE from Vishwakarma 2023 WL (dep_wl_country_byyear.csv);
                  the gridded within-country pattern is set separately by the ISIMIP weight.
      'total'   : ISIMIP3a NOy+NHx national means (dep_total_country_byyear.csv).
      'oxidized': ISIMIP3a NOy-only (dep_ox_country_byyear.csv).
    Clamps to table coverage (holds the last year for years beyond it); falls back to the constant
    dep_ox_country.csv. Returns None if no source is available."""
    dep = cfg["deposition"]
    basis = dep.get("basis", "oxidized")
    fname = {"wl_hybrid": dep.get("wl_byyear_file", "dep_wl_country_byyear.csv"),
             "total": dep.get("byyear_file", "dep_total_country_byyear.csv"),
             }.get(basis, dep.get("oxidized_byyear_file", "dep_ox_country_byyear.csv"))
    byyear = coef(cfg, fname)
    if byyear.exists():
        t = pd.read_csv(byyear)
        col = "dep_kgha" if (basis in ("total", "wl_hybrid") and "dep_kgha" in t.columns) else "dep_ox_kgha"
        yrs = t["year"].astype(int)
        y = int(min(max(year, yrs.min()), yrs.max()))      # clamp to coverage
        sub = t[t["year"].astype(int) == y][["ISO3", col]].rename(columns={col: "dep_ox_kgha"})
        if len(sub):
            return sub
    try:
        return utils.load_coef(coef(cfg, cfg["deposition"]["oxidized_file"]),
                               "oxidized-N deposition by country",
                               required_cols=["ISO3", "dep_ox_kgha"])[["ISO3", "dep_ox_kgha"]]
    except utils.MissingSourceFile:
        return None


def dep_noncrop_kgha_for_year(cfg, year: int):
    """Per-country NON-cropland ISIMIP N deposition rate (kg N/ha, total N) for `year`, as
    DataFrame[ISO3, dep_noncrop_kgha], from coefficients/<deposition.noncrop_byyear_file>
    (built by scripts/31d). Used by the PIECEWISE landscape/total deposition: cropland gets the
    Vishwakarma WL rate, non-cropland gets this ISIMIP rate. Clamps to table coverage; returns
    None if the table is absent (pipeline then falls back to the WL rate over all land)."""
    fname = cfg["deposition"].get("noncrop_byyear_file", "dep_noncrop_country_byyear.csv")
    path = coef(cfg, fname)
    if not path.exists():
        return None
    t = pd.read_csv(path)
    if "dep_noncrop_kgha" not in t.columns or "year" not in t.columns:
        return None
    yrs = t["year"].astype(int)
    y = int(min(max(year, yrs.min()), yrs.max()))              # clamp to coverage
    sub = t[t["year"].astype(int) == y][["ISO3", "dep_noncrop_kgha"]]
    return sub if len(sub) else None


def compute(cfg, raw_dir: Path, year: int, areas: pd.DataFrame,
            nutrient: str = "N") -> pd.DataFrame:
    if nutrient != "N":
        return pd.DataFrame(columns=["ISO3", "dep_ox_kg"])
    try:
        dep = dep_kgha_for_year(cfg, year)
        if dep is None:
            raise utils.MissingSourceFile("deposition")
    except utils.MissingSourceFile:
        if not cfg["deposition"].get("required", True):
            print("  [warn] DEP_ox not available -> treated as 0 (set deposition.required:true "
                  "after running scripts/04+05_*deposition to include it)")
            return pd.DataFrame(columns=["ISO3", "dep_ox_kg"])
        raise
    df = areas[["ISO3", "AREA_HA"]].merge(dep[["ISO3", "dep_ox_kgha"]], on="ISO3", how="left")
    df["dep_ox_kg"] = df["dep_ox_kgha"].astype(float) * df["AREA_HA"].astype(float)
    return df[["ISO3", "dep_ox_kg"]].dropna()
