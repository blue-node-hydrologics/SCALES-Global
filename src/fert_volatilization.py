"""Synthetic-fertilizer NH3 volatilization (FracGASF) by country.

For the soil-surface budget harmonized with SCALES-US (as-applied), synthetic fertilizer is
netted for the NH3 lost to the atmosphere after application. The per-country loss fraction
(FracGASF) is built bottom-up from the national fertilizer **product mix** (FAOSTAT
'Fertilizers by Product', RFB, element 5157 Agricultural Use — reported in tonnes of PHYSICAL
PRODUCT) weighted by each product's N content and its product-specific NH3-N loss fraction
(coefficients/fertilizer_volatilization_global.csv):

    FracGASF(country) = Σ_p (use_p · Ncontent_p · loss_p) / Σ_p (use_p · Ncontent_p)

PROVENANCE of the product loss fractions: IPCC 2019 V4 Ch.11 Table 11.3 provides only TWO synthetic
values — urea (and urea-based) = 0.15, and all OTHER synthetic N = 0.05. We keep urea = 0.15 (IPCC)
and refine the "other" category to product resolution using the EMEP/EEA 2023 Guidebook and Bouwman
et al. 2002 (e.g., DAP alkaline micro-site -> 0.10; MAP acidic micro-site -> 0.03; nitrates ~0.01;
anhydrous injected ~0.02). Only urea is a direct IPCC value; the rest are EMEP/Bouwman/expert.
CAVEATS (see paper): the product weighting omits soil pH (the dominant driver), climate/temperature
(IPCC does NOT climate-disaggregate FracGASF — that split is for EF1 N2O), application method
(incorporation/injection), and enhanced-efficiency fertilizers (invisible to FAOSTAT). FracGASF is
NH3+NOx, not NH3-only. This reproduces the SCALES-US type-weighted value (USA ~0.078) and is higher
in urea-dominant regions (e.g., India ~0.13). Where RFB product data are missing for a country/year,
the global N-weighted default (~0.11 = urea-share-weighted IPCC binary) is used.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
from . import faostat_api as fao, faostat_prep as prep
from .config import coef

DEFAULT_FRACGASF = 0.11   # global N-weighted fallback (~IPCC 2019 synthetic default)


def _coef_table(cfg) -> pd.DataFrame:
    return pd.read_csv(coef(cfg, "fertilizer_volatilization_global.csv"))


def fracgasf_by_country(cfg, raw_dir: Path, year: int) -> pd.DataFrame:
    """[ISO3, fracgasf] — N-weighted mean fertilizer NH3 loss fraction per country for `year`."""
    ct = _coef_table(cfg)[["item_code", "n_content", "fracgasf"]]
    df = fao.read_domain("RFB", raw_dir)
    df = fao.filter_rows(df, year=year, element=5157)             # Agricultural Use (t product)
    df = df.merge(ct, left_on="Item Code", right_on="item_code", how="inner")
    df = prep.to_iso3(df, cfg)
    df["N_kg"] = df["Value"].astype(float) * 1000.0 * df["n_content"]    # t product -> kg N
    df["vol_kg"] = df["N_kg"] * df["fracgasf"]
    g = df.groupby("ISO3", as_index=False).agg(N_kg=("N_kg", "sum"), vol_kg=("vol_kg", "sum"))
    g["fracgasf"] = (g["vol_kg"] / g["N_kg"]).fillna(DEFAULT_FRACGASF)
    return g[["ISO3", "fracgasf"]]


def volatilization_loss(cfg, raw_dir: Path, year: int, fert_nat: pd.DataFrame,
                        fert_col: str = "fert_kg") -> pd.DataFrame:
    """Given a national fertilizer table [ISO3, fert_col] (kg N), return [ISO3, fert_vol_kg] =
    fert_col x FracGASF(country). Countries with no RFB product mix use DEFAULT_FRACGASF."""
    fg = fracgasf_by_country(cfg, raw_dir, year)
    m = fert_nat.merge(fg, on="ISO3", how="left")
    m["fracgasf"] = m["fracgasf"].fillna(DEFAULT_FRACGASF)
    m["fert_vol_kg"] = m[fert_col].astype(float) * m["fracgasf"]
    return m[["ISO3", "fert_vol_kg"]]
