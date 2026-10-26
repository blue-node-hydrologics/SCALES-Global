"""Prep FAOSTAT long tables into tidy per-ISO3 frames for the budget.

Handles the FAO area-code -> ISO3 crosswalk and drops FAO regional/aggregate
'areas' (Area Code >= 5000 are aggregates) so only real countries remain.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
from . import faostat_api as fao
from . import utils

# FAOSTAT legacy area code -> placeholder ISO3 for historical dissolution aggregates. Tagging (rather
# than dropping) these lets every component compute a term total for the predecessor, which
# dissolution.redistribute() then splits onto the modern successor states. See src/dissolution.py.
DISSOLUTION_PSEUDO = {228: "SUN", 248: "YUG", 51: "CSK", 15: "BLX",
                      186: "SCG", 62: "ETX", 206: "SDX"}


def _crosswalk(cfg) -> pd.DataFrame:
    """FAO area code -> ISO3. Prefer the shipped crosswalk; else use the 'Area Code (M49)'
    column present in modern normalized bulks via country_converter."""
    from .config import coef
    path = coef(cfg, cfg["geo"]["country_crosswalk_file"])
    if path.exists():
        cw = pd.read_csv(path)
        return cw[["fao_area_code", "ISO3"]].dropna()
    return pd.DataFrame(columns=["fao_area_code", "ISO3"])


def to_iso3(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """Attach an ISO3 column and drop aggregates. Works whether the bulk carries
    'Area Code (M49)' or the legacy 'Area Code'."""
    df = df.copy()
    cw = _crosswalk(cfg)
    if not cw.empty and "Area Code" in df.columns:
        df = df.merge(cw, left_on="Area Code", right_on="fao_area_code", how="left")
    elif "Area Code (M49)" in df.columns:
        import logging, country_converter as cc
        logging.getLogger("country_converter").setLevel(logging.CRITICAL)  # silence not-found spam
        # M49 codes arrive as "'004"; country_converter wants integer ISO-numeric codes.
        num = pd.to_numeric(df["Area Code (M49)"].astype(str).str.lstrip("'"), errors="coerce")
        uniq = sorted({int(v) for v in num.dropna().unique()})          # convert unique only
        conv = cc.CountryConverter().convert([str(u) for u in uniq],
                                             src="ISOnumeric", to="ISO3", not_found=None)
        m = {u: (c if c not in (None, "not found") else pd.NA) for u, c in zip(uniq, conv)}
        # vectorized lookup : identical result to the former per-row lambda, which took
        # ~19 s per model year on the 3.4M-row domains. Unmatched / NaN codes -> missing.
        df["ISO3"] = num.map({float(k): v for k, v in m.items()})
    else:
        df["ISO3"] = pd.NA
    # drop FAO aggregates (legacy area codes >= 5000) and unmapped rows
    if "Area Code" in df.columns:
        ac = pd.to_numeric(df["Area Code"], errors="coerce")
        df = df[ac < 5000]
        # force-tag historical dissolution aggregates (USSR=228, Yugoslavia=248, ...) with a
        # placeholder ISO3 so they survive as pseudo-countries for dissolution.redistribute() to
        # back-cast onto successors. Their legacy area codes are unambiguous, so overwrite whatever
        # the M49/country_converter step produced (which leaves e.g. USSR as the numeric string "810").
        ac = pd.to_numeric(df["Area Code"], errors="coerce")
        need = ac.isin(DISSOLUTION_PSEUDO)
        if need.any():
            df.loc[need, "ISO3"] = ac[need].map(DISSOLUTION_PSEUDO)
    # keep only valid 3-letter ISO3; this also drops FAO super-aggregates that
    # country_converter leaves as a numeric string (e.g. the "China" aggregate -> "159",
    # which would otherwise double-count alongside "China, mainland" = CHN).
    df = df[df["ISO3"].notna()]
    df = df[df["ISO3"].astype(str).str.fullmatch(r"[A-Z]{3}")]
    return df


def tidy(domain: str, cfg, raw_dir: Path, *, year, element, item, value_name: str) -> pd.DataFrame:
    """Download/read a domain, filter, attach ISO3, return [ISO3, value_name(, Item Code)]."""
    df = fao.read_domain(domain, raw_dir)
    df = fao.filter_rows(df, year=year, element=element, item=item)
    df = to_iso3(df, cfg)
    keep = ["ISO3", "Item Code", "Value"]
    out = df[[c for c in keep if c in df.columns]].rename(columns={"Value": value_name})
    return out.reset_index(drop=True)
