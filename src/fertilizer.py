"""FERT — synthetic fertilizer nutrient applied to cropland.

  FERT_cropland(country) = SF_agri_use(country) * CF(country)

SF = FAOSTAT RFN "Agricultural Use" (item N=3102, P2O5=3103). P2O5 -> P x0.4364.
CF = cropland fraction (1.0 default; <1 for 21 CNB Table-1 countries). The (1-CF)
remainder is grassland fertilizer and feeds PAST (deferred).
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
from . import faostat_prep as prep
from . import faostat_api as fao
from . import dissolution
from .config import coef

_RFN_CACHE: dict = {}   # (item, element) -> (pivot ISO3 x year of tonnes, global-per-year series)


def _rfn_series(cfg, raw_dir, item, element):
    """Full RFN agri-use series for (item, element): pivot ISO3 x year (tonnes) + global-per-year.
    Cached so the historical run reads RFN once, not once per year."""
    key = (item, element)
    if key in _RFN_CACHE:
        return _RFN_CACHE[key]
    d = fao.filter_rows(fao.read_domain("RFN", raw_dir), element=element, item=item)
    d = prep.to_iso3(d, cfg)
    yc = next(c for c in d.columns if c.lower() == "year")
    d = d.groupby(["ISO3", yc])["Value"].sum().reset_index()
    piv = d.pivot(index="ISO3", columns=yc, values="Value")
    _RFN_CACHE[key] = (piv, piv.sum(axis=0, min_count=1))
    return _RFN_CACHE[key]


def _agri_use_year(cfg, raw_dir, year, item, element) -> pd.DataFrame:
    """National fertilizer agri-use (tonnes) for `year`, with a TEMPORAL GAP-FILL: a country present in
    RFN in other years but MISSING this year (e.g. Taiwan, absent from RFN before 1990; some small
    territories) is back-/forward-cast from its nearest-available year, scaled by the global fertilizer
    trajectory global(year)/global(nearest). Countries reporting an explicit 0 keep it (not a gap);
    countries never in RFN stay absent. Returns [ISO3, tonnes]."""
    piv, glob = _rfn_series(cfg, raw_dir, item, element)
    if year in piv.columns:                       # fast path: countries reporting THIS year
        base = piv[year].dropna()
    else:
        base = pd.Series(dtype=float)
    gY = float(glob[year]) if year in glob.index else None
    out = base.to_dict()
    from .dissolution import SUCCESSORS, PSEUDOS
    for iso in piv.index:
        if iso in out:
            continue                              # reported this year (incl. explicit 0)
        # never extrapolate a dissolution predecessor (USSR, ...) outside its real reporting years —
        # forward-casting it past the split would phantom-duplicate the successors that replace it
        if iso in PSEUDOS:
            continue
        s = piv.loc[iso].dropna()
        if s.empty:
            continue                              # never in RFN -> leave absent
        # don't back-cast a dissolution successor (RUS, UKR, ...) before it first reports — those
        # pre-split years are supplied by dissolution.redistribute() from the predecessor total
        # (USSR, ...). Filling here too would double-count.
        if iso in SUCCESSORS and year < int(min(s.index)):
            continue
        ny = min(s.index, key=lambda y: abs(int(y) - year))   # nearest available year
        gN = float(glob[ny]) if ny in glob.index else None
        out[iso] = float(s[ny]) * (gY / gN) if (gY is not None and gN) else float(s[ny])
    return pd.DataFrame({"ISO3": list(out), "tonnes": list(out.values())})


def _cf_table(cfg, nutrient: str) -> pd.DataFrame:
    path = coef(cfg, cfg["fertilizer"]["cropland_fraction_file"])
    col = {"N": "cf_N", "P": "cf_P", "K": "cf_K"}[nutrient]
    if not path.exists():
        return pd.DataFrame(columns=["ISO3", "cf"])
    cf = pd.read_csv(path)
    return cf[["ISO3", col]].rename(columns={col: "cf"})


def compute(cfg, raw_dir: Path, year: int, nutrient: str = "N",
            apply_cf: bool = True) -> pd.DataFrame:
    """National synthetic-fertilizer N (kg) by country.

    apply_cf=True  -> FERT_cropland = AgriUse * CF (cropland scope, FAO-comparable; default).
    apply_cf=False -> the FULL national AgriUse total (CF=1), i.e. fertilizer on ALL land
                      (cropland + grassland), used by the land/NANI view so the whole national
                      total is placed on the land (matching the SCALES-US 100% allocation).
    """
    codes = cfg["faostat"]["codes"]
    item = codes["fert_item_N"] if nutrient == "N" else codes["fert_item_P2O5"]
    # temporal gap-fill for entities missing from RFN in this year (e.g. Taiwan pre-1990)
    df = _agri_use_year(cfg, raw_dir, year, item, codes["fert_element_agri_use"])
    df["kg"] = df["tonnes"].astype(float) * cfg["conversions"]["tonnes_to_kg"]
    if nutrient == "P":
        df["kg"] *= cfg["conversions"]["p2o5_to_p"]

    if not apply_cf:
        df["fert_kg"] = df["kg"]
        out = df.groupby("ISO3", as_index=False)["fert_kg"].sum()
        return dissolution.redistribute_table(out, cfg, raw_dir, "fert")
    cf = _cf_table(cfg, nutrient)
    df = df.merge(cf, on="ISO3", how="left")
    df["cf"] = df["cf"].fillna(cfg["fertilizer"]["default_cf"])
    df["fert_kg"] = df["kg"] * df["cf"]
    out = df.groupby("ISO3", as_index=False)["fert_kg"].sum()
    return dissolution.redistribute_table(out, cfg, raw_dir, "fert")


def grassland(cfg, raw_dir: Path, year: int, nutrient: str = "N") -> pd.DataFrame:
    """The deferred (1-CF) grassland fertilizer = full national AgriUse minus the cropland part.
    [ISO3, fert_kg]. Feeds the land/NANI view (allocated onto grazing land)."""
    full = compute(cfg, raw_dir, year, nutrient, apply_cf=False).rename(columns={"fert_kg": "full"})
    crop = compute(cfg, raw_dir, year, nutrient, apply_cf=True).rename(columns={"fert_kg": "crop"})
    m = full.merge(crop, on="ISO3", how="left").fillna({"crop": 0.0})
    m["fert_kg"] = (m["full"] - m["crop"]).clip(lower=0.0)
    return m[["ISO3", "fert_kg"]]
