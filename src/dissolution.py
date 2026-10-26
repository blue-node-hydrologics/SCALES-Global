"""Back-cast FAOSTAT dissolution aggregates onto their modern successor states.

FAOSTAT reports several historical countries as single "areas" for the years they existed and
their successor states only afterwards (USSR 1961-1991 -> 15 states from 1992; Yugoslav SFR;
Czechoslovakia; Belgium-Luxembourg; Serbia and Montenegro; Ethiopia PDR; former Sudan). Our ISO3
crosswalk cannot map these aggregates, so — left alone — every FAOSTAT statistical term (fertilizer,
crop removal, manure, BNF, population) silently drops the entire predecessor before its split year.
That understates pre-split global totals (~10 Tg N of fertilizer in 1990 alone), creates a spurious
jump at the dissolution year, and leaves the successor countries blank in every pre-split grid.

The FAO Cropland Nutrient Budget keeps each aggregate as its own reporting unit (USSR row 1961-1991,
Russia + Ukraine from 1992); it does not back-cast. SCALES-Global is a *gridded, modern-boundary*
product spanning 1961-2023, so instead we **back-cast**: each predecessor's pre-split national total
for a term is distributed among its successor states in proportion to the successors' mean share of
that same term over their first few post-split years. This yields continuous per-country series,
populates the successor grid cells before the split, and removes the discontinuity. For the FAO-CNB
consistency check the successors are re-aggregated back to the predecessor unit (see scripts/50).

Mechanics: `to_iso3` (in faostat_prep) tags each predecessor row with a placeholder ISO3
(prep.DISSOLUTION_PSEUDO: 228->'SUN', 248->'YUG', ...), so every component computes a term total for
the pseudo-country exactly as for a real one. `redistribute()` then splits those pseudo-country totals
onto the successors and drops the pseudo rows, once, just before the national budget is assembled.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd

from . import faostat_api as fao
from . import faostat_prep as prep
from .config import coef, get_path

# pseudo ISO3 -> (successor modern ISO3s, share window [y0, y1] where successors report separately)
_SPLIT: dict[str, tuple[list[str], tuple[int, int]]] = {
    "SUN": (["RUS", "UKR", "BLR", "MDA", "EST", "LVA", "LTU", "GEO", "ARM", "AZE",
             "KAZ", "UZB", "TKM", "KGZ", "TJK"], (1992, 1996)),   # USSR
    "YUG": (["SVN", "HRV", "BIH", "MKD", "SRB", "MNE"], (2006, 2010)),  # Yugoslav SFR
    "SCG": (["SRB", "MNE"], (2006, 2010)),                              # Serbia and Montenegro
    "CSK": (["CZE", "SVK"], (1993, 1997)),                             # Czechoslovakia
    "BLX": (["BEL", "LUX"], (2000, 2004)),                             # Belgium-Luxembourg
    "ETX": (["ETH", "ERI"], (1993, 1997)),                            # Ethiopia PDR
    "SDX": (["SDN", "SSD"], (2012, 2016)),                            # former Sudan
}

# term -> FAOSTAT driver used to derive successor shares (domain, element, item; item None = all).
# "bnf" reuses the crop driver (BNF geography tracks legume cropping and no lighter national proxy is
# available pre-split); "population" is read from the population coefficient file instead of a domain.
_DRIVER = {
    "fert":    ("RFN", "fert_element_agri_use", "fert_item_N"),
    "crop":    ("QCL", "crop_element_production", None),
    "manure":  ("EMN", "emn_applied_element", "emn_all_animals_item"),
    "bnf":     ("QCL", "crop_element_production", None),
    "population": ("_POP", None, None),
    # area denominators (RL land-use domain) — split so pre-split successors get a cropland/land
    # denominator for their back-cast flows, otherwise per-hectare surplus would divide by zero
    "cropland": ("RL", "land_element_area", "land_item_cropland"),
    "pasture":  ("RL", "land_element_area", "land_item_pasture"),
    "land":     ("RL", "land_element_area", "land_item_land"),
}

# area column -> the land-type share used to split it in redistribute_areas()
_AREA_COL_TERM = {"AREA_HA": "cropland", "PASTURE_HA": "pasture", "LAND_HA": "land"}

# component name -> term whose successor shares are used to split it. Comps absent here (dep, dep_land,
# forage) are grid/area-derived on modern boundaries and carry no pseudo-country rows, so are skipped.
COMP_TERM = {
    "fert": "fert", "fert_grass": "fert", "fert_vol": "fert", "fert_vol_full": "fert",
    "crop": "crop",
    "man_crop": "manure", "man_past": "manure", "man_vol": "manure", "man_vol_full": "manure",
    "bnf": "bnf", "bnf_grass": "bnf", "forage": "pasture",
    "hum_sewage": "population", "hum_sew_water": "population", "hum_sew_land": "population",
    "hum_crop": "population", "hum_ns": "population", "hum_bio": "population",
    "hum_landfill": "population", "food_waste": "population",
}

PSEUDOS: set[str] = set(_SPLIT)   # predecessor placeholder ISO3s (SUN, YUG, CSK, ...)

# all modern successor ISO3s. A successor must NOT be independently back-cast (e.g. by the fertilizer
# temporal gap-fill) into years before it first reports, because those years belong to its predecessor
# aggregate and are supplied by redistribute() — doing both would double-count.
SUCCESSORS: set[str] = {c for (succ, _win) in _SPLIT.values() for c in succ}

_SHARE_CACHE: dict[str, dict[str, dict[str, float]]] = {}


def _driver_series(cfg, raw_dir, term) -> pd.DataFrame:
    """Full ISO3 x year driver table for `term` (columns ISO3, year, val)."""
    dom, el_key, it_key = _DRIVER[term]
    if dom == "_POP":
        path = coef(cfg, cfg["human"]["population_file"])
        p = pd.read_csv(path)
        return p.rename(columns={"population": "val"})[["ISO3", "year", "val"]]
    codes = cfg["faostat"]["codes"]
    el = codes[el_key] if el_key else None
    it = codes[it_key] if it_key else None
    d = fao.filter_rows(fao.read_domain(dom, raw_dir), element=el, item=it)
    d = prep.to_iso3(d, cfg)
    yc = next(c for c in d.columns if c.lower() == "year")
    return d.groupby(["ISO3", yc])["Value"].sum().reset_index().rename(columns={yc: "year", "Value": "val"})


def shares(cfg, raw_dir, term) -> dict[str, dict[str, float]]:
    """{pseudo -> {successor ISO3 -> share}} for `term`, cached. Shares are each successor's mean
    fraction of the successor-group total of the driver over the split window. Falls back to equal
    shares if the window carries no signal."""
    if term in _SHARE_CACHE:
        return _SHARE_CACHE[term]
    ser = _driver_series(cfg, raw_dir, term)
    out: dict[str, dict[str, float]] = {}
    for pseudo, (succ, (y0, y1)) in _SPLIT.items():
        w = ser[(ser["ISO3"].isin(succ)) & (ser["year"].between(y0, y1))]
        tot = w.groupby("ISO3")["val"].sum()
        s = {c: float(tot.get(c, 0.0)) for c in succ}
        denom = sum(s.values())
        if denom > 0:
            out[pseudo] = {c: v / denom for c, v in s.items()}
        else:                                   # no driver signal -> split evenly
            out[pseudo] = {c: 1.0 / len(succ) for c in succ}
    _SHARE_CACHE[term] = out
    return out


def redistribute_table(df: pd.DataFrame, cfg, raw_dir, term: str, keep_cols=()) -> pd.DataFrame:
    """Split every pseudo-country (SUN, YUG, ...) row in `df` onto its successor states by `term`'s
    shares, add to the successors, and drop the pseudo rows. `keep_cols` are extra grouping keys to
    preserve (e.g. ["fao_item_code"] for the per-item crop/BNF grid tables); every other non-ISO3
    column is treated as a value column and scaled by each successor's share. No-op when `df` carries
    no pseudo rows (post-split years, grid/area-derived tables). Idempotent."""
    if df is None or "ISO3" not in df.columns or not len(df):
        return df
    present = [p for p in PSEUDOS if p in set(df["ISO3"])]
    if not present:
        return df
    keep = [c for c in keep_cols if c in df.columns]
    vcols = [c for c in df.columns if c != "ISO3" and c not in keep]
    sh = shares(cfg, raw_dir, term)
    rows = df[~df["ISO3"].isin(PSEUDOS)].copy()               # keep real countries
    addrows = []
    for p in present:
        for _, prow in df[df["ISO3"] == p].iterrows():
            base = {k: prow[k] for k in keep}
            for succ, frac in sh[p].items():
                r = {"ISO3": succ, **base}
                for c in vcols:
                    v = pd.to_numeric(pd.Series([prow[c]]), errors="coerce").fillna(0.0).iloc[0]
                    r[c] = float(v) * frac
                addrows.append(r)
    if not addrows:
        return rows
    merged = pd.concat([rows, pd.DataFrame(addrows)], ignore_index=True)
    return merged.groupby(["ISO3"] + keep, as_index=False)[vcols].sum()


def redistribute(comps: dict, cfg, raw_dir, comp_term: dict | None = None) -> dict:
    """Catch-all: apply redistribute_table to every component in `comps` by its COMP_TERM mapping.
    Kept as a belt-and-suspenders pass; the leaf component functions already self-redistribute, so
    this is normally a no-op."""
    comp_term = comp_term or COMP_TERM
    for name, df in comps.items():
        term = comp_term.get(name)
        if term is not None:
            comps[name] = redistribute_table(df, cfg, raw_dir, term)
    return comps


def redistribute_areas(areas: pd.DataFrame, cfg, raw_dir) -> pd.DataFrame:
    """Split a country_areas table's pseudo-country rows (SUN, ...) onto successors, each area column
    by its own land-type shares (cropland by cropland shares, pasture by pasture, land by land), so
    pre-split successors carry the denominator their back-cast flows need. No-op when no pseudo rows."""
    if areas is None or "ISO3" not in areas.columns:
        return areas
    present = [p for p in PSEUDOS if p in set(areas["ISO3"])]
    if not present:
        return areas
    cols = [c for c in _AREA_COL_TERM if c in areas.columns]
    rows = areas[~areas["ISO3"].isin(PSEUDOS)].copy()
    add: dict[str, dict[str, float]] = {}
    for p in present:
        prow = areas[areas["ISO3"] == p]
        for col in cols:
            v = float(pd.to_numeric(prow[col], errors="coerce").fillna(0.0).sum())
            if v == 0:
                continue
            for succ, frac in shares(cfg, raw_dir, _AREA_COL_TERM[col])[p].items():
                add.setdefault(succ, {}).setdefault(col, 0.0)
                add[succ][col] += v * frac
    if not add:
        return rows
    addrows = [{"ISO3": s, **{c: d.get(c, 0.0) for c in cols}} for s, d in add.items()]
    merged = pd.concat([rows, pd.DataFrame(addrows)], ignore_index=True)
    return merged.groupby("ISO3", as_index=False)[cols].sum()
