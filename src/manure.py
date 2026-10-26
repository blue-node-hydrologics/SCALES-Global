"""MAN — livestock manure N applied to cropland soils (cropland scope).

method='emn' (recommended): take FAOSTAT EMN "Manure applied to soils" N directly.
  Robust to code drift by matching the Element label substring.

method='bottom_up': animals(QCL) x IPCC-2019 Table 10.19 Nex(region,species)
  x applied_fraction. Needs coefficients/livestock_excretion.csv (+ tam, region map),
  which require source transcription (see DOWNLOAD_CHECKLIST). The applied fraction
  splits excreted N into applied-to-soils (-> MAN, cropland) vs left-on-pasture
  (-> PAST, deferred); global default ~0.22 applied / ~0.70 pasture.

Units: EMN N content is reported in kg N (verify unit column on download; set
cfg.manure.emn_unit_to_kg if a domain release uses tonnes).
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
from . import faostat_api as fao
from . import faostat_prep as prep
from . import utils
from .config import coef

EMN_APPLIED_LABEL = "applied to soils"   # element-label substring to match


def _emn(cfg, raw_dir: Path, year: int) -> pd.DataFrame:
    """Gross manure N applied to soils, from EMN element 72381, item 'All Animals' (1755).
    Using the All-Animals total avoids double counting the per-species + aggregate item rows;
    using the exact element code avoids catching the '...volatilises/leaches' sub-elements."""
    return _emn_element(cfg, raw_dir, year, cfg["faostat"]["codes"]["emn_applied_element"])


def _emn_element(cfg, raw_dir: Path, year: int, element_code) -> pd.DataFrame:
    """Generic EMN N-content reader for one element + 'All Animals' (1755), -> [ISO3, manure_kg].
    Using the All-Animals total avoids per-species double counting; the exact element code
    avoids catching the '...volatilises/leaches' sub-elements."""
    codes = cfg["faostat"]["codes"]
    df = fao.read_domain("EMN", raw_dir)
    df = fao.filter_rows(df, year=year, element=element_code,
                         item=codes["emn_all_animals_item"])
    df = prep.to_iso3(df, cfg)
    unit_to_kg = cfg["manure"].get("emn_unit_to_kg", 1.0)   # EMN N content already in kg
    df["manure_kg"] = df["Value"].astype(float) * unit_to_kg
    out = df.groupby("ISO3", as_index=False)["manure_kg"].sum()
    from . import dissolution
    return dissolution.redistribute_table(out, cfg, raw_dir, "manure")


def applied(cfg, raw_dir: Path, year: int) -> pd.DataFrame:
    """Manure N applied to soils (element 72381) -> cropland input. [ISO3, manure_kg]."""
    return _emn_element(cfg, raw_dir, year, cfg["faostat"]["codes"]["emn_applied_element"])


def pasture(cfg, raw_dir: Path, year: int) -> pd.DataFrame:
    """Manure N left on pasture by grazing animals (element 72380) -> grazing-land input.
    [ISO3, manure_kg]. Distinct pool from 'applied to soils'; goes to pasture/rangeland."""
    return _emn_element(cfg, raw_dir, year, cfg["faostat"]["codes"]["emn_pasture_element"])


def _emn_by_species(cfg, raw_dir: Path, year: int, element_code, item_codes) -> pd.DataFrame:
    """EMN N-content for one element and a set of species item codes, summed per country.
    item_codes may be a single code or a comma-separated list / iterable (EMN can split a
    GLW species into several EMN items, e.g. dairy vs non-dairy cattle). -> [ISO3, manure_kg]."""
    if isinstance(item_codes, str):
        item_codes = [c.strip() for c in item_codes.split(",") if c.strip()]
    elif not isinstance(item_codes, (list, tuple, set)):
        item_codes = [item_codes]
    # normalize to ints where possible (FAOSTAT item codes are numeric)
    norm = []
    for c in item_codes:
        try:
            norm.append(int(str(c).strip()))
        except (ValueError, TypeError):
            norm.append(c)
    df = fao.read_domain("EMN", raw_dir)
    df = fao.filter_rows(df, year=year, element=element_code)
    df = df[df["Item Code"].astype(str).str.strip().isin([str(c) for c in norm])]
    if df.empty:
        return pd.DataFrame(columns=["ISO3", "manure_kg"])
    df = prep.to_iso3(df, cfg)
    unit_to_kg = cfg["manure"].get("emn_unit_to_kg", 1.0)
    df["manure_kg"] = df["Value"].astype(float) * unit_to_kg
    return df.groupby("ISO3", as_index=False)["manure_kg"].sum()


def applied_fraction_by_species(cfg, raw_dir: Path, year: int):
    """Data-driven per-species applied fraction a_s(i,t) from FAOSTAT EMN.

    For each GLW species s (species_map.csv: glw_code -> emn_item_codes), compute
        a_s(i,t) = EMN_applied(s,i,t) / [ EMN_applied(s,i,t) + EMN_pasture(s,i,t) ]
    i.e. the fraction of that species' excreted-and-recoverable manure that FAOSTAT
    reports as applied to soils (element 72381) vs left on pasture (72380), per country
    and year. This is the country-varying analogue of the old global cattle_confined_fraction,
    extended to every species and read directly from the same activity data as the national
    manure magnitude, so the applied/pasture spatial split is internally consistent.

    Returns {glw_code: {ISO3: a_s}}. Species/countries with no EMN rows are omitted (the
    caller falls back to the configured per-code default). a_s is clamped to [0, 1]."""
    import csv
    map_file = cfg["manure"].get("species_map_file", "species_map.csv")
    map_path = coef(cfg, map_file)
    applied_el = cfg["faostat"]["codes"]["emn_applied_element"]
    pasture_el = cfg["faostat"]["codes"]["emn_pasture_element"]
    out = {}
    with open(map_path, newline="") as fh:
        for row in csv.DictReader(fh):
            glw = (row.get("glw_code") or "").strip()
            items = (row.get("emn_item_codes") or "").strip()
            if not glw or not items:
                continue
            app = _emn_by_species(cfg, raw_dir, year, applied_el, items)
            pas = _emn_by_species(cfg, raw_dir, year, pasture_el, items)
            m = app.merge(pas, on="ISO3", how="outer", suffixes=("_app", "_pas")).fillna(0.0)
            frac = {}
            for _, r in m.iterrows():
                denom = float(r["manure_kg_app"]) + float(r["manure_kg_pas"])
                if denom > 0:
                    a = float(r["manure_kg_app"]) / denom
                    frac[r["ISO3"]] = min(1.0, max(0.0, a))
            if frac:
                out[glw] = frac
    return out


def _bottom_up(cfg, raw_dir: Path, year: int, nutrient: str) -> pd.DataFrame:
    codes = cfg["faostat"]["codes"]
    animals = prep.tidy("QCL", cfg, raw_dir, year=year,
                        element=codes["animal_element_stocks"], item=None, value_name="head")
    exc = utils.load_coef(coef(cfg, cfg["manure"]["excretion_file"]),
                          "IPCC Table 10.19 excretion",
                          required_cols=["fao_item_code", "ipcc_region", "nex_kg_head_yr"])
    region = utils.load_coef(coef(cfg, cfg["manure"]["region_map_file"]),
                             "ISO3 -> IPCC region map", required_cols=["ISO3", "ipcc_region"])
    df = (animals.merge(region, on="ISO3", how="left")
                 .merge(exc, left_on=["Item Code", "ipcc_region"],
                        right_on=["fao_item_code", "ipcc_region"], how="inner"))
    af = cfg["manure"]["applied_fraction_default"]
    df["manure_kg"] = df["head"].astype(float) * df["nex_kg_head_yr"].astype(float) * af
    return df.groupby("ISO3", as_index=False)["manure_kg"].sum()


def compute(cfg, raw_dir: Path, year: int, nutrient: str = "N") -> pd.DataFrame:
    if nutrient == "N" and cfg["manure"]["method"] == "emn":
        return _emn(cfg, raw_dir, year)
    return _bottom_up(cfg, raw_dir, year, nutrient)
