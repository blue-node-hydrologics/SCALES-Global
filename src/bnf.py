"""BNF — biological N fixation (N only).

Grain legumes (production-driven, Peoples 2021 + Herridge 2022 structure):
    fixed_N = production_t * grainN_kg_per_t * wholeplant_factor * %Ndfa
  - grainN_kg_per_t = the Ludemann N-removal coefficient (coefficients/crop_uptake.csv),
    i.e. kg N in the harvested grain per tonne fresh product (validated set, r~1.0 vs FAO).
  - wholeplant_factor scales grain N -> total plant N (shoot residue + roots +
    rhizodeposition), i.e. inverse N-harvest-index x belowground factor. Group values
    are calibrated so grain-N x %Ndfa x factor reproduces the global grain-legume BNF
    totals published by Herridge et al. 2022 for 2018: soybean 25.0 Tg (factor 1.9),
    groundnut 3.3 Tg (factor 3.13), pulses 7.2 Tg (factor 4.42). Pulses carry a higher
    factor than soybean because their lower N-harvest-index leaves much more N in shoot
    residue + belowground. %Ndfa is region-specific for soybean, group-based for pulses
    (Peoples et al. 2021). Parameters in coefficients/bnf_params.csv. NOTE: calibration
    is to Herridge's own published totals (the literature), not to the FAO CNB benchmark.

Free-living fixation (area x rate): 25 kg N/ha on rice + sugarcane harvested area (FAO CNB).
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
from . import faostat_prep as prep
from . import utils
from .config import coef

SOYBEAN_BRAZIL = {"BRA"}
SOYBEAN_EUROPE = {
    "ALB","AUT","BEL","BGR","BIH","BLR","CHE","CZE","DEU","DNK","ESP","EST","FIN",
    "FRA","GBR","GRC","HRV","HUN","IRL","ITA","LTU","LUX","LVA","MDA","MKD","NLD",
    "NOR","POL","PRT","ROU","RUS","SRB","SVK","SVN","SWE","UKR",
}


def _params(cfg) -> pd.DataFrame:
    return utils.load_coef(coef(cfg, cfg["bnf"]["params_file"]), "BNF parameters",
                           required_cols=["fao_item_code", "ndfa_group", "wholeplant_factor"])


def _grain_n(cfg) -> dict:
    """item_code -> grain N removal coefficient (kg N / t fresh) from crop_uptake.csv."""
    cu = utils.load_coef(coef(cfg, "crop_uptake.csv"), "crop removal coefficients",
                         required_cols=["fao_item_code", "removal_kg_per_t_N"])
    return dict(zip(cu["fao_item_code"].astype("Int64"), cu["removal_kg_per_t_N"].astype(float)))


def _ndfa(row, iso3: str) -> float:
    if str(row["ndfa_group"]) == "soybean":
        if iso3 in SOYBEAN_BRAZIL:
            return float(row.get("ndfa_brazil") or 0.78)
        if iso3 in SOYBEAN_EUROPE:
            return float(row.get("ndfa_europe") or 0.44)
        return float(row.get("ndfa_row") or 0.61)
    return float(row["ndfa_default"])


def grassland(cfg, raw_dir: Path, year: int, areas: pd.DataFrame, nutrient: str = "N") -> pd.DataFrame:
    """Pasture/forage-legume biological N fixation over permanent pasture (landscape/total only).
    BNF_grass = PASTURE_HA x grassland_rate_kg_ha. Coarse global-average rate (legume-pasture
    fraction not resolved); calibrate to Herridge et al. (2008). [ISO3, bnf_grass_kg]."""
    if nutrient != "N" or areas is None or "PASTURE_HA" not in areas.columns:
        return pd.DataFrame(columns=["ISO3", "bnf_grass_kg"])
    rate = float(cfg["bnf"].get("grassland_rate_kg_ha", 0.0))
    out = areas[["ISO3", "PASTURE_HA"]].copy()
    out["bnf_grass_kg"] = out["PASTURE_HA"].astype(float) * rate
    return out[["ISO3", "bnf_grass_kg"]]


def components(cfg, raw_dir: Path, year: int, nutrient: str = "N"):
    """(legume, free_living) national BNF tables, each [ISO3, bnf_kg] and each back-cast onto
    dissolution successors. Split out so the coefficient-level Monte Carlo can apply
    the legume coefficients (cN, f_bnf, wpf) and r_free separately; compute() returns their sum."""
    if nutrient != "N":
        e = pd.DataFrame(columns=["ISO3", "bnf_kg"])
        return e, e.copy()
    codes = cfg["faostat"]["codes"]
    prod = prep.tidy("QCL", cfg, raw_dir, year=year,
                     element=codes["crop_element_production"], item=None, value_name="tonnes")
    area = prep.tidy("QCL", cfg, raw_dir, year=year,
                     element=codes["crop_element_area"], item=None, value_name="ha")
    p = _params(cfg)
    grainN = _grain_n(cfg)

    # 1) grain legumes
    leg = p[p["ndfa_group"] != "freeliving"]
    lj = prod.merge(leg, left_on="Item Code", right_on="fao_item_code", how="inner")
    rows = []
    for _, r in lj.iterrows():
        gn = grainN.get(pd.array([r["Item Code"]], dtype="Int64")[0])
        if gn is None or pd.isna(gn):
            continue
        kg = (float(r["tonnes"]) * float(gn) * float(r["wholeplant_factor"])
              * _ndfa(r, r["ISO3"]))
        rows.append((r["ISO3"], kg))
    legN = pd.DataFrame(rows, columns=["ISO3", "bnf_kg"])

    # 2) free-living: rice + sugarcane area x rate
    fl_items = p[p["ndfa_group"] == "freeliving"]["fao_item_code"].tolist()
    rate = cfg["bnf"]["freeliving_rate_kg_ha"]
    fl = area[area["Item Code"].isin(fl_items)].copy()
    fl["bnf_kg"] = fl["ha"].astype(float) * rate
    flN = fl[["ISO3", "bnf_kg"]]

    from . import dissolution
    legN = legN.groupby("ISO3", as_index=False)["bnf_kg"].sum()
    flN = flN.groupby("ISO3", as_index=False)["bnf_kg"].sum()
    return (dissolution.redistribute_table(legN, cfg, raw_dir, "bnf"),
            dissolution.redistribute_table(flN, cfg, raw_dir, "bnf"))


def compute(cfg, raw_dir: Path, year: int, nutrient: str = "N") -> pd.DataFrame:
    if nutrient != "N":
        return pd.DataFrame(columns=["ISO3", "bnf_kg"])
    legN, flN = components(cfg, raw_dir, year, nutrient)
    # redistribution is linear, so splitting the two parts separately and summing is identical to
    # splitting their sum (the previous order of operations)
    return pd.concat([legN, flN], ignore_index=True).groupby("ISO3", as_index=False)["bnf_kg"].sum()
