"""Assemble the country-scale N (+P scaffold) mass balance.

Three N surpluses are produced from one budget:
  CROPLAND surplus (per ha cropland; includes human excreta recycled TO cropland, hum_crop):
    NS_crop = FERT_crop + BNF + DEP_crop + MAN_app + HUM_crop - CROP - FERTvol_crop - MANvol_app
  CROPLAND surplus, FAO-comparable variant (hum_crop EXCLUDED, to match the FAO CNB input set):
    NS_crop_faocomp = FERT_crop + BNF + DEP_crop + MAN_app - CROP - FERTvol_crop - MANvol_app
  LANDSCAPE surplus (per ha whole-country land; land-applied inputs, sewage discharge EXCLUDED):
    NS_land = FERT_crop + FERT_grass + BNF + BNF_grass + DEP_land + MAN_app + MAN_past
              + FOOD_waste + HUM_recycle - CROP - FORAGE - FERTvol_all - MANvol_all
  FULL / TOTAL surplus (landscape + human sewage discharge):
    NS_full = NS_land x LAND_HA + HUM  (per ha land)
FOOD_waste = consumer-wasted food N (landfill/compost) and HUM_recycle = human excreta recycled to
cropland (night soil/biosolids) are LAND-APPLIED human-system inputs (landscape); HUM = the remaining
human excreta discharged via sewage (total only). FORAGE = grassland forage removal. Cropland
components are divided by cropland area (AREA_HA); landscape/full by total land area (LAND_HA).
Merged on ISO3; missing components treated as 0.
"""
from __future__ import annotations
import pandas as pd

# CROPLAND inputs (human waste EXCLUDED; it enters only the full surplus)
N_INPUTS = ["fert", "bnf", "dep", "man_crop"]
N_OUTPUTS = ["crop"]
N_LOSSES = ["fert_vol", "man_vol"]                       # cropland NH3 losses
# extra national components used by the landscape/full (whole-country) surpluses.
# hum_crop (= hum_ns + hum_bio) ALSO enters the cropland surplus (below); hum_ns/hum_bio are carried
# for reporting/downscaling only (NOT re-summed into any surplus). hum_sewage = sewage discharge (total).
N_FULL = ["fert_grass", "dep_land", "man_past", "hum_sewage", "hum_sew_water", "hum_sew_land",
          "fert_vol_full", "man_vol_full", "forage", "bnf_grass", "food_waste",
          "hum_crop", "hum_ns", "hum_bio", "hum_landfill"]
P_INPUTS = ["fert", "man_crop", "hum_sewage", "dep_p"]
P_OUTPUTS = ["crop"]


def assemble(components: dict, areas: pd.DataFrame, nutrient: str = "N") -> pd.DataFrame:
    inputs = N_INPUTS if nutrient == "N" else P_INPUTS
    outputs = N_OUTPUTS if nutrient == "N" else P_OUTPUTS
    # NH3 volatilization losses (cropland) — subtracted from the surplus, excluded from NUE.
    losses = [t for t in N_LOSSES if nutrient == "N" and components.get(t) is not None]
    # extra components for the full (whole-country) surplus
    full = [t for t in N_FULL if nutrient == "N" and components.get(t) is not None]
    names = inputs + outputs + losses + full

    keep = ["ISO3"] + [c for c in ("AREA_HA", "LAND_HA") if c in areas.columns]
    base = areas[keep].copy()
    for name in names:
        df = components.get(name)
        col = name + "_kg"
        if df is None or df.empty:
            base[col] = 0.0
            continue
        valcol = [c for c in df.columns if c != "ISO3"][0]
        base = base.merge(df.rename(columns={valcol: col}), on="ISO3", how="left")
    kg_cols = [n + "_kg" for n in names]
    base[kg_cols] = base[kg_cols].fillna(0.0)

    for n in names:
        base[n + "_kgha"] = base[n + "_kg"] / base["AREA_HA"]
    in_ha = [n + "_kgha" for n in inputs]
    out_ha = [n + "_kgha" for n in outputs]
    fv_ha = base["fert_vol_kgha"] if "fert_vol_kgha" in base else 0.0
    mv_ha = base["man_vol_kgha"] if "man_vol_kgha" in base else 0.0
    base["inputs_kgha"] = base[in_ha].sum(axis=1)
    base["outputs_kgha"] = base[out_ha].sum(axis=1)

    # CROPLAND surplus. FAO-comparable variant EXCLUDES the human recycled term (matches the FAO CNB
    # input set); the headline cropland surplus ADDS hum_crop (night soil + biosolids applied to cropland).
    hum_crop_ha = base["hum_crop_kgha"] if "hum_crop_kgha" in base.columns else 0.0
    base["surplus_cropland_faocomp_kgha"] = (base["inputs_kgha"] - base["outputs_kgha"] - fv_ha - mv_ha)
    base["surplus_cropland_kgha"] = base["surplus_cropland_faocomp_kgha"] + hum_crop_ha
    base["surplus_kgha"] = base["surplus_cropland_kgha"]          # backward-compatible alias
    # NUE on the FAO-comparable input set (hum_crop excluded), for the CNB comparison
    base["NUE"] = base["outputs_kgha"] / base["inputs_kgha"].where(base["inputs_kgha"] > 0)

    # FULL surplus (per ha of whole-country land area; all components incl. human waste)
    if nutrient == "N" and "LAND_HA" in base.columns:
        def kg(c):
            return base[c + "_kg"] if (c + "_kg") in base.columns else 0.0
        fv_full = base["fert_vol_full_kg"] if "fert_vol_full_kg" in base.columns else kg("fert_vol")
        mv_full = base["man_vol_full_kg"] if "man_vol_full_kg" in base.columns else kg("man_vol")
        land_area = base["LAND_HA"].where(base["LAND_HA"] > 0)
        # LANDSCAPE surplus = all land-surface inputs/outputs EXCLUDING human waste,
        # net of NH3 losses and grassland forage removal
        landscape_mass = (kg("fert") + kg("fert_grass") + kg("bnf") + kg("bnf_grass") + kg("dep_land")
                          + kg("man_crop") + kg("man_past") + kg("food_waste") + kg("hum_crop")
                          + kg("hum_sew_land") + kg("hum_landfill")
                          - kg("crop") - kg("forage") - fv_full - mv_full)
        base["surplus_landscape_kgha"] = landscape_mass / land_area
        # TOTAL surplus = landscape + SEWERED human discharge to surface water (non-sewered is on land,
        # already in landscape above). hum_sew_water falls back to full hum_sewage if the split is absent.
        hum_water = kg("hum_sew_water") if ("hum_sew_water_kg" in base.columns) else kg("hum_sewage")
        base["surplus_full_kgha"] = (landscape_mass + hum_water) / land_area
    base["nutrient"] = nutrient
    return base
