"""Orchestrate the global cropland mass balance: download, compute, validate."""
from __future__ import annotations
from pathlib import Path
import pandas as pd

from . import (config as cfgmod, faostat_api, country_areas,
               fertilizer, crop_uptake, bnf, manure, deposition, human, mass_balance,
               fert_volatilization, man_volatilization, forage, dissolution)


def download(cfg) -> dict:
    raw = cfgmod.get_path(cfg, "raw")
    return faostat_api.download_all(cfg, raw)


def build_components(cfg, raw, year: int, areas: pd.DataFrame, nutrient: str = "N") -> dict:
    """All national budget components for one year and nutrient, back-cast onto dissolution
    successors. Split out of compute_balance  so the coefficient-level Monte Carlo
    (src/uncertainty_mc.py) uses exactly the same component calculations as the deterministic run."""
    comps = {
        "fert":   fertilizer.compute(cfg, raw, year, nutrient),
        "crop":   crop_uptake.compute(cfg, raw, year, nutrient),
        "bnf":    bnf.compute(cfg, raw, year, nutrient),
        "man_crop": manure.compute(cfg, raw, year, nutrient),
        "hum_sewage":  human.compute(cfg, raw, year, nutrient),   # environmental discharge (total only)
        "dep":    deposition.compute(cfg, raw, year, areas, nutrient),
    }
    if nutrient == "N":
        # human sewage discharge split by sewer connection: sewered -> surface water (total only);
        # non-sewered -> deposited on land (landscape + total). hum_sewage kept as the total reference.
        _hs = comps["hum_sewage"]
        comps["hum_sew_water"] = _hs[["ISO3", "hum_sew_water_kg"]] if "hum_sew_water_kg" in _hs else _hs.rename(columns={_hs.columns[1]: "hum_sew_water_kg"})[["ISO3", "hum_sew_water_kg"]]
        comps["hum_sew_land"] = _hs[["ISO3", "hum_sew_land_kg"]] if "hum_sew_land_kg" in _hs else pd.DataFrame(columns=["ISO3", "hum_sew_land_kg"])
        comps["hum_sewage"] = _hs[["ISO3", "hum_sewage_kg"]]
        # --- cropland NH3 volatilization losses (soil-surface basis) ---
        comps["fert_vol"] = fert_volatilization.volatilization_loss(cfg, raw, year, comps["fert"])
        comps["man_vol"] = man_volatilization.volatilization_loss(cfg, comps["man_crop"], scope="applied")
        # --- extra national components for the FULL (whole-country) surplus ---
        comps["fert_grass"] = fertilizer.grassland(cfg, raw, year, nutrient)   # (1-CF) grassland fertilizer
        comps["man_past"] = manure.pasture(cfg, raw, year)                     # manure left on pasture (EMN 72380)
        # deposition over ALL land — PIECEWISE (matches the gridded engine):
        #   DEP_land = W_i * A_crop  +  Ibar_noncrop_i * (A_land - A_crop)
        # cropland gets the Vishwakarma WL rate W_i; non-cropland gets the native ISIMIP rate
        # (dep_noncrop table, scripts/31d). Falls back to W_i over ALL land if that table is absent.
        dep_rate = deposition.dep_kgha_for_year(cfg, year)          # W_i (cropland rate)
        if dep_rate is not None and "LAND_HA" in areas.columns:
            acols = ["ISO3", "AREA_HA", "LAND_HA"] if "AREA_HA" in areas.columns else ["ISO3", "LAND_HA"]
            dl = areas[acols].merge(dep_rate, on="ISO3", how="left")
            nc = deposition.dep_noncrop_kgha_for_year(cfg, year)   # Ibar_noncrop (kg/ha) or None
            if nc is not None and "AREA_HA" in dl.columns:
                dl = dl.merge(nc, on="ISO3", how="left")
                # where the non-cropland mean is missing, fall back to the cropland rate
                dl["dep_noncrop_kgha"] = dl["dep_noncrop_kgha"].fillna(dl["dep_ox_kgha"])
                noncrop_ha = (dl["LAND_HA"].astype(float) - dl["AREA_HA"].astype(float)).clip(lower=0.0)
                dl["dep_land_kg"] = (dl["dep_ox_kgha"].astype(float) * dl["AREA_HA"].astype(float)
                                     + dl["dep_noncrop_kgha"].astype(float) * noncrop_ha)
            else:                                                   # no non-cropland table -> legacy form
                dl["dep_land_kg"] = dl["dep_ox_kgha"].astype(float) * dl["LAND_HA"].astype(float)
            comps["dep_land"] = dl[["ISO3", "dep_land_kg"]].dropna()
        # full NH3 losses: fertilizer on ALL agricultural land, and applied + pasture manure
        full_fert = fertilizer.compute(cfg, raw, year, nutrient, apply_cf=False)
        comps["fert_vol_full"] = fert_volatilization.volatilization_loss(cfg, raw, year, full_fert)
        mv_app = man_volatilization.volatilization_loss(cfg, comps["man_crop"], scope="applied")
        mv_pas = man_volatilization.volatilization_loss(cfg, comps["man_past"], scope="pasture")
        mvf = (mv_app.rename(columns={"man_vol_kg": "_a"})
               .merge(mv_pas.rename(columns={"man_vol_kg": "_p"}), on="ISO3", how="outer").fillna(0.0))
        mvf["man_vol_full_kg"] = mvf["_a"] + mvf["_p"]
        comps["man_vol_full"] = mvf[["ISO3", "man_vol_full_kg"]]
        # grassland forage N removal (landscape + total): min(demand, pasture supply ceiling)
        comps["forage"] = forage.forage_removal(cfg, raw, year, areas=areas)
        # pasture/forage-legume BNF over permanent pasture (landscape + total input)
        comps["bnf_grass"] = bnf.grassland(cfg, raw, year, areas)
        # consumer food-waste N (landscape + total input): the food N not eaten -> landfill/compost
        comps["food_waste"] = human.food_waste(cfg, raw, year, nutrient)
        # human excreta recycled onto CROPLAND (cropland + landscape + total input): night soil
        # (hum_ns) + biosolids (hum_bio); hum_crop = hum_ns + hum_bio is the term used in the surplus
        _hr = human.human_recycle(cfg, raw, year, nutrient)
        comps["hum_crop"] = _hr[["ISO3", "hum_crop_kg"]] if len(_hr) else _hr
        comps["hum_ns"] = _hr[["ISO3", "hum_ns_kg"]] if len(_hr) else pd.DataFrame(columns=["ISO3", "hum_ns_kg"])
        comps["hum_bio"] = _hr[["ISO3", "hum_bio_kg"]] if len(_hr) else pd.DataFrame(columns=["ISO3", "hum_bio_kg"])
        # landfilled sewage sludge -> retained in the terrestrial landscape (landscape/total, not cropland)
        comps["hum_landfill"] = (_hr[["ISO3", "hum_landfill_kg"]] if (len(_hr) and "hum_landfill_kg" in _hr)
                                 else pd.DataFrame(columns=["ISO3", "hum_landfill_kg"]))
    # back-cast FAOSTAT dissolution aggregates (USSR, Yugoslavia, ...) onto modern successor
    # states before assembling the national budget, so pre-split years are spatially resolved
    comps = dissolution.redistribute(comps, cfg, raw)
    return comps


def compute_balance(cfg, year: int) -> dict:
    raw = cfgmod.get_path(cfg, "raw")
    out_dir = cfgmod.get_path(cfg, "outputs")
    areas = country_areas.load_areas(cfg, raw, year)   # self-redistributes dissolution aggregates
    results = {}
    for nutrient in cfg["nutrients"]:
        comps = build_components(cfg, raw, year, areas, nutrient)
        table = mass_balance.assemble(comps, areas, nutrient)
        out_path = out_dir / f"mass_balance_{nutrient}_{year}.csv"
        table.to_csv(out_path, index=False)
        results[nutrient] = {
            "n_countries": int(table["ISO3"].nunique()),
            "mean_surplus": float(table["surplus_kgha"].mean()),
            "global_inputs_Tg": float(table[[c for c in table if c.endswith("_kg") and
                                             not c.startswith("crop")]].sum().sum() / 1e9),
            "path": str(out_path),
        }
    return results


def validate(cfg, year: int) -> pd.DataFrame:
    """Compare our N surplus vs FAOSTAT ESB (Cropland Nutrient Budget) for `year`."""
    raw = cfgmod.get_path(cfg, "raw")
    from . import faostat_prep as prep
    esb = faostat_api.read_domain("ESB", raw)
    esb = prep.to_iso3(faostat_api.filter_rows(esb, year=year), cfg)
    ours = pd.read_csv(cfgmod.get_path(cfg, "outputs") / f"mass_balance_N_{year}.csv")
    return ours, esb
