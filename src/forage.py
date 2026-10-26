"""FORAGE — grassland forage N removal (landscape + total surpluses only).

TREND-informed hybrid (Byrnes et al. 2020; Chang et al. 2026): grazing-livestock forage DEMAND
bounded by a pasture SUPPLY ceiling. Only the forage that is actually grazed or cut leaves the
grassland surface, so removal is the lesser of (a) the N taken up by grazing animals and (b) the
maximum N the pasture can yield:

    FORAGE(i,t) = min[ MAN_PRP(i,t) + N_prod,graze(i,t) ,  AREA_past(i,t) * p_past ]

  demand   = MAN_PRP (FAOSTAT EMN "left on pasture", 72380; dung/urine returned to pasture)
             + N_prod,graze (ruminant meat/milk/wool, FAOSTAT QCL production x product N content,
               coefficients/animal_product_N.csv, grazing-attributed).
  ceiling  = national permanent-pasture area (FAOSTAT RL, PASTURE_HA) x p_past, the area-based
             max pasture N uptake (coefficients/pasture_uptake.csv, by IPCC region; GLOBAL default).
             The ceiling caps removal in intensively stocked / feed-supplemented systems where
             grazing demand would exceed what the pasture can produce.

Keeping MAN_PRP as an input and subtracting FORAGE makes the grassland balance net to external
inputs minus animal-product export, reconciling with the manure term by construction. See
Affects the landscape and total surpluses only; cropland unchanged.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from . import faostat_prep as prep
from . import manure
from . import utils
from .config import coef

DEFAULT_P_PAST = 100.0   # kg N ha-1 yr-1 — pasture N-uptake supply ceiling (calibrate; see CSV)


def _product_N(cfg, raw_dir: Path, year: int) -> pd.DataFrame:
    """N exported in grazing-animal products = QCL production x product N content. [ISO3, prod_n_kg]."""
    codes = cfg["faostat"]["codes"]
    ct = pd.read_csv(coef(cfg, "animal_product_N.csv"))
    ct = ct[ct["system"].astype(str).str.lower() == "grazing"][["item_code", "n_content_kg_per_t"]]
    prod = prep.tidy("QCL", cfg, raw_dir, year=year,
                     element=codes["crop_element_production"], item=None, value_name="tonnes")
    df = prod.merge(ct, left_on="Item Code", right_on="item_code", how="inner")
    if df.empty:
        print("  [warn] FORAGE: no QCL ruminant-product items matched animal_product_N.csv "
              "(check item codes); product-export term = 0")
        return pd.DataFrame({"ISO3": [], "prod_n_kg": []})
    df["prod_n_kg"] = df["tonnes"].astype(float) * df["n_content_kg_per_t"].astype(float)
    print(f"  FORAGE: matched {df['Item Code'].nunique()} ruminant-product items "
          f"{sorted(df['Item Code'].unique().tolist())}")
    return df.groupby("ISO3", as_index=False)["prod_n_kg"].sum()


def _p_past_by_country(cfg, raw_dir: Path) -> tuple[dict, float]:
    """Return (ISO3 -> p_past kg/ha, global_default). Uses pasture_uptake.csv by IPCC region,
    mapped to ISO3 via the manure region map; falls back to the GLOBAL row / DEFAULT_P_PAST."""
    glob = DEFAULT_P_PAST
    iso2p: dict = {}
    try:
        pt = pd.read_csv(coef(cfg, "pasture_uptake.csv"))
        reg2p = dict(zip(pt["ipcc_region"].astype(str), pt["p_past_kg_ha"].astype(float)))
        glob = reg2p.get("GLOBAL", DEFAULT_P_PAST)
        rmap = utils.load_coef(coef(cfg, cfg["manure"]["region_map_file"]),
                               "ISO3 -> IPCC region map", required_cols=["ISO3", "ipcc_region"])
        for _, r in rmap.iterrows():
            iso2p[r["ISO3"]] = reg2p.get(str(r["ipcc_region"]), glob)
    except Exception:  # noqa: BLE001
        pass
    return iso2p, glob


def product_n(cfg, raw_dir: Path, year: int) -> pd.DataFrame:
    """National N exported in grazing-animal products. [ISO3, prod_n_kg].

    Public entry point for the grid-scale Eq. 29 in downscale.py, which needs the demand
    components separately: MAN_graze comes from the gridded PRP field and N_prod,graze is
    downscaled on the same grazing proxy."""
    return _product_N(cfg, raw_dir, year)


def forage_removal(cfg, raw_dir: Path, year: int, areas: pd.DataFrame | None = None) -> pd.DataFrame:
    """LEGACY national-scale Eq. 29. Kept for the national timeseries only.

    Do NOT use to build the gridded forage term. Taking the minimum on national aggregates
    over-credits removal, because min(sum(demand), sum(supply)) >= sum(min(demand, supply)):
    a national ceiling cannot see the concentration of livestock onto part of a country's
    grazing land. Against the 2000 grids the overestimate is 77% at p_graze = 100 kg N/ha.
    downscale.py now evaluates Eq. 29 per cell and reports the national total as the sum.

    Grassland forage N removed = min(demand, supply ceiling). [ISO3, forage_kg].

    demand  = MAN_PRP + grazing-animal-product N.
    ceiling = PASTURE_HA x p_past(region)  (applied only if `areas` carries PASTURE_HA).
    """
    prp = manure.pasture(cfg, raw_dir, year).rename(columns={"manure_kg": "prp_kg"})
    prod = _product_N(cfg, raw_dir, year)
    m = prp.merge(prod, on="ISO3", how="outer").fillna(0.0)
    m["demand_kg"] = m["prp_kg"] + m["prod_n_kg"]

    if areas is not None and "PASTURE_HA" in areas.columns:
        iso2p, glob = _p_past_by_country(cfg, raw_dir)
        m = m.merge(areas[["ISO3", "PASTURE_HA"]], on="ISO3", how="left")
        m["p_past"] = m["ISO3"].map(iso2p).fillna(glob)
        m["ceiling_kg"] = m["PASTURE_HA"].fillna(0.0) * m["p_past"]
        m["forage_kg"] = np.where(m["ceiling_kg"] > 0,
                                  np.minimum(m["demand_kg"], m["ceiling_kg"]), m["demand_kg"])
        nbind = int(((m["ceiling_kg"] > 0) & (m["demand_kg"] > m["ceiling_kg"])).sum())
        print(f"  FORAGE: pasture supply ceiling binds in {nbind} countries "
              f"(demand > PASTURE_HA x p_past)")
    else:
        m["forage_kg"] = m["demand_kg"]   # no ceiling if pasture area unavailable
    from . import dissolution
    return dissolution.redistribute_table(m[["ISO3", "forage_kg"]], cfg, raw_dir, "pasture")
