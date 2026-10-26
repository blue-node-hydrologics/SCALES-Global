"""CROP — nutrient removal in harvested crop products.

  CROP(country) = sum_crop[ production_tonnes(crop) * removal_coef(crop, nutrient) ]

production = FAOSTAT QCL (element 5510, tonnes at standard moisture).
removal_coef = Ludemann et al. "World crop coefficients for UN FAO"
  (Dryad 10.5061/dryad.n2z34tn0x), kg nutrient / tonne product (elemental P; no
  P2O5 conversion). This is the exact set the FAO CNB uses, so it auto-aligns with
  the benchmark.  -> coefficients/crop_uptake.csv  (NEEDS SOURCE: see checklist)
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
from . import faostat_prep as prep
from . import utils
from .config import coef


def _coef_table(cfg, nutrient: str) -> pd.DataFrame:
    path = coef(cfg, "crop_uptake.csv")
    col = {"N": "removal_kg_per_t_N", "P": "removal_kg_per_t_P"}[nutrient]
    df = utils.load_coef(path, "crop removal coefficients (Ludemann/Dryad)",
                         required_cols=["fao_item_code", col])
    return df[["fao_item_code", col]].rename(columns={col: "coef"})


def compute(cfg, raw_dir: Path, year: int, nutrient: str = "N") -> pd.DataFrame:
    codes = cfg["faostat"]["codes"]
    prod = prep.tidy("QCL", cfg, raw_dir, year=year,
                     element=codes["crop_element_production"], item=None, value_name="tonnes")
    ctab = _coef_table(cfg, nutrient)
    df = prod.merge(ctab, left_on="Item Code", right_on="fao_item_code", how="inner")
    df["crop_kg"] = df["tonnes"].astype(float) * df["coef"].astype(float)
    out = df.groupby("ISO3", as_index=False)["crop_kg"].sum()
    from . import dissolution
    return dissolution.redistribute_table(out, cfg, raw_dir, "crop")
