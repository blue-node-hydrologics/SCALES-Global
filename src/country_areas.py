"""Area denominators from FAOSTAT RL (Land Use), in hectares.

Cropland scope -> normalize by cropland area (item 6620; RL element 5110 is in
1000 ha, so x1000 -> ha). Pasture area also returned for the later PAST pass.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
from . import faostat_prep as prep


def load_areas(cfg, raw_dir: Path, year: int) -> pd.DataFrame:
    codes = cfg["faostat"]["codes"]
    crop = prep.tidy("RL", cfg, raw_dir, year=year, element=codes["land_element_area"],
                     item=codes["land_item_cropland"], value_name="kha_crop")
    past = prep.tidy("RL", cfg, raw_dir, year=year, element=codes["land_element_area"],
                     item=codes["land_item_pasture"], value_name="kha_past")
    crop = crop.groupby("ISO3", as_index=False)["kha_crop"].sum()
    past = past.groupby("ISO3", as_index=False)["kha_past"].sum()
    df = crop.merge(past, on="ISO3", how="outer")
    # total land area (denominator for the FULL surplus); NaN if the RL land item is absent
    land_code = codes.get("land_item_land")
    if land_code is not None:
        try:
            land = prep.tidy("RL", cfg, raw_dir, year=year, element=codes["land_element_area"],
                             item=land_code, value_name="kha_land")
            land = land.groupby("ISO3", as_index=False)["kha_land"].sum()
            df = df.merge(land, on="ISO3", how="left")
        except Exception:  # noqa: BLE001
            df["kha_land"] = float("nan")
    else:
        df["kha_land"] = float("nan")
    df["AREA_HA"] = df["kha_crop"].fillna(0.0) * 1000.0          # cropland ha (cropland denominator)
    df["PASTURE_HA"] = df["kha_past"].fillna(0.0) * 1000.0
    df["LAND_HA"] = df["kha_land"].astype("float64") * 1000.0    # total land ha (FULL denominator); NaN if missing
    out = df[df["AREA_HA"] > 0][["ISO3", "AREA_HA", "PASTURE_HA", "LAND_HA"]].reset_index(drop=True)
    # back-cast dissolution aggregates (USSR, ...) onto successor states so pre-split years have a
    # cropland/pasture/land denominator (see src/dissolution.py)
    from . import dissolution
    return dissolution.redistribute_areas(out, cfg, raw_dir)
