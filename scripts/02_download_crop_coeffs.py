#!/usr/bin/env python3
"""Build coefficients/crop_uptake.csv from the Ludemann et al. Dryad dataset
(doi:10.5061/dryad.n2z34tn0x).

Dryad blocks automated downloads, so the workflow is:
  1. Download the dataset zip from https://datadryad.org/dataset/doi:10.5061/dryad.n2z34tn0x
     (the "Download dataset" button) and drop the .zip into data/raw/ludemann_crop_coeffs/.
  2. Run this script. It extracts the zip and maps World_crop_coefficients_for_UN_FAO.csv
     (Tier-1 world values, the exact set the FAO Cropland Nutrient Budget uses) into the
     crop_uptake.csv schema.

Usage:  python scripts/02_download_crop_coeffs.py

NOTE: a country-specific (Tier-2) upgrade is available in the same zip
(Tier_1_and_2_crop_coefficients.csv) — see scripts/02b_crop_coeffs_tier2.py (optional).
"""
import sys, zipfile, glob
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import os; os.chdir(ROOT)
from src import config as cfgmod  # noqa: E402

WORLD = "World_crop_coefficients_for_UN_FAO.csv"
PAGE = "https://datadryad.org/dataset/doi:10.5061/dryad.n2z34tn0x"


def _locate_world(src_dir: Path) -> Path:
    hit = list(src_dir.rglob(WORLD))
    if hit:
        return hit[0]
    zips = list(src_dir.glob("*.zip"))
    if zips:
        with zipfile.ZipFile(zips[0]) as z:
            z.extractall(src_dir / "_x")
        hit = list(src_dir.rglob(WORLD))
        if hit:
            return hit[0]
    raise SystemExit(
        f"\n{'='*72}\nCould not find {WORLD}.\n"
        f"  1. Download the dataset zip from {PAGE}\n"
        f"  2. Put the .zip in {src_dir}\n  3. Re-run this script.\n{'='*72}")


def main():
    cfg = cfgmod.load_config()
    src_dir = cfgmod.get_path(cfg, "raw") / "ludemann_crop_coeffs"
    src_dir.mkdir(parents=True, exist_ok=True)
    world = _locate_world(src_dir)
    df = pd.read_csv(world)
    out = pd.DataFrame({
        "fao_item_code": df["item_code"].astype("Int64"),
        "item_name": df["item"],
        "removal_kg_per_t_N": pd.to_numeric(df["N_kg_per_t_fresh_wt"], errors="coerce"),
        "removal_kg_per_t_P": pd.to_numeric(df["P_kg_per_t_fresh_wt"], errors="coerce"),
        "moisture_basis": "fresh weight",
        "status": "ACTIVE",
        "source": "Ludemann et al. 2023 World/Tier-1 (Dryad 10.5061/dryad.n2z34tn0x)",
    }).dropna(subset=["removal_kg_per_t_N"])
    out_csv = cfgmod.get_path(cfg, "coefficients") / "crop_uptake.csv"
    out.to_csv(out_csv, index=False)
    print(f"wrote {len(out)} crop coefficients -> {out_csv}")


if __name__ == "__main__":
    main()
