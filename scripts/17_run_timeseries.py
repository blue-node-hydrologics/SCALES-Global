#!/usr/bin/env python3
"""Run the SCALES-Global country budget across many years -> national + global N time series.

FAOSTAT spans 1961-2024, so fertilizer, manure, BNF and crop removal are fully time-resolved.
Deposition uses the constant ISIMIP 2014-proxy (dep_ox_country.csv); human waste is only
available for years present in population_country.csv (2020/2022) -> the AGRONOMIC surplus
(fert+manure+BNF+dep-crop, no human) is the clean, fully time-resolved series.

Caches the FAOSTAT reader so the multi-year loop reads each bulk once. Writes incrementally
to data/outputs/timeseries_N_global.csv (resumable) and keeps per-year national tables.

Usage:  python scripts/17_run_timeseries.py --start 1961 --end 2024
"""
import sys, argparse, functools
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import os; os.chdir(ROOT)
from src import config as cfgmod, faostat_api, pipeline  # noqa: E402

# cache the bulk reader: parse each domain CSV once, persist to parquet so EVERY
# process (not just this one) starts fast.
_CACHE = ROOT / "data/raw/faostat/_parquet_cache"; _CACHE.mkdir(parents=True, exist_ok=True)
_orig_read = faostat_api.read_domain


@functools.lru_cache(maxsize=None)
def _fast_read(domain, raw_dir):
    pq = _CACHE / f"{domain}.parquet"
    if pq.exists():
        return pd.read_parquet(pq)
    df = _orig_read(domain, raw_dir)
    try:
        df.to_parquet(pq)
    except Exception:  # noqa: BLE001
        pass
    return df


faostat_api.read_domain = _fast_read

GLOBAL_CSV = ROOT / "data/outputs/timeseries_N_global.csv"
COMPS = ["fert_kg", "bnf_kg", "dep_kg", "man_crop_kg", "hum_sewage_kg", "crop_kg",
         "fert_vol_kg", "man_vol_kg",
         "fert_grass_kg", "dep_land_kg", "man_past_kg", "fert_vol_full_kg", "man_vol_full_kg",
         "forage_kg", "bnf_grass_kg", "food_waste_kg", "hum_crop_kg", "hum_ns_kg", "hum_bio_kg",
         "hum_landfill_kg", "hum_sew_water_kg", "hum_sew_land_kg"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=1961)
    ap.add_argument("--end", type=int, default=2024)
    ap.add_argument("--force", action="store_true", help="recompute all years (truncate output)")
    args = ap.parse_args()
    cfg = cfgmod.load_config()
    out_dir = cfgmod.get_path(cfg, "outputs")

    done = set()
    if args.force:
        GLOBAL_CSV.write_text("")   # truncate (cloud Drive blocks rm; open-write works)
    elif GLOBAL_CSV.exists():
        done = set(pd.read_csv(GLOBAL_CSV)["year"].astype(int))

    import time
    for year in range(args.start, args.end + 1):
        if year in done:
            continue
        t0 = time.time()
        try:
            pipeline.compute_balance(cfg, year)
            t = pd.read_csv(out_dir / f"mass_balance_N_{year}.csv")
            g = {c: float(t[c].sum()) / 1e9 for c in COMPS if c in t}  # Tg N
            crop_ha = float(t["AREA_HA"].sum())
            land_ha = float(t["LAND_HA"].sum()) if "LAND_HA" in t else float("nan")
            fvol = g.get("fert_vol_kg", 0)   # cropland synthetic-fertilizer NH3 loss
            mvol = g.get("man_vol_kg", 0)    # cropland applied-manure NH3 loss
            # CROPLAND surplus. FAO-comparable variant excludes hum_crop; the headline cropland surplus
            # adds hum_crop (night soil + biosolids applied to cropland).
            hum_crop_Tg = g.get("hum_crop_kg", 0)
            agron_faocomp = (g["fert_kg"] + g["bnf_kg"] + g.get("dep_kg", 0) + g["man_crop_kg"]
                             - g["crop_kg"] - fvol - mvol)
            agron = agron_faocomp + hum_crop_Tg
            # LANDSCAPE surplus (whole-country land, NO human sewage): full fertilizer + deposition over
            # all land + applied & pasture manure + hum_crop, net of all NH3 losses and forage removal
            fvol_full = g.get("fert_vol_full_kg", fvol)
            mvol_full = g.get("man_vol_full_kg", mvol)
            forage = g.get("forage_kg", 0)
            # landscape includes non-sewered human discharge (hum_sew_land) and landfilled sludge
            # (hum_landfill, retained in the terrestrial landscape but not cropland)
            hum_sew_land_Tg = g.get("hum_sew_land_kg", 0)
            hum_sew_water_Tg = g.get("hum_sew_water_kg", g.get("hum_sewage_kg", 0))
            hum_landfill_Tg = g.get("hum_landfill_kg", 0)
            landscape = (g["fert_kg"] + g.get("fert_grass_kg", 0) + g["bnf_kg"] + g.get("bnf_grass_kg", 0)
                         + g.get("dep_land_kg", g.get("dep_kg", 0)) + g["man_crop_kg"]
                         + g.get("man_past_kg", 0) + g.get("food_waste_kg", 0) + hum_crop_Tg + hum_sew_land_Tg
                         + hum_landfill_Tg
                         - g["crop_kg"] - forage - fvol_full - mvol_full)
            full = landscape + hum_sew_water_Tg              # TOTAL = landscape + SEWERED discharge to water
            row = {"year": year, "fert_Tg": g["fert_kg"], "fert_grass_Tg": g.get("fert_grass_kg", 0),
                   "bnf_Tg": g["bnf_kg"], "dep_Tg": g.get("dep_kg", 0), "dep_land_Tg": g.get("dep_land_kg", 0),
                   "man_crop_Tg": g["man_crop_kg"], "man_past_Tg": g.get("man_past_kg", 0),
                   "hum_sewage_Tg": g.get("hum_sewage_kg", 0), "hum_sew_water_Tg": hum_sew_water_Tg,
                   "hum_sew_land_Tg": hum_sew_land_Tg, "crop_Tg": g["crop_kg"], "forage_Tg": forage,
                   "bnf_grass_Tg": g.get("bnf_grass_kg", 0),
                   "food_waste_Tg": g.get("food_waste_kg", 0), "hum_crop_Tg": hum_crop_Tg,
                   "hum_ns_Tg": g.get("hum_ns_kg", 0), "hum_bio_Tg": g.get("hum_bio_kg", 0),
                   "hum_landfill_Tg": hum_landfill_Tg,
                   "fert_vol_Tg": fvol, "man_vol_Tg": mvol,
                   "fert_vol_full_Tg": fvol_full, "man_vol_full_Tg": mvol_full,
                   "surplus_cropland_Tg": agron, "surplus_cropland_faocomp_Tg": agron_faocomp,
                   "surplus_landscape_Tg": landscape, "surplus_full_Tg": full,
                   "cropland_Mha": crop_ha / 1e6, "land_Mha": land_ha / 1e6,
                   "surplus_cropland_kgha": agron * 1e9 / crop_ha,
                   "surplus_cropland_faocomp_kgha": agron_faocomp * 1e9 / crop_ha,
                   "surplus_landscape_kgha": (landscape * 1e9 / land_ha) if land_ha == land_ha and land_ha > 0 else float("nan"),
                   "surplus_full_kgha": (full * 1e9 / land_ha) if land_ha == land_ha and land_ha > 0 else float("nan")}
            hdr = (not GLOBAL_CSV.exists()) or GLOBAL_CSV.stat().st_size == 0
            pd.DataFrame([row]).to_csv(GLOBAL_CSV, mode="a", header=hdr, index=False)
            print(f"  {year}: agron {agron:.1f} Tg (fert {g['fert_kg']:.1f}, man {g['man_crop_kg']:.1f}, "
                  f"bnf {g['bnf_kg']:.1f}, crop {g['crop_kg']:.1f})  [{time.time()-t0:.1f}s]")
        except Exception as e:  # noqa: BLE001
            print(f"  {year}: ERROR {e}")


if __name__ == "__main__":
    main()
