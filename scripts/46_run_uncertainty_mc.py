#!/usr/bin/env python3
"""Coefficient-level Monte Carlo uncertainty for the SCALES-Global national budget.

Each coefficient is drawn from its own distribution (coefficients/uncertainty_spec.csv and
uncertainty_region_cv.csv) and every term is recomputed from the drawn values; see
src/uncertainty_mc.py.
left in place for comparison.

Outputs (default data/outputs/uncertainty_mc/):
  uncertainty_N_<year>.csv            per-country surplus statistics (cropland / landscape / total)
  uncertainty_N_<year>_global.csv     global surplus statistics
  uncertainty_terms_N_<year>.csv      global statistics for each budget term (Figure 6c)
  cov_N_<year>.npz                    per-country relative covariance of the terms (for script 47)
  uncertainty_N_timeseries.csv        one row per year (with --start/--end)
  global_draws_N_<yr>.npz             global surplus and term totals for every realization (for Figure 6)
  validation_central_N_<year>.csv     with --validate: central reproduction check
  forage_lattice/forage_lattice_<year>.npz   cached grid-scale forage lattice

Usage:
  python scripts/46_run_uncertainty_mc.py --year 2000 --validate          # check first
  python scripts/46_run_uncertainty_mc.py --start 1961 --end 2023 --draws 10000
"""
import sys
import argparse
import functools
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import os; os.chdir(ROOT)                                           # noqa: E402,E702
from src import config as cfgmod, faostat_api                        # noqa: E402
from src import uncertainty_mc as MC                                 # noqa: E402

# same FAOSTAT reader cache as scripts/17 and 26: parse each bulk once, persist to parquet
_PQ = ROOT / "data/raw/faostat/_parquet_cache"; _PQ.mkdir(parents=True, exist_ok=True)
_orig_read = faostat_api.read_domain


@functools.lru_cache(maxsize=None)
def _fast_read(domain, raw_dir):
    pq = _PQ / f"{domain}.parquet"
    if pq.exists():
        return pd.read_parquet(pq)
    df = _orig_read(domain, raw_dir)
    try:
        df.to_parquet(pq)
    except Exception:  # noqa: BLE001
        pass
    return df


faostat_api.read_domain = _fast_read


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--year", type=int)
    ap.add_argument("--start", type=int)
    ap.add_argument("--end", type=int)
    ap.add_argument("--draws", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=1000, help="draws per batch (memory)")
    ap.add_argument("--grids", default=None, help="folder with surplus_N_<year>_0p05.nc (default: outputs/grids)")
    ap.add_argument("--out", default=None, help="output folder (default: outputs/uncertainty_mc)")
    ap.add_argument("--no-grid-forage", action="store_true",
                    help="skip the grid-scale forage lattice; use the national forage with the fallback CV")
    ap.add_argument("--validate", action="store_true",
                    help="write the central-reproduction check for each year (recommended on the first run)")
    a = ap.parse_args()
    if a.year:
        years = [a.year]
    elif a.start and a.end:
        years = list(range(a.start, a.end + 1))
    else:
        ap.error("give --year or --start/--end")

    cfg = cfgmod.load_config()
    outputs = cfgmod.get_path(cfg, "outputs")
    out = Path(a.out) if a.out else outputs / "uncertainty_mc"
    out.mkdir(parents=True, exist_ok=True)
    grids = Path(a.grids) if a.grids else outputs / "grids"
    spec, rcv = MC.load_spec()
    s = pd.read_csv(MC.SPEC_FILE)
    flagged = s[~s["status"].astype(str).isin(["AGREED", "FALLBACK"])]
    if len(flagged):
        print("  NOTE: coefficients with provisional CVs (uncertainty_spec.csv 'status'): "
              + ", ".join(f"{r.coef}={r.cv if pd.notna(r.cv) else 'regional'} [{r.status}]"
                          for r in flagged.itertuples()))

    ts_rows = []
    for year in years:
        t0 = time.time()
        inp, comps, areas = MC.load_year_inputs(cfg, year, grids_dir=None if a.no_grid_forage else grids,
                                                lattice_dir=out / "forage_lattice",
                                                use_grid_forage=not a.no_grid_forage)
        for n_ in inp.notes:
            print(f"  [{year}] note: {n_}")
        if a.validate:
            v = MC.validate_central(inp, comps, areas)
            v.to_csv(out / f"validation_central_N_{year}.csv", index=False)
            print(v.to_string(index=False, float_format=lambda x: f"{x:.6g}"))
        res = MC.run_ensemble(inp, spec, rcv, n=a.draws, seed=a.seed + year, chunk=a.chunk)
        MC.country_summary(res).to_csv(out / f"uncertainty_N_{year}.csv", index=False)
        g = MC.global_summary(res)
        g.to_csv(out / f"uncertainty_N_{year}_global.csv", index=False)
        MC.term_summary(res).to_csv(out / f"uncertainty_terms_N_{year}.csv", index=False)
        MC.save_covariance(res, out / f"cov_N_{year}.npz")
        MC.save_global_draws(res, out / f"global_draws_N_{year}.npz")
        row = {"year": year}
        for scope, r in g.set_index("scope").iterrows():
            for k in ("central_Tg", "mc_mean_Tg", "sd_Tg", "p2.5_Tg", "p25_Tg", "p50_Tg", "p75_Tg", "p97.5_Tg",
                      "rel_uncertainty_pct"):
                row[f"{scope}_{k}"] = r[k]
        ts_rows.append(row)
        print(f"  [{year}] done in {time.time()-t0:.0f}s")

    if len(years) > 1:
        ts = pd.DataFrame(ts_rows)
        f = out / "uncertainty_N_timeseries.csv"
        if f.exists():                       # merge with years already on disk
            old = pd.read_csv(f)
            ts = pd.concat([old[~old["year"].isin(ts["year"])], ts]).sort_values("year")
        ts.to_csv(f, index=False)
        print(f"  wrote {f}")


if __name__ == "__main__":
    main()
