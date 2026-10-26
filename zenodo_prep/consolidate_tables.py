#!/usr/bin/env python3
"""
Consolidate the SCALES-Global CSV outputs into two tidy, self-contained tables
with uncertainty folded in, for public deposit.

Inputs:
    <outputs>/mass_balance_N_<year>.csv      per-country budget, one file per year 1961-2023
    <outputs>/timeseries_N_global.csv        global annual totals (Tg N)
    <uncertainty>/uncertainty_N_<year>.csv   per-country surplus Monte Carlo uncertainty
    <uncertainty>/uncertainty_N_timeseries.csv  global annual Monte Carlo band
    (both from scripts/46_run_uncertainty_mc.py; <uncertainty> defaults to <outputs>/uncertainty_mc)

Outputs (in <out>):
    national_budget_N_1961_2023.csv      long format: one row per (ISO3, year),
                                         all budget terms + surpluses + NUE +
                                         surplus uncertainty (mean/sd/95%CI/CV)
    global_timeseries_N_1961_2023.csv    global annual totals + MC uncertainty band
    per_year_national/                   the original per-year files, copied for
                                         anyone who wants them

The uncertainty "total" scope is renamed "full" to match the budget's
surplus_full (surplus incl. human sewage).

Usage:
    python consolidate_tables.py --outputs data/outputs --out data/outputs/zenodo/tables
    python consolidate_tables.py selftest
"""
from __future__ import annotations
import argparse
import glob
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCOPE_RENAME = {"total": "full"}  # harmonize uncertainty scope -> budget naming


def _year_from(path: str) -> int | None:
    for tok in Path(path).stem.split("_"):
        if len(tok) == 4 and tok.isdigit():
            return int(tok)
    return None


def build_national(outputs: Path, out: Path, unc_dir: Path | None = None) -> Path:
    mb_files = sorted(glob.glob(str(outputs / "mass_balance_N_*.csv")))
    mb_files = [f for f in mb_files if _year_from(f) is not None]
    frames = []
    for f in mb_files:
        y = _year_from(f)
        df = pd.read_csv(f)
        if len(df) < 10:     # skip stub files (e.g. an empty 2024)
            continue
        df.insert(1, "year", y)
        frames.append(df)
    if not frames:
        raise SystemExit(f"no mass_balance_N_*.csv found in {outputs}")
    nat = pd.concat(frames, ignore_index=True)

    # attach per-country surplus uncertainty, if present
    unc_files = sorted(glob.glob(str((unc_dir or outputs) / "uncertainty_N_*.csv")))
    unc_files = [f for f in unc_files
                 if _year_from(f) is not None and "_global" not in Path(f).name]
    uframes = []
    for f in unc_files:
        y = _year_from(f)
        u = pd.read_csv(f)
        if "ISO3" not in u.columns:
            continue
        u.insert(1, "year", y)
        uframes.append(u)
    if uframes:
        unc = pd.concat(uframes, ignore_index=True)
        unc = unc.rename(columns={c: c.replace("_total_", "_full_") for c in unc.columns})
        nat = nat.merge(unc, on=["ISO3", "year"], how="left")
        n_cov = unc["year"].nunique()
        print(f"  national uncertainty merged for {n_cov} year(s)")
    else:
        print("  (no per-country uncertainty files found -- run scripts/46; "
              "budget written without uncertainty columns)")

    nat = nat.sort_values(["ISO3", "year"]).reset_index(drop=True)
    dest = out / "national_budget_N_1961_2023.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    nat.to_csv(dest, index=False)
    print(f"  wrote {dest.name}: {len(nat):,} rows x {nat.shape[1]} cols "
          f"({nat['ISO3'].nunique()} countries, "
          f"{int(nat['year'].min())}-{int(nat['year'].max())})")
    return dest


def build_global(outputs: Path, out: Path, unc_dir: Path | None = None) -> Path:
    ts = pd.read_csv(outputs / "timeseries_N_global.csv")
    up = (unc_dir or outputs) / "uncertainty_N_timeseries.csv"
    if up.exists():
        u = pd.read_csv(up)
        u = u.rename(columns={c: c.replace("total_", "full_") for c in u.columns})
        ts = ts.merge(u, on="year", how="left")
        print("  global time-series uncertainty band merged")
    else:
        print("  (uncertainty_N_timeseries.csv not found; global series written w/o band)")
    dest = out / "global_timeseries_N_1961_2023.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    ts.to_csv(dest, index=False)
    print(f"  wrote {dest.name}: {len(ts)} years x {ts.shape[1]} cols")
    return dest


def copy_per_year(outputs: Path, out: Path):
    d = out / "per_year_national"
    d.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in sorted(glob.glob(str(outputs / "mass_balance_N_*.csv"))):
        if _year_from(f) is None:
            continue
        if pd.read_csv(f, nrows=12).shape[0] < 10:
            continue
        shutil.copy2(f, d / Path(f).name)
        n += 1
    print(f"  copied {n} per-year national files -> per_year_national/")


def selftest():
    import tempfile
    tmp = Path(tempfile.mkdtemp()); src = tmp / "outputs"; src.mkdir()
    # fake 3 years of a national budget + matching uncertainty, 12 countries
    iso = [f"C{i:02d}" for i in range(12)]
    n = len(iso); rng = np.random.default_rng(0)
    for y in (2000, 2001, 2002):
        pd.DataFrame({
            "ISO3": iso, "AREA_HA": rng.uniform(5e7, 1e8, n),
            "surplus_cropland_kgha": rng.uniform(10, 60, n),
            "surplus_landscape_kgha": rng.uniform(5, 20, n),
            "surplus_full_kgha": rng.uniform(6, 22, n), "NUE": rng.uniform(.4, .6, n),
            "nutrient": "N",
        }).to_csv(src / f"mass_balance_N_{y}.csv", index=False)
        d = {"ISO3": iso}
        for scope in ("cropland", "landscape", "total"):
            m = rng.uniform(3, 12, n)
            d[f"surplus_{scope}_mean_TgN"] = m
            d[f"surplus_{scope}_sd_TgN"] = m * .1
            d[f"surplus_{scope}_p2.5_TgN"] = m * .8
            d[f"surplus_{scope}_p97.5_TgN"] = m * 1.2
            d[f"surplus_{scope}_cv"] = np.full(n, .1)
        pd.DataFrame(d).to_csv(src / f"uncertainty_N_{y}.csv", index=False)
    # a stub file that must be skipped
    pd.DataFrame({"ISO3": ["ZZZ"]}).to_csv(src / "mass_balance_N_2024.csv", index=False)
    # global series + band
    yrs = [2000, 2001, 2002]
    pd.DataFrame({"year": yrs, "surplus_cropland_Tg": [60, 61, 62],
                  "surplus_full_Tg": [70, 71, 72]}).to_csv(
        src / "timeseries_N_global.csv", index=False)
    pd.DataFrame({"year": yrs, "cropland_central_Tg": [60, 61, 62],
                  "cropland_sd_Tg": [6, 6, 6], "total_central_Tg": [70, 71, 72],
                  "total_sd_Tg": [7, 7, 7]}).to_csv(
        src / "uncertainty_N_timeseries.csv", index=False)

    out = tmp / "tables"
    print("[selftest] national:")
    natp = build_national(src, out)
    print("[selftest] global:")
    globp = build_global(src, out)
    copy_per_year(src, out)

    nat = pd.read_csv(natp)
    ok = True
    ok &= len(nat) == 36 and nat["year"].nunique() == 3           # 12 iso x 3 yr, stub skipped
    ok &= "surplus_full_mean_TgN" in nat.columns                  # total -> full rename
    ok &= "surplus_total_mean_TgN" not in nat.columns
    ok &= nat["surplus_cropland_sd_TgN"].notna().all()            # uncertainty joined
    g = pd.read_csv(globp)
    ok &= "full_sd_Tg" in g.columns and "total_sd_Tg" not in g.columns
    print("[selftest]", "ALL CHECKS PASSED" if ok else "*** CHECKS FAILED ***")
    print("  national columns:", list(nat.columns))
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", nargs="?", default="run", choices=["run", "selftest"])
    ap.add_argument("--outputs")
    ap.add_argument("--out")
    ap.add_argument("--uncertainty", default=None,
                    help="folder with the Monte Carlo outputs (default: <outputs>/uncertainty_mc)")
    args = ap.parse_args()
    if args.cmd == "selftest":
        sys.exit(0 if selftest() else 1)
    if not args.outputs or not args.out:
        ap.error("--outputs and --out are required for 'run'")
    outputs = Path(args.outputs); out = Path(args.out)
    unc = Path(args.uncertainty) if args.uncertainty else outputs / "uncertainty_mc"
    print("[national]"); build_national(outputs, out, unc)
    print("[global]"); build_global(outputs, out, unc)
    print("[per-year copy]"); copy_per_year(outputs, out)
    print(f"[done] -> {out}")


if __name__ == "__main__":
    main()
