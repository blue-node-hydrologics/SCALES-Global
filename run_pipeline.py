#!/usr/bin/env python3
"""CLI for the global cropland N (+P scaffold) mass balance.

Examples:
  python run_pipeline.py --download                      # pull all FAOSTAT domains
  python run_pipeline.py --year 2022 --compute
  python run_pipeline.py --year 2020 --compute --validate   # vs FAO CNB (ESB)

Network note: --download needs internet access to FAOSTAT. Downloads cache to data/raw/faostat/.
"""
import argparse
import os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)                      # run from repo root regardless of caller's cwd

from src import config as cfgmod
from src import pipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=None)
    ap.add_argument("--download", action="store_true", help="download FAOSTAT domains")
    ap.add_argument("--compute", action="store_true", help="compute the balance")
    ap.add_argument("--validate", action="store_true", help="compare N surplus vs FAO ESB")
    args = ap.parse_args()

    cfg = cfgmod.load_config()
    year = args.year or cfg["default_year"]

    if args.download:
        print("downloading FAOSTAT domains ...")
        for d, p in pipeline.download(cfg).items():
            print(f"  {d}: {p}")

    if args.compute:
        print(f"[{year}] computing cropland balance ...")
        res = pipeline.compute_balance(cfg, year)
        for nut, r in res.items():
            print(f"  {nut}: countries={r['n_countries']} "
                  f"mean_surplus={r['mean_surplus']:.1f} kg/ha  -> {r['path']}")

    if args.validate:
        ours, esb = pipeline.validate(cfg, cfg["validation"]["benchmark_year"])
        print(f"  validation: our table {len(ours)} countries; "
              f"FAO ESB {esb['ISO3'].nunique()} countries for "
              f"{cfg['validation']['benchmark_year']} (join on ISO3 to compare components)")


if __name__ == "__main__":
    main()
