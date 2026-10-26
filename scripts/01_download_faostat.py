#!/usr/bin/env python3
"""Download all FAOSTAT activity domains (RFN, QCL, EMN, RL, ESB) as normalized bulk
CSVs into data/raw/faostat/. Idempotent (skips files already present).

Usage:  python scripts/01_download_faostat.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import os; os.chdir(ROOT)                  # run from repo root regardless of caller's cwd
from src import config as cfgmod          # noqa: E402
from src import faostat_api               # noqa: E402


def main():
    cfg = cfgmod.load_config()
    raw = cfgmod.get_path(cfg, "raw")
    print("Downloading FAOSTAT bulk domains -> data/raw/faostat/")
    for domain, path in faostat_api.download_all(cfg, raw).items():
        mb = path.stat().st_size / 1e6
        print(f"  {domain:4s}  {mb:7.1f} MB  {path.name}")
    print("\nDone. Next: 02_download_crop_coeffs.py, 03_download_population.py, "
          "04_download_deposition.py")


if __name__ == "__main__":
    main()
