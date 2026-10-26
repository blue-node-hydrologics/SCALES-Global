#!/usr/bin/env python3
"""Download the external inputs for the treatment-netted human-waste term.

Pulls two open datasets used by the human-waste back-cast:

  1. JONES et al. (2021) wastewater production/collection/treatment/reuse — country + 5 arcmin,
     ~2015 — the modern anchor for the treated-fraction back-cast.
     PANGAEA collection doi:10.1594/PANGAEA.918731.
  2. MADDISON PROJECT DATABASE (GDP per capita, historical) — the back-cast driver
     (with HYDE urban-population fraction, already in the model).

FAOSTAT per-capita PROTEIN supply (the other input) is NOT here — it now downloads with the rest of
FAOSTAT via `python run_pipeline.py --download` (domains FBS + FBSH were added to faostat_api).

Run from the repository root (needs internet access):
    python scripts/42_download_wastewater.py

If a URL 404s (repositories move files), the script prints the landing page so you can grab the file
manually into the target folder shown.
"""
from __future__ import annotations
import sys, zipfile
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

# --- sources (confirm URLs if they 404; landing pages given for manual fallback) ---
JONES_ZIP = "https://store.pangaea.de/Publications/JonesE-etal_2020/Global_wastewater.zip"  # PANGAEA store (3.9 MB)
JONES_LANDING = "https://doi.pangaea.de/10.1594/PANGAEA.918731"
JONES_DIR = RAW / "wastewater"
# Maddison Project Database 2023 (GDP per capita to 2022; Bolt & van Zanden 2025, J. Econ. Surveys).
MADDISON_URL = "https://www.rug.nl/ggdc/historicaldevelopment/maddison/data/mpd2023_web.xlsx"
MADDISON_LANDING = "https://www.rug.nl/ggdc/historicaldevelopment/maddison/releases/maddison-project-database-2023"
GDP_DIR = RAW / "gdp"


def stream(url: str, out: Path, resume=True) -> bool:
    out.parent.mkdir(parents=True, exist_ok=True)
    have = out.stat().st_size if (resume and out.exists()) else 0
    headers = {"Range": f"bytes={have}-"} if have else {}
    try:
        with requests.get(url, headers=headers, stream=True, timeout=120) as r:
            if r.status_code in (403, 404):
                print(f"  [!] {url} -> HTTP {r.status_code}"); return False
            r.raise_for_status()
            done = have
            with open(out, "ab" if have else "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 22):
                    f.write(chunk); done += len(chunk)
                    print(f"\r  {out.name}: {done/1e6:8.1f} MB", end="", flush=True)
            print()
        return True
    except Exception as e:  # noqa: BLE001
        print(f"  [!] download failed for {url}: {e}"); return False


def main():
    print("1) Jones et al. (2021) wastewater (PANGAEA 918731) ->", JONES_DIR)
    zpath = JONES_DIR / "PANGAEA_918731_allfiles.zip"
    if stream(JONES_ZIP, zpath) and zipfile.is_zipfile(zpath):
        with zipfile.ZipFile(zpath) as z:
            z.extractall(JONES_DIR)
        print(f"  unzipped {sum(1 for _ in JONES_DIR.rglob('*') if _.is_file())} files. "
              f"Country-level treatment/collection tables are the ones we need.")
    else:
        print(f"  MANUAL: open {JONES_LANDING} , download the country-level (and gridded) "
              f"collection+treatment files into {JONES_DIR}/ .")

    print("\n2) Maddison Project GDP per capita ->", GDP_DIR)
    mpath = GDP_DIR / "mpd2023_web.xlsx"
    if not stream(MADDISON_URL, mpath):
        print(f"  MANUAL: download the Maddison Project Database 2023 (GDP per capita) from "
              f"{MADDISON_LANDING} into {GDP_DIR}/ (mpd2023_web.xlsx). scripts/45 auto-uses the newest mpd*.xlsx.")

    print("\n3) FAOSTAT protein supply (FBS + FBSH): run  python run_pipeline.py --download  "
          "(domains added to faostat_api).")
    print("done.")


if __name__ == "__main__":
    main()
