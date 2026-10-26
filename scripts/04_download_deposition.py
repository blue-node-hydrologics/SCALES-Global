#!/usr/bin/env python3
"""Download ISIMIP3a N-deposition (NOy + NHx, 0.5deg) into data/raw/deposition/.

Uses the official isimip-client. NOTE: client.datasets(**facets) returns a LIST of
dataset dicts (each with a 'files' list); we iterate it directly.
(2015-2021 are held at 2014 in ISIMIP3a, so the 2014 field is the 2022 NOy proxy; the
true-2022 EMEP Europe overwrite is optional, see bottom.)

Usage:  pip install isimip-client ;  python scripts/04_download_deposition.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import os; os.chdir(ROOT)
from src import config as cfgmod  # noqa: E402

# Facet attempts (first that returns datasets wins). datasets() returns a LIST.
QUERIES = [
    dict(query="ISIMIP3a ndep histsoc"),
    dict(simulation_round="ISIMIP3a", climate_variable="ndep"),
    dict(query="ndep noy nhx"),
    dict(query="nitrogen deposition ISIMIP3a"),
]
DOI_PAGE = "https://data.isimip.org/datasets/?query=ndep"


def _files_of(ds: dict):
    """Yield (name, url) for .nc files in a dataset dict, tolerant of field naming."""
    for f in ds.get("files", []) or []:
        name = f.get("name") or f.get("path") or ""
        url = f.get("file_url") or f.get("path") or ""
        if name.endswith(".nc"):
            yield name, url


def main():
    out_dir = cfgmod.get_path(cfgmod.load_config(), "raw") / "deposition"
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        from isimip_client.client import ISIMIPClient
    except ImportError:
        raise SystemExit("pip install isimip-client  (then rerun)")
    client = ISIMIPClient()

    datasets = []
    for q in QUERIES:
        try:
            ds = client.datasets(**q)          # -> list of dataset dicts
            if ds:
                print(f"  matched {len(ds)} dataset(s) with {q}")
                datasets = ds
                break
            print(f"  0 datasets for {q}")
        except Exception as e:  # noqa: BLE001
            print(f"  query {q} errored: {e}")

    targets = []
    for ds in datasets:
        for name, url in _files_of(ds):
            if ("noy" in name.lower() or "nhx" in name.lower()) and url:
                targets.append((name, url))

    if not targets:
        raise SystemExit(
            "\nCould not auto-locate the NOy/NHx NetCDFs.\n"
            f"  MANUAL: open {DOI_PAGE}\n"
            "  download the ISIMIP3a histsoc files 'ndep-noy_*.nc' (and 'ndep-nhx_*.nc')\n"
            f"  into {out_dir}\n  then run: python scripts/05_build_dep_ox.py")

    for name, url in targets:
        if (out_dir / name).exists():
            print(f"  have {name}"); continue
        print(f"  downloading {name} ...")
        try:
            client.download(url, path=str(out_dir), extract=True)
        except Exception:  # noqa: BLE001
            import requests
            (out_dir / name).write_bytes(requests.get(url, timeout=1200).content)

    print("\nDone. Next: python scripts/05_build_dep_ox.py")
    print("EMEP true-2022 Europe overwrite (optional): download annual DepN 2022 from the EMEP\n"
          "  THREDDS (https://thredds.met.no/thredds/catalog/data/EMEP/) into the same folder.")


if __name__ == "__main__":
    main()
