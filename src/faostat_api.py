"""FAOSTAT data acquisition (bulk normalized CSVs, with REST-API fallback).

RUN LOCALLY: needs network access to FAOSTAT (bulks-faostat.fao.org /
faostatservices.fao.org). Downloads are cached to data/raw/faostat/ and are
git-ignored, exactly like the US model's NASS downloads.

Domains: RFN (fertilizer), QCL (crops + live animals), EMN (manure), RL (land
use), ESB (Cropland Nutrient Budget benchmark).
"""
from __future__ import annotations
import io
import zipfile
from pathlib import Path
import xml.etree.ElementTree as ET
import pandas as pd

BULK_INDEX = "https://bulks-faostat.fao.org/production/datasets_E.xml"

# Fallback bulk filenames (normalized/long format) if the index lookup fails.
_FALLBACK_BULK = {
    "RFN": "Inputs_FertilizersNutrient_E_All_Data_(Normalized).zip",
    "RFB": "Inputs_FertilizersProduct_E_All_Data_(Normalized).zip",  # fertilizer by PRODUCT (for FracGASF)
    "QCL": "Production_Crops_Livestock_E_All_Data_(Normalized).zip",
    "EMN": "Environment_LivestockManure_E_All_Data_(Normalized).zip",
    "RL":  "Inputs_LandUse_E_All_Data_(Normalized).zip",
    "ESB": "Environment_Cropland_nutrient_budget_E_All_Data_(Normalized).zip",
    "OA":  "Population_E_All_Data_(Normalized).zip",   # annual national population (historical human waste)
    "FBS": "FoodBalanceSheets_E_All_Data_(Normalized).zip",           # per-capita protein supply, 2010-
    "FBSH": "FoodBalanceSheetsHistoric_E_All_Data_(Normalized).zip",  # per-capita protein supply, 1961-2013
    "MK":  "Macro-Statistics_Key_Indicators_E_All_Data_(Normalized).zip",  # GDP per capita (recycling/treatment proxy)
}


def _bulk_url(domain: str) -> str:
    """Resolve the current bulk URL for a domain from the FAOSTAT index, else fallback."""
    import requests
    try:
        xml = requests.get(BULK_INDEX, timeout=120).content
        root = ET.fromstring(xml)
        for ds in root.iter("Dataset"):
            code = (ds.findtext("DatasetCode") or "").upper()
            url = ds.findtext("FileLocation") or ""
            if code == domain.upper() and "Normalized" in url:
                return url
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] bulk index lookup failed ({e}); using fallback filename")
    return f"https://bulks-faostat.fao.org/production/{_FALLBACK_BULK[domain.upper()]}"


def download_domain(domain: str, raw_dir: Path) -> Path:
    """Download a domain's normalized bulk ZIP to data/raw/faostat/. Returns the ZIP path."""
    import requests
    out_dir = Path(raw_dir) / "faostat"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{domain.upper()}_normalized.zip"
    if out.exists():
        return out
    url = _bulk_url(domain)
    print(f"  downloading {domain} <- {url}")
    r = requests.get(url, timeout=900)
    r.raise_for_status()
    out.write_bytes(r.content)
    return out


def read_domain(domain: str, raw_dir: Path) -> pd.DataFrame:
    """Load a downloaded domain ZIP into a long DataFrame (auto-downloads if missing)."""
    zip_path = Path(raw_dir) / "faostat" / f"{domain.upper()}_normalized.zip"
    if not zip_path.exists():
        zip_path = download_domain(domain, raw_dir)
    with zipfile.ZipFile(zip_path) as z:
        csv_name = next(n for n in z.namelist() if n.lower().endswith(".csv"))
        with z.open(csv_name) as f:
            df = pd.read_csv(io.BytesIO(f.read()), encoding="latin-1", low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    return df


def filter_rows(df: pd.DataFrame, *, year=None, element=None, item=None) -> pd.DataFrame:
    """Filter a FAOSTAT long table by Year / Element Code / Item Code (scalars or lists)."""
    def _isin(col, val):
        if val is None:
            return pd.Series(True, index=df.index)
        vals = val if isinstance(val, (list, tuple, set)) else [val]
        return df[col].isin(vals)
    mask = pd.Series(True, index=df.index)
    if year is not None:
        mask &= _isin("Year", year)
    if element is not None:
        mask &= _isin("Element Code", element)
    if item is not None:
        mask &= _isin("Item Code", item)
    return df[mask].copy()


def download_all(cfg, raw_dir: Path) -> dict[str, Path]:
    """Download every configured domain. Returns {domain: zip_path}."""
    domains = sorted(set(cfg["faostat"]["domains"].values()))
    return {d: download_domain(d, raw_dir) for d in domains}
