#!/usr/bin/env python3
"""
# SCALES-Global Technical Validation suite (Table 11 and the validation figure)

Runs every Technical Validation comparison from the current outputs (grids in
`data/outputs/grids/`, reconciled national tables in `data/outputs/`) and reports all of them with ONE
common set of statistics, so that the key results in Table 11 are comparable across tests.

## Common metric

Every paired comparison is made on intensive values, **kg N ha⁻¹ yr⁻¹**, with SCALES-Global on the y
axis and the reference on the x axis, and reports:

- `n`: number of paired units (countries, counties or 0.5° cells);
- `R2`: squared Pearson correlation of the per-hectare values (linear);
- `p`: two-sided p-value of the Pearson correlation (t test with n − 2 degrees of freedom);
  `p_log` likewise for the log10 correlation. RMSE and bias are kept in the CSV columns only;
- `ratio`: ratio of totals, Σ SCALES mass / Σ reference mass over the paired units;
- `R2_log`: squared Pearson correlation of log10 values over pairs where both are positive (reported
  for the county and grid comparisons, whose values span several orders of magnitude).

Tests without a paired spatial reference (human-N literature constraints, mass conservation) are
reported with the ratio only (SCALES / reference, or Σ grid / national).

## Tests (name: reference, scale, year, SCALES quantity)

- `fao_cnb`: FAO Cropland Nutrient Budget (FAOSTAT ESB), country, 2020; cropland surplus
  (FAO-comparable, no human N) per ha cropland vs the ESB N balance per ha; components alongside.
- `batool`: Batool et al. (2022) NUTS0 total soil N surplus, country (Europe), 2019; landscape surplus
  without human N, per ha of land.
- `trend` (optional, not run by default): TREND-Nitrogen v3 (Chang et al. 2026), US county, 2017; total anthropogenic surplus
  (`surplus_full`) per ha of county land vs TREND NSurplus per ha; a human-free comparison
  (NSurplus − Human vs landscape surplus minus the human land terms) is given as a component row.
- `nni`: EPA Next-Generation National Nutrient Inventory, US county, 2017; agricultural landscape
  surplus built identically on both sides from six components (fertilizer + manure + fixation +
  deposition − crop removal − pasture/forage removal), per ha of county area; components alongside.
- `li_nani`: Li et al. (2024) NANI (`NANI_LI_2024/NANI2.tif`, kg N per 5′ cell), global 0.5° grid, 2015.
  NANI is a net balance (net food/feed import = human + livestock consumption − crop and livestock
  production), so it is compared with the SCALES total anthropogenic surplus per ha of land.
- `hani`: HaNi (Tian et al. 2022) cropland fertilizer, global 0.5° grid, 2019 (also 1980 and 2000),
  per ha of grid cell; manure alongside.
- `deposition`: Vishwakarma et al. (2023) cropland deposition, country, 2020. SCALES uses the WL product
  as its cropland deposition rate, so it is compared with the mean of the three other products (AH, AL,
  WH); the share of countries inside the four-product range is also reported.
- `tuholske`: Tuholske et al. (2021) national sewered human N effluent, 2015 (before their 43% retention,
  all land), vs SCALES sewered discharge to surface water, per ha of land; the Tuholske total and per-capita
  comparisons alongside. Reads `data/validation/tuholske2021_national_2015.csv`.
- `human`: global human-N streams, 2000, vs published constraints (ratio only; table, not figure).
- `mass`: mass conservation, Σ grid / national for every country × allocated term (fertilizer, manure,
  crop removal, fixation, human N), 1961, 1980, 2000, 2020, 2023.
- `trajectory`: global cropland surplus per ha, 1961–2022, vs the FAO CNB world series.

## Outputs (`data/outputs/validation/`)

- `table11.csv`: one row per test with the common metrics and the formatted key result;
- `table11_components.csv`: component-level rows (CNB, TREND, NNI, HaNi);
- `pairs_<test>.csv`: the paired values behind each statistic (for the figure or reanalysis);
- `Fig_technical_validation.png` / `.pdf`: one panel per test, same annotation in every panel.

Usage (from the repository root):

    python scripts/55_validation_suite.py                  # all tests, table and figure
    python scripts/55_validation_suite.py --tests fao_cnb batool --no-figure
    python scripts/55_validation_suite.py --figure-only    # redraw from the cached pairs
"""
from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

try:  # macOS default limit (256 open files) is exhausted by the grids and fonts; raise the soft limit
    import resource
    _soft, _hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (min(8192, _hard if _hard > 0 else 8192), _hard))
except Exception:  # noqa: BLE001
    pass

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
from src import config as cfgmod, downscale as ds  # noqa: E402

CFG = cfgmod.load_config()
RAW = Path(cfgmod.get_path(CFG, "raw"))
OUT = Path(cfgmod.get_path(CFG, "outputs"))
VAL = OUT / "validation"
GRID = str(OUT / "grids" / "surplus_N_{}_0p05.nc")
YEARS = {"fao_cnb": 2020, "batool": 2019, "trend": 2017, "nni": 2017, "li_nani": 2015, "hani": 2019,
         "deposition": 2020, "tuholske": 2015, "human": 2000}
ESB_ITEMS = {5087: "fert", 5089: "manure", 5076: "dep", 5077: "crop", 5078: "bnf", 5081: "balance"}


# ----------------------------------------------------------------------------------------------
# common statistics
# ----------------------------------------------------------------------------------------------
def metrics(ours, ref, mass_ours=None, mass_ref=None, log=False):
    """Common metric set on per-ha values (SCALES = ours, reference = ref)."""
    o = np.asarray(ours, float); r = np.asarray(ref, float)
    m = np.isfinite(o) & np.isfinite(r)
    o, r = o[m], r[m]
    out = {"n": int(m.sum())}
    if out["n"] < 3:
        return out | {"R2": np.nan, "p": np.nan, "RMSE": np.nan, "bias": np.nan, "ratio": np.nan,
                      "R2_log": np.nan, "p_log": np.nan}
    out["R2"] = float(np.corrcoef(o, r)[0, 1] ** 2)
    out["p"] = pearson_p(o, r)
    out["RMSE"] = float(np.sqrt(np.mean((o - r) ** 2)))
    out["bias"] = float(np.mean(o - r))
    out["slope"] = float(np.polyfit(r, o, 1)[0])
    if mass_ours is not None and mass_ref is not None:
        mo = np.asarray(mass_ours, float)[m]; mr = np.asarray(mass_ref, float)[m]
        out["ratio"] = float(np.nansum(mo) / np.nansum(mr)) if np.nansum(mr) else np.nan
    else:
        out["ratio"] = float(np.sum(o) / np.sum(r)) if np.sum(r) else np.nan
    pos = (o > 0) & (r > 0)
    out["R2_log"] = (float(np.corrcoef(np.log10(o[pos]), np.log10(r[pos]))[0, 1] ** 2)
                     if (log and pos.sum() > 2) else np.nan)
    out["p_log"] = (pearson_p(np.log10(o[pos]), np.log10(r[pos])) if (log and pos.sum() > 2) else np.nan)
    return out


def pearson_p(x, y):
    """Two-sided p-value of the Pearson correlation of x and y."""
    try:
        from scipy import stats
        return float(stats.pearsonr(x, y)[1])
    except Exception:  # noqa: BLE001  (fallback: normal approximation of the t statistic)
        import math
        n = len(x); r = float(np.corrcoef(x, y)[0, 1])
        if n < 3 or abs(r) >= 1:
            return 0.0 if abs(r) >= 1 else float("nan")
        t = r * math.sqrt((n - 2) / (1 - r * r))
        return float(math.erfc(abs(t) / math.sqrt(2)))


def fmt_p(p):
    if not np.isfinite(p):
        return "p = n/a"
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def key_result(s, with_log=False):
    if not np.isfinite(s.get("R2", np.nan)):
        return ""
    txt = f"R² = {s['R2']:.2f} ({fmt_p(s['p'])})"
    if with_log and np.isfinite(s.get("R2_log", np.nan)):
        txt += f"; R²(log) = {s['R2_log']:.2f} ({fmt_p(s.get('p_log', np.nan))})"
    return txt + f"; ratio = {s['ratio']:.2f}; n = {s['n']:,}"


def save_pairs(name, df):
    VAL.mkdir(parents=True, exist_ok=True)
    df.to_csv(VAL / f"pairs_{name}.csv", index=False)


# ----------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------
def national(year):
    return pd.read_csv(OUT / f"mass_balance_N_{year}.csv")


def open_grid(year):
    import xarray as xr
    d = xr.open_dataset(GRID.format(year))
    if float(d.lat[0]) > float(d.lat[-1]):
        d = d.sortby("lat")
    return d


def cell_area_ha(lat, dlat, dlon):
    return ds._cell_area_ha(np.asarray(lat, "float64"), dlat, dlon) / 1e4      # m2 -> ha


def to_half_degree(values, lat, lon):
    """Sum a regular lat/lon field of MASS (kg per cell) onto the canonical 0.5° grid by cell centre.
    Returns (sum[360, 720], valid_count[360, 720]); NaN cells are ignored."""
    v = np.asarray(values, "float64")
    li = np.clip(np.floor((np.asarray(lat) + 90.0) / 0.5).astype(int), 0, 359)
    lj = np.clip(np.floor((((np.asarray(lon) + 180.0) % 360.0)) / 0.5).astype(int), 0, 719)
    idx = (li[:, None] * 720 + lj[None, :]).ravel()
    vv = v.ravel(); ok = np.isfinite(vv)
    s = np.bincount(idx[ok], weights=vv[ok], minlength=360 * 720).reshape(360, 720)
    c = np.bincount(idx[ok], minlength=360 * 720).reshape(360, 720)
    return s, c


HALF_LAT = -89.75 + 0.5 * np.arange(360)
HALF_HA = cell_area_ha(HALF_LAT, 0.5, 0.5)[:, None] * np.ones((1, 720))


def iso3_from_iso2(codes):
    import logging
    import country_converter as cc
    logging.getLogger("country_converter").setLevel(logging.CRITICAL)
    return cc.CountryConverter().convert(list(codes), src="ISO2", to="ISO3", not_found=None)


def read_esb():
    from src import faostat_api as fao
    return fao.read_domain("ESB", RAW)


# ----------------------------------------------------------------------------------------------
# tests
# ----------------------------------------------------------------------------------------------
def test_fao_cnb(year=YEARS["fao_cnb"]):
    from src import faostat_api as fao, faostat_prep as prep
    e = read_esb()
    e = prep.to_iso3(fao.filter_rows(e, year=year, element=7276), CFG)       # kg N per ha cropland
    piv = (e[e["Item Code"].isin(ESB_ITEMS)]
           .pivot_table(index="ISO3", columns="Item Code", values="Value", aggfunc="mean"))
    piv.columns = ["cnb_" + ESB_ITEMS[c] for c in piv.columns]
    t = national(year).set_index("ISO3").join(piv, how="inner")
    t = t[t["AREA_HA"] > 0]
    pairs = pd.DataFrame({"ISO3": t.index, "ours": t["surplus_cropland_faocomp_kgha"],
                          "ref": t["cnb_balance"], "area_ha": t["AREA_HA"]}).dropna()
    pairs = pairs[(pairs.ours.abs() < 1000) & (pairs.ref.abs() < 1000)]          # drop micro-state artefacts
    save_pairs("fao_cnb", pairs)
    s = metrics(pairs.ours, pairs.ref, pairs.ours * pairs.area_ha, pairs.ref * pairs.area_ha)
    comps = []
    for c, col in [("fertilizer", "fert"), ("manure", "man_crop"), ("crop removal", "crop"),
                   ("biological fixation", "bnf"), ("deposition", "dep")]:
        ref_c = {"fert": "cnb_fert", "man_crop": "cnb_manure", "crop": "cnb_crop", "bnf": "cnb_bnf",
                 "dep": "cnb_dep"}[col]
        if ref_c in t:
            cc = metrics(t[f"{col}_kgha"], t[ref_c], t[f"{col}_kg"], t[ref_c] * t["AREA_HA"])
            comps.append({"test": "fao_cnb", "component": c, **cc})
    return {"test": "fao_cnb", "category": "Internal implementation",
            "reference": "FAO Cropland Nutrient Budget (FAOSTAT ESB)", "scale": "Country",
            "year": year, "product": "Cropland surplus (FAO-comparable)", **s}, comps


def test_batool(year=YEARS["batool"]):
    b = pd.read_csv(RAW / "europe_batool2022" / "N_sur_total_kg_ha_NUTS0_v1.csv")
    b = b[b["Year"] == year].copy()
    b["ISO3"] = iso3_from_iso2(b["NUTS_ID"].str.upper().replace({"UK": "GB", "EL": "GR"}))
    b = b[b["ISO3"].astype(str).str.fullmatch(r"[A-Z]{3}")]
    ref = b.groupby("ISO3")["N_sur_kg_ha_mean"].mean()
    t = national(year).set_index("ISO3")
    la = t["LAND_HA"].where(t["LAND_HA"] > 0)
    human = (t.get("hum_crop_kg", 0) + t.get("hum_sew_land_kg", 0) + t.get("hum_landfill_kg", 0)
             + t.get("food_waste_kg", 0))
    ours = t["surplus_landscape_kgha"] - (human / la).fillna(0.0)
    pairs = pd.DataFrame({"ours": ours, "land_ha": la}).join(ref.rename("ref"), how="inner").dropna()
    pairs = pairs.reset_index().rename(columns={"index": "ISO3"})
    save_pairs("batool", pairs)
    s = metrics(pairs.ours, pairs.ref, pairs.ours * pairs.land_ha, pairs.ref * pairs.land_ha)
    return {"test": "batool", "category": "Independent validation", "reference": "Batool et al. (2022)",
            "scale": "Country (Europe)", "year": year, "product": "Landscape surplus, no human N", **s}, []


NNI_COMP = {  # label: (SCALES grid layers, NNI column, sign in the surplus)
    "fertilizer": (["fert_kg", "fert_grass_kg"], "n_farm_fert_kg", +1),
    "manure": (["man_crop_kg", "man_past_kg"], "n_manure_kg", +1),
    "biological fixation": (["bnf_kg", "bnf_grass_kg"], "n_crop_fix_kg", +1),
    "deposition": (["dep_land_kg"], "n_atm_dep_kg", +1),
    "crop removal": (["crop_kg"], "n_crop_rem_kg", -1),
    "forage removal": (["forage_kg"], "n_pasture_rem_kg", -1),
}


def test_nni(year=YEARS["nni"]):
    import geopandas as gpd
    import rasterio.features
    from affine import Affine
    d = open_grid(year)
    win = d.sel(lat=slice(24, 50), lon=slice(-125, -66))
    lat_, lon_ = win.lat.values, win.lon.values
    res = float(abs(lon_[1] - lon_[0]))
    transform = Affine(res, 0, float(lon_.min() - res / 2), 0, -res, float(lat_.max() + res / 2))
    g = gpd.read_file(RAW / "us_counties" / "cb_2021_us_county_500k.shp").to_crs(4326)
    g["idx"] = np.arange(1, len(g) + 1)
    cidx = rasterio.features.rasterize(((geom, i) for geom, i in zip(g.geometry, g["idx"])),
                                       out_shape=(len(lat_), len(lon_)), transform=transform, fill=0,
                                       dtype="int32").ravel()
    n = len(g) + 1

    def agg(layers):
        a = sum(np.nan_to_num(win[l].values.astype("float64")) for l in layers if l in win)
        return np.bincount(cidx, weights=np.asarray(a)[::-1, :].ravel(), minlength=n)[g["idx"].values]

    sc = pd.DataFrame({"GEOID": g["GEOID"].astype(str).str.zfill(5)})
    for lab, (layers, _, _) in NNI_COMP.items():
        sc[lab] = agg(layers)
    nni = pd.read_csv(RAW / "NNI_brehob_et_al" / "CountyNNIkg.csv")
    nni = nni[nni["year"] == year].copy()
    nni["GEOID"] = nni["FIPS"].astype("Int64").astype(str).str.zfill(5)
    keep = ["GEOID", "area_sqkm"] + [c for _, c, _ in NNI_COMP.values()]
    df = sc.merge(nni[keep], on="GEOID", how="inner")
    df["area_ha"] = df["area_sqkm"] * 100.0
    df = df[df["area_ha"] > 0]
    ours_m = sum(sgn * df[lab] for lab, (_, _, sgn) in NNI_COMP.items())
    ref_m = sum(sgn * df[col].fillna(0.0) for _, col, sgn in NNI_COMP.values())
    pairs = pd.DataFrame({"GEOID": df["GEOID"], "ours": ours_m / df["area_ha"], "ref": ref_m / df["area_ha"],
                          "area_ha": df["area_ha"]})
    for lab, (_, col, _) in NNI_COMP.items():
        pairs[f"{lab}_ours"] = df[lab] / df["area_ha"]
        pairs[f"{lab}_ref"] = df[col] / df["area_ha"]
    save_pairs("nni", pairs)
    s = metrics(pairs.ours, pairs.ref, ours_m, ref_m, log=True)
    comps = []
    for lab, (_, col, _) in NNI_COMP.items():
        cc = metrics(pairs[f"{lab}_ours"], pairs[f"{lab}_ref"], df[lab], df[col], log=True)
        comps.append({"test": "nni", "component": lab, **cc})
    return {"test": "nni", "category": "Independent validation",
            "reference": "EPA Next-Generation National Nutrient Inventory", "scale": "County (US)",
            "year": year, "product": "Agricultural landscape surplus (six components)", **s}, comps


def test_li_nani(year=YEARS["li_nani"]):
    """Li et al. (2024) NANI = N_im + N_fer + N_dep + N_fix with N_im = N_hc + N_lc - N_lp - N_cp, i.e. a NET
    balance: human and livestock consumption are added and crop and livestock production subtracted. It is
    therefore compared with the SCALES-Global total anthropogenic surplus (surplus_full) per ha of land.
    NANI2.tif is kg N per 5' cell (global ~197 Tg)."""
    import rasterio
    d = open_grid(year)
    lat, lon = d.lat.values, d.lon.values
    ca = cell_area_ha(lat, float(abs(lat[1] - lat[0])), float(abs(lon[1] - lon[0])))[:, None]
    mass = d["surplus_full_kgha"].values.astype("float64") * ca                # kg per 0.05° cell
    o_sum, o_cnt = to_half_degree(mass, lat, lon)
    tif = RAW / "NANI_LI_2024" / "NANI2.tif"
    with rasterio.open(tif) as src:
        a = src.read(1).astype("float64"); nod = src.nodata
        a = np.where(np.isfinite(a) & (a != nod) & (a > -1e30), a, np.nan)    # kg N per 5' cell
        res = src.res[0]
        llat = src.bounds.top - (np.arange(src.height) + 0.5) * res
        llon = src.bounds.left + (np.arange(src.width) + 0.5) * res
    l_sum, l_cnt = to_half_degree(a, llat, llon)
    ok = (o_cnt > 0) & (l_cnt > 0)
    pairs = pd.DataFrame({"ours": (o_sum / HALF_HA)[ok], "ref": (l_sum / HALF_HA)[ok],
                          "ours_kg": o_sum[ok], "ref_kg": l_sum[ok], "cell_ha": HALF_HA[ok]})
    save_pairs("li_nani", pairs)
    print(f"  [li_nani] Li NANI {np.nansum(l_sum)/1e9:.1f} Tg N, SCALES total surplus {np.nansum(o_sum)/1e9:.1f} Tg")
    s = metrics(pairs.ours, pairs.ref, pairs.ours_kg, pairs.ref_kg, log=True)
    return {"test": "li_nani", "category": "Independent validation", "reference": "Li et al. (2024) NANI",
            "scale": "Global 0.5° grid", "year": year, "product": "Total anthropogenic surplus", **s}, []


def _county_sums(d, variables):
    """Sum gridded kg variables (0.05°) to US counties (2021 Census boundaries) -> DataFrame by GEOID."""
    import geopandas as gpd
    import rasterio.features
    from affine import Affine
    w = d.sel(lat=slice(24, 50), lon=slice(-125, -66))
    lat, lon = w.lat.values, w.lon.values
    res = float(abs(lon[1] - lon[0]))
    tr = Affine(res, 0, float(lon.min() - res / 2), 0, -res, float(lat.max() + res / 2))
    g = gpd.read_file(RAW / "us_counties" / "cb_2021_us_county_500k.shp").to_crs(4326)
    g["idx"] = np.arange(1, len(g) + 1)
    cid = rasterio.features.rasterize(((geom, i) for geom, i in zip(g.geometry, g["idx"])),
                                      out_shape=(len(lat), len(lon)), transform=tr, fill=0, dtype="int32")
    flat = cid.ravel(); n = len(g) + 1
    out = pd.DataFrame({"GEOID": g["GEOID"].astype(str).str.zfill(5),
                        "area_ha": g["ALAND"].astype(float) / 1e4})
    for name, v in variables.items():
        arr = np.nan_to_num(v.sel(lat=w.lat, lon=w.lon).values.astype("float64"))[::-1, :]
        out[name] = np.bincount(flat, weights=arr.ravel(), minlength=n)[g["idx"].values]
    return out


def _trend_year(name, year):
    t = pd.read_csv(RAW / "trend_v3" / "TREND-nitrogen" / f"{name}.txt")
    t["GEOID"] = t["GEOID"].astype(float).astype(int).map(lambda v: f"{v:05d}")
    return t[["GEOID", f"y{year}"]].rename(columns={f"y{year}": name})


def test_trend(year=YEARS["trend"]):
    """TREND-Nitrogen v3 (county NSurplus, kg N per ha of county area, NANI convention incl. human and
    livestock N) vs the SCALES-Global total anthropogenic surplus (surplus_full) per ha of county land.
    A second, human-free comparison (NSurplus - Human vs landscape surplus minus the human land terms)
    is reported as a component row because the two models treat human N differently."""
    d = open_grid(year)
    lat = d.lat.values; lon = d.lon.values
    ca = cell_area_ha(lat, float(abs(lat[1] - lat[0])), float(abs(lon[1] - lon[0])))[:, None]
    import xarray as xr
    da = lambda a: xr.DataArray(a, coords={"lat": lat, "lon": lon}, dims=("lat", "lon"))
    hum = sum(np.nan_to_num(d[v].values.astype("float64"))
              for v in ("hum_crop_kg", "hum_sew_land_kg", "food_waste_kg") if v in d)
    full = np.nan_to_num(d["surplus_full_kgha"].values.astype("float64")) * ca
    land = np.nan_to_num(d["surplus_landscape_kgha"].values.astype("float64")) * ca - hum
    c = _county_sums(d, {"full_kg": da(full), "nonhuman_kg": da(land)})
    t = _trend_year("NSurplus", year).merge(_trend_year("Human", year), on="GEOID")
    a = pd.read_csv(RAW / "trend_v3" / "TREND-nitrogen" / "NSurplus.txt", usecols=["GEOID", "AREA (HA)"])
    a["GEOID"] = a["GEOID"].astype(float).astype(int).map(lambda v: f"{v:05d}")
    m = c.merge(t, on="GEOID").merge(a, on="GEOID")
    m = m[(m.area_ha > 0) & np.isfinite(m.NSurplus)]
    pairs = pd.DataFrame({"GEOID": m.GEOID, "ours": m.full_kg / m.area_ha, "ref": m.NSurplus,
                          "ours_kg": m.full_kg, "ref_kg": m.NSurplus * m["AREA (HA)"],
                          "ours_nonhuman": m.nonhuman_kg / m.area_ha, "ref_nonhuman": m.NSurplus - m.Human})
    save_pairs("trend", pairs)
    s = metrics(pairs.ours, pairs.ref, pairs.ours_kg, pairs.ref_kg, log=True)
    nh = metrics(pairs.ours_nonhuman, pairs.ref_nonhuman,
                 pairs.ours_nonhuman * m.area_ha.values, pairs.ref_nonhuman * m["AREA (HA)"].values, log=True)
    comps = [{"test": "trend", "component": "total anthropogenic surplus", **s},
             {"test": "trend", "component": "surplus without human N (both sides)", **nh}]
    return {"test": "trend", "category": "Independent validation", "reference": "TREND-Nitrogen v3",
            "scale": "County (US)", "year": year, "product": "Total anthropogenic surplus", **s}, comps


def _hani_layer(name, year):
    import xarray as xr
    h = xr.open_dataset(RAW / "hani" / f"{name}.nc", decode_times=False)
    a = h[name].isel(time=year - 1925).values.astype("float64") / 1e3        # g N -> kg N per cell
    lat = h["lat"].values; lon = h["lon"].values
    lon = ((lon + 180.0) % 360.0) - 180.0
    return a, lat, lon


def test_hani(year=YEARS["hani"], extra_years=(1980, 2000)):
    comps = []; main = None
    for y in (year,) + tuple(extra_years):
        d = open_grid(y)
        lat, lon = d.lat.values, d.lon.values
        f_o, _ = to_half_degree(np.nan_to_num(d["fert_kg"].values.astype("float64")), lat, lon)
        m_o, _ = to_half_degree(np.nan_to_num(d["man_crop_kg"].values.astype("float64")), lat, lon)
        a1, hlat, hlon = _hani_layer("nfer_crop_nh4", y); a2, _, _ = _hani_layer("nfer_crop_no3", y)
        f_h, _ = to_half_degree(np.nan_to_num(a1) + np.nan_to_num(a2), hlat, hlon)
        am, _, _ = _hani_layer("nmanure_app_crop", y)
        m_h, _ = to_half_degree(np.nan_to_num(am), hlat, hlon)
        for lab, o, h in [("fertilizer", f_o, f_h), ("manure", m_o, m_h)]:
            ok = (o > 0) | (h > 0)
            p = pd.DataFrame({"ours": (o / HALF_HA)[ok], "ref": (h / HALF_HA)[ok], "ours_kg": o[ok],
                              "ref_kg": h[ok]})
            s = metrics(p.ours, p.ref, p.ours_kg, p.ref_kg, log=True)
            comps.append({"test": "hani", "component": f"{lab} {y}", **s})
            if lab == "fertilizer" and y == year:
                save_pairs("hani", p); main = s
    return {"test": "hani", "category": "Component validation", "reference": "HaNi (Tian et al. 2022)",
            "scale": "Global 0.5° grid", "year": year, "product": "Cropland fertilizer", **main}, comps


def _vish_product(path, year):
    import logging
    import country_converter as cc
    logging.getLogger("country_converter").setLevel(logging.CRITICAL)
    df = pd.read_excel(path)
    ycols = [c for c in df.columns if str(c).strip().isdigit() and 1900 < int(str(c)) < 2100]
    ccol = next(c for c in df.columns if c not in ycols)
    ycol = next((c for c in ycols if int(str(c)) == year), None)
    if ycol is None:
        return None
    s = pd.DataFrame({"ISO3": cc.convert(df[ccol].astype(str).tolist(), to="ISO3", not_found=None),
                      "v": pd.to_numeric(df[ycol], errors="coerce")})
    s = s[s["ISO3"].astype(str).str.fullmatch(r"[A-Z]{3}")]
    return s.groupby("ISO3")["v"].mean()


def deposition_components(pairs):
    """SCALES cropland deposition (= the WL product by construction) against WL and each alternative
    Vishwakarma product separately; the mean-of-products row in Table 11 blends a near-identical product
    (WH) with two different ones (AH, AL), so the per-product rows are what Table 11 reports."""
    out = []
    for tag in ("WL", "WH", "AL", "AH"):
        if tag in pairs:
            k = pairs.dropna(subset=[tag, "ours"])
            out.append({"test": "deposition", "component": f"vs {tag}",
                        **metrics(k["ours"], k[tag], k["ours"] * k["area_ha"], k[tag] * k["area_ha"])})
    return out


def test_deposition(year=YEARS["deposition"]):
    vdir = RAW / "vishwakarma2023"
    prods = {}
    for tag in ("AH", "AL", "WH", "WL"):
        f = sorted(vdir.glob(f"*_{tag}*.xlsx"))
        if f:
            p = _vish_product(f[0], year)
            if p is not None:
                prods[tag] = p
    w = pd.DataFrame(prods)
    others = [c for c in ("AH", "AL", "WH") if c in w]
    t = national(year).set_index("ISO3")
    # SCALES cropland deposition rate (kg N per ha cropland) = the WL rate, in the grid and nationally.
    ours = (t["dep_kg"] / t["AREA_HA"].where(t["AREA_HA"] > 0)).rename("ours")
    j = w.join(ours, how="inner").dropna(subset=["ours"] + others)
    j["ref"] = j[others].mean(axis=1)
    lo = j[[c for c in ("AH", "AL", "WH", "WL") if c in j]].min(axis=1)
    hi = j[[c for c in ("AH", "AL", "WH", "WL") if c in j]].max(axis=1)
    j["inside_range"] = (j["ours"] >= lo) & (j["ours"] <= hi)
    pairs = j.reset_index().rename(columns={"index": "ISO3"})
    pairs["area_ha"] = pairs["ISO3"].map(t["AREA_HA"])
    save_pairs("deposition", pairs)
    s = metrics(pairs.ours, pairs.ref, pairs.ours * pairs.area_ha, pairs.ref * pairs.area_ha)
    s["inside_range_pct"] = float(100 * pairs["inside_range"].mean())
    comps = deposition_components(pairs)
    return {"test": "deposition", "category": "Component validation",
            "reference": "Vishwakarma et al. (2023), mean of AH, AL, WH", "scale": "Country",
            "year": year, "product": "Cropland deposition", **s}, comps


HUMAN_BENCH = [  # (label, SCALES column in timeseries, benchmark Tg, source)
    ("Sewered human N to surface water", "hum_sew_water_Tg", 6.4, "Van Drecht et al. (2009)"),
    ("Sewered human N to surface water", "hum_sew_water_Tg", 7.7, "Morée et al. (2013)"),
]
# (the Morée et al. (2013) urban-only figure for excreta recycled to cropland is not comparable with the
#  SCALES urban + rural night-soil and biosolids term, so it is no longer used)


def test_human(year=YEARS["human"]):
    ts = pd.read_csv(OUT / "timeseries_N_global.csv").set_index("year")
    rows = []
    for lab, col, ref, src in HUMAN_BENCH:
        v = float(ts.loc[year, col]) if col in ts else np.nan
        rows.append({"test": "human", "component": f"{lab} vs {src}", "scales_Tg": v, "ref_Tg": ref,
                     "ratio": v / ref})
    fw = float(ts.loc[2022, "food_waste_Tg"]) if 2022 in ts.index else np.nan
    rows.append({"test": "human", "component": "Consumer food waste N (2022) vs UNEP (2024) 631 Mt x 0.6% N",
                 "scales_Tg": fw, "ref_Tg": 631 * 0.006, "ratio": fw / (631 * 0.006)})
    pairs = pd.DataFrame(rows); save_pairs("human", pairs)
    rr = pairs["ratio"].to_numpy()
    s = {"n": len(rows), "ratio": float(np.nanmedian(rr)), "ratio_min": float(np.nanmin(rr)),
         "ratio_max": float(np.nanmax(rr))}
    return {"test": "human", "category": "Component validation", "reference": "Human-N literature constraints",
            "scale": "Global", "year": year, "product": "Human N pathways", **s}, rows


TUHOLSKE = ROOT / "data" / "validation" / "tuholske2021_national_2015.csv"


def test_tuholske(year=YEARS["tuholske"]):
    """Tuholske et al. (2021, PLOS ONE 16:e0258898) national sewered human N effluent, 2015, before their
    uniform 43% freshwater retention, all land (their 1 km effluent rasters, KNB doi:10.5063/F76B09, summed
    by country; data/validation/tuholske2021_national_2015.csv). Compared with the SCALES sewered discharge to surface
    water (hum_sew_water), per ha of land. Components: the Tuholske total (sewered + septic + open within
    1 km of water) and both comparisons per capita."""
    c = pd.read_csv(TUHOLSKE)
    land = national(year).set_index("ISO3")["LAND_HA"]
    c["land_ha"] = c["ISO3"].map(land)
    c = c[(c.land_ha > 0) & (c.hum_sew_water_kg_all > 0) & (c.tuh_treated_kg_all > 0)]
    pairs = pd.DataFrame({"ISO3": c.ISO3, "ours": c.hum_sew_water_kg_all / c.land_ha,
                          "ref": c.tuh_treated_kg_all / c.land_ha, "ours_kg": c.hum_sew_water_kg_all,
                          "ref_kg": c.tuh_treated_kg_all, "ref_total_kg": c.tuh_total_kg_all,
                          "population": c.population, "land_ha": c.land_ha})
    save_pairs("tuholske", pairs)
    s = metrics(pairs.ours, pairs.ref, pairs.ours_kg, pairs.ref_kg, log=True)
    comps = [{"test": "tuholske", "component": "sewered, per ha of land", **s}]
    t = pairs[pairs.ref_total_kg > 0]
    comps.append({"test": "tuholske", "component": "vs Tuholske total (sewered + septic + open), per ha",
                  **metrics(t.ours, t.ref_total_kg / t.land_ha, t.ours_kg, t.ref_total_kg, log=True)})
    comps.append({"test": "tuholske", "component": "sewered, per capita",
                  **metrics(pairs.ours_kg / pairs.population, pairs.ref_kg / pairs.population,
                            pairs.ours_kg, pairs.ref_kg, log=True)})
    return {"test": "tuholske", "category": "Component validation", "reference": "Tuholske et al. (2021)",
            "scale": "Country", "year": year, "product": "Sewered human N to surface water", **s}, comps


# terms whose national totals are ALLOCATED to the grid (Eqs. 1-2). Deposition is a rate times area
# (cropland: WL x HYDE cropland in the grid, WL x FAOSTAT cropland nationally) and forage/all-land
# deposition are national sums of the grid by definition, so they are not mass-conservation tests.
MASS_TERMS = ["fert", "man_crop", "man_past", "crop", "bnf", "fert_grass", "bnf_grass", "hum_ns", "hum_bio",
              "food_waste"]


def test_mass(years=(1961, 1980, 2000, 2020, 2023)):
    cache = ds.load_weight_cache(CFG)
    mask, idx2iso = cache["mask"], cache["idx2iso"]
    flat = mask.ravel(); ok = flat >= 0; k = np.where(ok, flat, 0); nk = int(mask.max()) + 1
    rows = []
    for y in years:
        d = open_grid(y); t = national(y).set_index("ISO3")
        for term in MASS_TERMS:
            if f"{term}_kg" not in d or f"{term}_kg" not in t:
                continue
            v = np.nan_to_num(d[f"{term}_kg"].values.astype("float64")).ravel()
            gs = np.bincount(k, weights=np.where(ok, v, 0.0), minlength=nk)
            for i, iso in idx2iso.items():
                nat = float(t[f"{term}_kg"].get(iso, 0.0))
                if nat > 1e6:                                         # > 1 Gg N
                    rows.append({"year": y, "term": term, "ISO3": iso, "grid_kg": gs[int(i)], "nat_kg": nat,
                                 "ratio": gs[int(i)] / nat})
    p = pd.DataFrame(rows); save_pairs("mass", p)
    r = p["ratio"].to_numpy()
    s = {"n": len(p), "ratio": float(np.median(r)), "within_1pct": float(100 * np.mean(np.abs(r - 1) <= 0.01)),
         "within_5pct": float(100 * np.mean(np.abs(r - 1) <= 0.05)),
         "global_ratio": float(p["grid_kg"].sum() / p["nat_kg"].sum()),
         "R2": float(np.corrcoef(p["grid_kg"], p["nat_kg"])[0, 1] ** 2), "p": pearson_p(p["grid_kg"], p["nat_kg"])}
    return {"test": "mass", "category": "Numerical integrity", "reference": "Mass conservation",
            "scale": "Country × term", "year": "1961, 1980, 2000, 2020, 2023", "product": "All gridded terms",
            **s}, []


def test_trajectory():
    e = read_esb()
    w = e[(e["Area"].astype(str) == "World") & (e["Element Code"] == 7276) & (e["Item Code"] == 5081)]
    ref = w.groupby("Year")["Value"].mean().rename("ref")
    ts = pd.read_csv(OUT / "timeseries_N_global.csv").set_index("year")
    j = ts[["surplus_cropland_faocomp_kgha", "cropland_Mha"]].join(ref, how="inner").dropna()
    pairs = pd.DataFrame({"year": j.index, "ours": j["surplus_cropland_faocomp_kgha"], "ref": j["ref"]})
    save_pairs("trajectory", pairs)
    s = metrics(pairs.ours, pairs.ref)
    return {"test": "trajectory", "category": "Whole-record plausibility",
            "reference": "FAO Cropland Nutrient Budget, world series", "scale": "Global",
            "year": f"{int(pairs.year.min())}–{int(pairs.year.max())}",
            "product": "Global cropland surplus per ha (FAO-comparable)", **s}, []


TESTS = {"fao_cnb": test_fao_cnb, "batool": test_batool, "trend": test_trend, "nni": test_nni, "li_nani": test_li_nani,
         "hani": test_hani, "deposition": test_deposition, "tuholske": test_tuholske, "human": test_human, "mass": test_mass,
         "trajectory": test_trajectory}


OPTIONAL = {"trend"}   # run only on request (--tests trend); not part of Table 11 or the figure


TABLE_EXCLUDE = {"human"}   # ratio-only literature constraints: reported in the text and components, not Table 11
DEP_TABLE = ("WH", "AL", "AH")


def format_key(row, comps=None):
    """Table 11 key result: R² and its p-value."""
    t = row["test"]
    if t == "deposition" and comps is not None and len(comps):
        c = comps[(comps["test"] == "deposition")].set_index("component")
        parts = [f"{c.loc['vs ' + g, 'R2']:.2f} vs {g}" for g in DEP_TABLE if f"vs {g}" in c.index]
        ps = [c.loc["vs " + g, "p"] for g in DEP_TABLE if f"vs {g}" in c.index and "p" in c]
        if parts:
            return "R² = " + ", ".join(parts) + (" (all p < 0.001)" if ps and max(ps) < 0.001 else "")
    if not np.isfinite(row.get("R2", np.nan)):
        return ""
    return f"R² = {row['R2']:.2f} ({fmt_p(row.get('p', np.nan))})"


def reformat_tables():
    """Rewrite table11.csv / table11_components.csv from the cached pairs without rerunning the tests:
    adds R² and p for mass conservation and the per-product deposition rows, drops ratio-only and
    optional rows, and rewrites every key result as R² and p."""
    t = pd.read_csv(VAL / "table11.csv")
    c = pd.read_csv(VAL / "table11_components.csv") if (VAL / "table11_components.csv").exists() else pd.DataFrame()
    if len(c):
        c = c[~c["test"].isin(OPTIONAL | {"deposition"})]
    if (VAL / "pairs_mass.csv").exists():
        m = pd.read_csv(VAL / "pairs_mass.csv")
        t.loc[t.test == "mass", "R2"] = float(np.corrcoef(m.grid_kg, m.nat_kg)[0, 1] ** 2)
        t.loc[t.test == "mass", "p"] = pearson_p(m.grid_kg, m.nat_kg)
    if (VAL / "pairs_deposition.csv").exists():
        c = pd.concat([c, pd.DataFrame(deposition_components(pd.read_csv(VAL / "pairs_deposition.csv")))],
                      ignore_index=True)
    if len(c) and "p" not in c:
        c["p"] = np.nan
    t = t[~t["test"].isin(OPTIONAL | TABLE_EXCLUDE)].copy()
    t["key_result"] = [format_key(r, c) for _, r in t.iterrows()]
    t.to_csv(VAL / "table11.csv", index=False); c.to_csv(VAL / "table11_components.csv", index=False)
    print(t[["test", "key_result"]].to_string(index=False))


# ----------------------------------------------------------------------------------------------
# figure
# ----------------------------------------------------------------------------------------------
def figure(table):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"], "font.size": 9})
    import logging; logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    BOX = dict(boxstyle="round,pad=0.35", fc="white", ec="#999999", alpha=0.95)
    PT = "#2f6db5"; ONE = "#333333"
    T = {r["test"]: r for _, r in table.iterrows()}
    panels = [("fao_cnb", "FAO Cropland Nutrient Budget, countries", "FAO CNB cropland N surplus",
               "SCALES-Global cropland N surplus", False),
              ("batool", "Batool et al., European countries", "Batool soil N surplus",
               "SCALES-Global landscape N surplus", False),
              ("nni", "EPA NNI, US counties", "NNI agricultural N surplus",
               "SCALES-Global agricultural N surplus", False),
              ("li_nani", "Li et al. NANI, 0.5° cells", "Li NANI", "SCALES-Global total N surplus", True),
              ("hani", "HaNi cropland fertilizer, 0.5° cells", "HaNi fertilizer N", "SCALES-Global fertilizer N",
               True),
              ("deposition", "Vishwakarma et al. deposition, countries", "Mean of AH, AL, WH products",
               "SCALES-Global cropland deposition", False),
              ("tuholske", "Tuholske et al. sewered human N, countries", "Tuholske sewered effluent N",
               "SCALES-Global sewered N to water", True)]
    fig, axes = plt.subplots(3, 3, figsize=(13.5, 13))
    letters = "abcdefghi"
    for k, (name, title, xl, yl, logax) in enumerate(panels):
        ax = axes.ravel()[k]
        f = VAL / f"pairs_{name}.csv"
        if not f.exists() or name not in T:
            ax.text(0.5, 0.5, "not run", ha="center", transform=ax.transAxes); continue
        p = pd.read_csv(f)
        x = p["ref"].to_numpy(float); y = p["ours"].to_numpy(float)
        if logax:
            m = (x > 0) & (y > 0); x, y = x[m], y[m]
            ax.set_xscale("log"); ax.set_yscale("log")
            lo = np.nanpercentile(np.r_[x, y], 0.5); hi = np.nanpercentile(np.r_[x, y], 99.9)
        else:
            lo = min(np.nanpercentile(np.r_[x, y], 0.5), 0); hi = np.nanpercentile(np.r_[x, y], 99.5) * 1.05
        ax.scatter(x, y, s=4 if len(x) > 2000 else 18, alpha=0.25 if len(x) > 2000 else 0.75, color=PT,
                   edgecolors="none")
        ax.plot([lo, hi], [lo, hi], "-", color=ONE, lw=1)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        r = T[name]
        txt = (f"R² = {r['R2']:.2f}, {fmt_p(r['p'])}\n"
               f"n = {int(r['n']):,}")
        if np.isfinite(r.get("R2_log", np.nan)):
            pass
        ax.text(0.04, 0.96, txt, transform=ax.transAxes, va="top", fontsize=8.5, bbox=BOX)
        ax.set_title(f"{letters[k]}   {title}", loc="left", fontsize=10.5, fontweight="bold")
        ax.set_xlabel(f"{xl} (kg N ha$^{{-1}}$ yr$^{{-1}}$)"); ax.set_ylabel(f"{yl} (kg N ha$^{{-1}}$ yr$^{{-1}}$)")
    # f: deposition, SCALES (= WL) against each alternative Vishwakarma product
    f = VAL / "pairs_deposition.csv"
    if f.exists():
        ax = axes.ravel()[5]; ax.cla()
        dp = pd.read_csv(f)
        cols = {"WH": "#2f6db5", "AL": "#c0562c", "AH": "#3f9b78"}
        lines = []
        for g, colr in cols.items():
            k = dp.dropna(subset=[g, "ours"])
            ax.scatter(k[g], k["ours"], s=14, alpha=0.75, color=colr, edgecolors="none", label=g)
            st = metrics(k["ours"], k[g])
            lines.append(f"{g}: R² = {st['R2']:.2f}, {fmt_p(st['p'])}")
        hi = float(np.nanmax(dp[["ours", "WH", "AL", "AH"]].to_numpy())) * 1.05
        ax.plot([0, hi], [0, hi], "-", color=ONE, lw=1); ax.set_xlim(0, hi); ax.set_ylim(0, hi)
        ax.legend(frameon=False, loc="lower right", title="Product", fontsize=8, title_fontsize=8)
        ax.text(0.04, 0.96, "\n".join(lines) + f"\nn = {len(dp):,}", transform=ax.transAxes, va="top",
                fontsize=8.5, bbox=BOX)
        ax.set_title("f   Vishwakarma et al. deposition, countries", loc="left", fontsize=10.5, fontweight="bold")
        ax.set_xlabel("Alternative product cropland deposition (kg N ha$^{-1}$ yr$^{-1}$)")
        ax.set_ylabel("SCALES-Global cropland deposition (kg N ha$^{-1}$ yr$^{-1}$)")
    # h: mass conservation
    ax = axes.ravel()[7]
    f = VAL / "pairs_mass.csv"
    if f.exists() and "mass" in T:
        mm = pd.read_csv(f)["ratio"].to_numpy()
        ax.hist(np.clip(mm, 0.9, 1.1), bins=np.arange(0.9, 1.1001, 0.005), color=PT, edgecolor="white")
        ax.axvline(1.0, color=ONE, lw=1)
        r = T["mass"]
        r2t = (f"R² = {r['R2']:.3f}, {fmt_p(r['p'])}\n" if np.isfinite(r.get("R2", np.nan)) else "")
        ax.text(0.04, 0.96, f"{r2t}median ratio = {r['ratio']:.3f}\n{r['within_1pct']:.1f}% within ±1%\n"
                f"n = {int(r['n']):,}", transform=ax.transAxes,
                va="top", fontsize=8.5, bbox=BOX)
        ax.set_xlabel("Σ grid / national (country × term)"); ax.set_ylabel("Count")
    ax.set_title("h   Mass conservation", loc="left", fontsize=10.5, fontweight="bold")
    # i: trajectory
    ax = axes.ravel()[8]
    f = VAL / "pairs_trajectory.csv"
    if f.exists() and "trajectory" in T:
        tr = pd.read_csv(f)
        ax.plot(tr["year"], tr["ref"], "-", color=ONE, lw=1.5, label="FAO CNB, world")
        ax.plot(tr["year"], tr["ours"], "-", color=PT, lw=1.8, label="SCALES-Global")
        ax.legend(frameon=False, loc="lower right")
        r = T["trajectory"]
        ax.text(0.04, 0.96, f"R² = {r['R2']:.2f}, {fmt_p(r['p'])}\nn = {int(r['n'])} years", transform=ax.transAxes, va="top",
                fontsize=8.5, bbox=BOX)
        ax.set_xlabel("Year"); ax.set_ylabel("Cropland N surplus (kg N ha$^{-1}$ yr$^{-1}$)")
    yrs = (f", {int(tr['year'].min())}–{int(tr['year'].max())}" if f.exists() and "trajectory" in T else "")
    ax.set_title(f"i   Global cropland surplus{yrs}", loc="left", fontsize=10.5, fontweight="bold")
    for a in axes.ravel():
        a.grid(False)
        for s_ in a.spines.values():
            s_.set_visible(True); s_.set_linewidth(0.9)
    fig.tight_layout(pad=1.4)
    for ext in ("png", "pdf"):
        fig.savefig(VAL / f"Fig_technical_validation.{ext}", dpi=250, facecolor="white")
    print(f"wrote {VAL / 'Fig_technical_validation.png'}")


# ----------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tests", nargs="+", default=[t for t in TESTS if t not in OPTIONAL], choices=list(TESTS))
    ap.add_argument("--no-figure", action="store_true")
    ap.add_argument("--figure-only", action="store_true")
    ap.add_argument("--reformat", action="store_true", help="rewrite Table 11 from the cached pairs (no tests)")
    args = ap.parse_args()
    VAL.mkdir(parents=True, exist_ok=True)
    if args.reformat:
        reformat_tables()
        if not args.no_figure:
            figure(pd.read_csv(VAL / "table11.csv"))
        return
    tpath, cpath = VAL / "table11.csv", VAL / "table11_components.csv"
    old = pd.read_csv(tpath) if tpath.exists() else pd.DataFrame()
    oldc = pd.read_csv(cpath) if cpath.exists() else pd.DataFrame()
    if not args.figure_only:
        rows, comps = [], []
        for name in args.tests:
            print(f"[{name}] running ...", flush=True)
            try:
                row, cc = TESTS[name]()
                row["key_result"] = format_key(row, pd.DataFrame(cc) if cc else None)
                rows.append(row); comps.extend(cc)
                print(f"[{name}] {row['key_result']}", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"[{name}] FAILED: {e}")
                traceback.print_exc()
        new = pd.DataFrame(rows)
        if len(old):
            old = old[~old["test"].isin(new["test"])] if len(new) else old
        table = pd.concat([old, new], ignore_index=True)
        table = table[~table["test"].isin((OPTIONAL - set(args.tests)) | TABLE_EXCLUDE)]   # stale TREND row; ratio-only rows
        order = {k: i for i, k in enumerate(TESTS)}
        table = table.sort_values("test", key=lambda s: s.map(order)).reset_index(drop=True)
        table.to_csv(tpath, index=False)
        newc = pd.DataFrame(comps)
        if len(oldc) and len(newc):
            oldc = oldc[~oldc["test"].isin(newc["test"])]
        pd.concat([oldc, newc], ignore_index=True).to_csv(cpath, index=False)
        print(f"wrote {tpath} and {cpath}")
    if not args.no_figure:
        figure(pd.read_csv(tpath))


if __name__ == "__main__":
    main()
