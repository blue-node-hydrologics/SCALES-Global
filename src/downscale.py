"""Tier-B downscaling engine: grid the validated country budget onto the CropGRIDS
0.05deg grid by mass-conserving proportional allocation (gTREND logic).

For each component X, national value X(k) is spread across the cells of country k by a
weight w that sums to 1 within k, then components are combined per cell:
    X_cell(c) = X(k) * w(c) / sum_{c' in k} w(c')

Weights:
  CROP : per crop item, CropGRIDS harvarea(item)              (sum per-item cell rasters)
  BNF  : per legume item, harvarea(item); rice+cane free-living, harvarea
  FERT : selected by config fertilizer.grid_weight (national magnitude always FAOSTAT RFN x CF):
         coello_cropgrids (default): Σ_g F_C(g,k)·H_g(m)s(m)/Σ_k H_g s, i.e.
           Coello et al. (2025) national crop-group split placed on CropGRIDS harvarea x s_env (the
           same basis as CROP removal); coello: Coello 5-arcmin dasymetric field (alternative);
           adalibieke; lutian; else sum_crop harvarea(crop) * NPKGRIDS Nrate(crop)
  HUM  : gridded population (resampled)        [needs population_grid]
  MAN  : gridded livestock-N density (GLW)      [needs glw]
  DEP  : native ISIMIP field (resampled)        [already a field]

Grid & country mask come from CropGRIDS (Countries_2018.nc, GAUL admin-0). harvarea is
already in hectares per cell, so weights need no area correction.

Conservation is asserted: zonal sum of each gridded component == the Tier-A national value.
"""
from __future__ import annotations
import glob
from pathlib import Path
import numpy as np
import pandas as pd

from . import config as cfgmod, faostat_prep as prep, utils
from .config import coef

CG_SUBDIR = "cropgrids/nc_maps/CROPGRIDSv1.08_NC_maps"
NPK_SUBDIR = "npkgrids/nc_maps"
LUTIAN_SUBDIR = "lu_tian_fertilizer/Lu-Tian_2016/Nfer_ASCII"   # nferyYYYY.asc, 0.5deg, 1900-2013
ADA_SUBDIR = "adalibieke_fertilization/derived"               # adalibieke_Nfert_YYYY_0p05.nc, 1961-2020
ADA_Y0, ADA_Y1 = 1961, 2020                                   # Adalibieke coverage (clamp outside)
COE_SUBDIR = "coello_fertilization/derived"                   # coello_{N,P2O5}fert_YYYY_0p05.nc, 1961-2019
COE_Y0, COE_Y1 = 1961, 2019                                   # Coello coverage (clamp outside)
ZMAN_SUBDIR = "zhang_manure/derived"                          # zhang_man_{crop,grass}_YYYY_0p05.nc weights
ZMAN_Y0 = 1961
ZMAN_LAST = {"crop": 2014, "grass": 2014}                     # Zhang cropland / production last data year


def _nearest_epoch(files, year, pat):
    """From `files`, return the path whose embedded 4-digit year (regex `pat`, group 1) is
    nearest `year` (and the chosen epoch). Used to pick GLW/MODIS/GHS epochs per model year."""
    import re
    cand = []
    for f in files:
        m = re.search(pat, Path(f).name)
        if m:
            cand.append((abs(int(m.group(1)) - year), int(m.group(1)), str(f)))
    if not cand:
        return None, None
    cand.sort()
    return cand[0][2], cand[0][1]


# ---------- grid + country mask ----------
def load_country_grid(cfg):
    """Return (mask_index_2d int32, ISO3-per-index dict, lat, lon). mask cells hold the
    CropGRIDS country index; <=0 are ocean/disputed."""
    import xarray as xr
    cg = Path(cfgmod.get_path(cfg, "raw")) / CG_SUBDIR
    cm = xr.open_dataset(cg / "Countries_2018.nc")
    mask = cm["country"].values.astype("int32")
    cw = pd.read_csv(coef(cfg, "cropgrids_country_crosswalk.csv"))
    idx2iso = dict(zip(cw["mask_index"].astype(int), cw["ISO3"]))
    return mask, idx2iso, cm["lat"].values, cm["lon"].values


def _layer(dir_path: Path, slug: str, var: str):
    """Load a CropGRIDS/NPKGRIDS variable; clip fill values / negatives / NaN to 0
    (these layers — areas and rates — are physically non-negative)."""
    import xarray as xr
    hits = glob.glob(str(dir_path / f"*_{slug}.nc"))
    if not hits:
        return None
    a = xr.open_dataset(hits[0])[var].values.astype("float32")
    a = np.where(np.isfinite(a) & (a > 0), a, 0.0)
    return a


# ---------- allocation core ----------
def allocate(national: pd.DataFrame, weight_2d: np.ndarray, mask: np.ndarray,
             idx2iso: dict, value_col: str) -> np.ndarray:
    """Distribute national[ISO3->value] across cells by weight_2d, normalized within each
    country. Returns a 2d kg array. national: columns [ISO3, value_col]."""
    flat_mask = mask.ravel()
    w = np.nan_to_num(weight_2d.ravel(), nan=0.0)
    w[w < 0] = 0.0
    nidx = int(flat_mask.max()) + 1
    wsum = np.bincount(flat_mask.clip(min=0), weights=w, minlength=nidx)  # weight sum per index
    val_by_idx = np.zeros(nidx, dtype="float64")
    iso2val = {k: v for k, v in zip(national["ISO3"], national[value_col]) if pd.notna(v)}
    for i, iso in idx2iso.items():
        if i < nidx and iso in iso2val and wsum[i] > 0:
            val_by_idx[i] = iso2val[iso]
    # per-cell value = national_value[idx] * w / wsum[idx]
    safe = np.where(wsum > 0, wsum, 1.0)
    cell = val_by_idx[flat_mask.clip(min=0)] * w / safe[flat_mask.clip(min=0)]
    cell[flat_mask < 1] = 0.0
    return cell.reshape(mask.shape)


# ---------- per-item national tables ----------
def national_crop_removal(cfg, year, nutrient="N") -> pd.DataFrame:
    """[ISO3, fao_item_code, value_kg] crop nutrient removal per item (production x coef)."""
    raw = cfgmod.get_path(cfg, "raw")
    codes = cfg["faostat"]["codes"]
    prod = prep.tidy("QCL", cfg, raw, year=year, element=codes["crop_element_production"],
                     item=None, value_name="tonnes")
    cu = utils.load_coef(coef(cfg, "crop_uptake.csv"), "crop coeffs",
                         required_cols=["fao_item_code", f"removal_kg_per_t_{nutrient}"])
    cu = cu[["fao_item_code", f"removal_kg_per_t_{nutrient}"]]
    df = prod.merge(cu, left_on="Item Code", right_on="fao_item_code", how="inner")
    df["value_kg"] = df["tonnes"].astype(float) * df[f"removal_kg_per_t_{nutrient}"].astype(float)
    out = df.groupby(["ISO3", "fao_item_code"], as_index=False)["value_kg"].sum()
    from . import dissolution
    return dissolution.redistribute_table(out, cfg, raw, "crop", keep_cols=["fao_item_code"])


# ---------- component downscalers ----------
def national_bnf_by_item(cfg, year) -> pd.DataFrame:
    """[ISO3, fao_item_code, value_kg] BNF per legume item + rice/cane free-living."""
    from . import bnf as bnfmod
    raw = cfgmod.get_path(cfg, "raw"); codes = cfg["faostat"]["codes"]
    prod = prep.tidy("QCL", cfg, raw, year=year, element=codes["crop_element_production"],
                     item=None, value_name="tonnes")
    area = prep.tidy("QCL", cfg, raw, year=year, element=codes["crop_element_area"],
                     item=None, value_name="ha")
    p = bnfmod._params(cfg); grainN = bnfmod._grain_n(cfg)
    rows = []
    leg = p[p["ndfa_group"] != "freeliving"]
    for _, r in prod.merge(leg, left_on="Item Code", right_on="fao_item_code", how="inner").iterrows():
        gn = grainN.get(pd.array([r["Item Code"]], dtype="Int64")[0])
        if gn is None or pd.isna(gn):
            continue
        rows.append((r["ISO3"], int(r["Item Code"]),
                     float(r["tonnes"]) * float(gn) * float(r["wholeplant_factor"]) * bnfmod._ndfa(r, r["ISO3"])))
    fl_items = p[p["ndfa_group"] == "freeliving"]["fao_item_code"].tolist()
    rate = cfg["bnf"]["freeliving_rate_kg_ha"]
    for _, r in area[area["Item Code"].isin(fl_items)].iterrows():
        rows.append((r["ISO3"], int(r["Item Code"]), float(r["ha"]) * rate))
    out = pd.DataFrame(rows, columns=["ISO3", "fao_item_code", "value_kg"]).dropna(subset=["value_kg"])
    from . import dissolution
    # one row per (country, item), so the gridded scatter and the residual check see the same values
    out = out.groupby(["ISO3", "fao_item_code"], as_index=False)["value_kg"].sum()
    return dissolution.redistribute_table(out, cfg, raw, "bnf", keep_cols=["fao_item_code"])


def load_population_grid(cfg, shape, year=2022):
    """Load GHS-POP (30 arcsec) at the epoch nearest `year` and block-aggregate to the model
    grid (3600x7200), oriented to match CropGRIDS (lat ascending). None if unavailable."""
    import glob as _glob
    pg = _glob.glob(str(Path(cfgmod.get_path(cfg, "raw")) / "population_grid" / "*.tif"))
    if not pg:
        return None
    chosen, epoch = _nearest_epoch(pg, year, r"E(\d{4})")
    if chosen is None:
        chosen = pg[0]
    try:
        import rasterio
        with rasterio.open(chosen) as src:
            a = src.read(1).astype("float64")
        a = np.where(np.isfinite(a) & (a > 0), a, 0.0)
        f = a.shape[0] // shape[0], a.shape[1] // shape[1]   # block factors (expect 6,6)
        if f[0] < 1 or f[1] < 1:
            return None
        a = a[:shape[0]*f[0], :shape[1]*f[1]]
        a = a.reshape(shape[0], f[0], shape[1], f[1]).sum(axis=(1, 3))   # block-sum -> coarse
        return a[::-1, :]   # GHS is north-up; CropGRIDS lat ascending -> flip vertically
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] population grid load failed ({e}); HUM will fall back to cropland weight")
        return None


def load_livestock_grid(cfg, shape, lat, lon, year=2022, codes=None, frac=None,
                        by_species=False):
    """Build the manure spatial weight = livestock N-excretion density.

    `frac` (optional {code: multiplier}) scales a species' contribution — used to split cattle
    between the confined (applied) and grazing (pasture) weights via cattle_confined_fraction.

    `by_species=True` returns a {glw_code: Nex-weighted density grid (lat ascending)} dict
    instead of the summed weight — used by the data-driven per-species applied/pasture split,
    where each species' density is combined with a country-varying applied fraction a_s(i) before
    being multiplied by the cropland vs grazing land base. `frac` is ignored in this mode.

    Prefers the AGLW annual gridded livestock product (Du et al. 2025; per-species, per-year,
    5 km, 1961-2021) when `manure.aglw_dir` is configured and present — giving a TIME-VARYING
    within-country livestock pattern (year clamped to <=2021). Falls back, per species, to the
    fixed-epoch GLW maps (nearest epoch) when AGLW is unavailable.

    If `codes` is given (list of species codes), only those species are summed — used to weight
    the applied pool by confined (monogastric) and the pasture pool by grazing ruminants. Each
    per-species density map is reprojected to the model 0.05deg grid and weighted by per-head Nex
    (coefficients/glw_excretion.csv) so the species mix reflects N (not head count), then summed.
    Returns a 2d density-like weight (lat ascending) or None if no data. The absolute scale is
    irrelevant — allocate() normalizes within each country; only the inter-species mix and
    within-country pattern matter (magnitude comes from FAOSTAT EMN).
    """
    from pathlib import Path as _P
    raw = _P(cfgmod.get_path(cfg, "raw"))
    gdir = raw / cfg["manure"].get("glw_dir", "gridded_livestock")
    adir_name = cfg["manure"].get("aglw_dir")
    adir = (raw / adir_name) if adir_name else None
    use_aglw = adir is not None and adir.exists()
    if not use_aglw and not gdir.exists():
        return None
    try:
        import rasterio
        from rasterio.warp import reproject, Resampling
        from rasterio.transform import from_origin
        exc = pd.read_csv(coef(cfg, cfg["manure"]["glw_excretion_file"]))
        if codes is not None:
            exc = exc[exc["glw_code"].isin(codes)]
        nex = dict(zip(exc["glw_code"], exc["nex_kg_head_yr"].astype(float)))
        aglw_map = cfg["manure"].get("aglw_species", {}) or {}
        yr_aglw = min(int(year), 2021)                       # AGLW covers 1961-2021 (held at 2021)
        # destination = model grid, north-up (row 0 = +90); flip to ascending at the end.
        dlon = float(abs(lon[1] - lon[0])); dlat = float(abs(lat[1] - lat[0]))
        west = float(min(lon)) - dlon / 2.0; north = float(max(lat)) + dlat / 2.0
        dst_transform = from_origin(west, north, dlon, dlat)
        acc = np.zeros(shape, dtype="float64")
        per_species = {}
        found = 0; n_aglw = 0; n_glw = 0
        for code, ne in nex.items():
            pick = None
            if use_aglw and code in aglw_map:               # AGLW per-year map for this species
                kw = str(aglw_map[code]).lower()

                def _real(p):                                # skip macOS AppleDouble junk (._*) + __MACOSX
                    return not p.name.startswith("._") and "__MACOSX" not in str(p)

                cand = [p for p in adir.rglob("*.tif*")
                        if _real(p) and kw in p.name.lower() and str(yr_aglw) in p.name]
                if not cand:                                 # year may live in the folder name
                    cand = [p for p in adir.rglob("*.tif*")
                            if _real(p) and kw in str(p).lower() and str(yr_aglw) in str(p)]
                if cand:
                    pick = str(sorted(cand)[0]); n_aglw += 1
            if pick is None:                                 # GLW nearest-epoch fallback
                hits = list(gdir.glob(f"*{code}*.tif")) if gdir.exists() else []
                if not hits:
                    print(f"  [warn] livestock species {code}: no AGLW/GLW file found"); continue
                p2, _ = _nearest_epoch(hits, year, r"(19\d\d|20\d\d)")
                pick = p2 or str(hits[0]); n_glw += 1
            with rasterio.open(pick) as src:
                src_a = src.read(1).astype("float64")
                src_a = np.where(np.isfinite(src_a) & (src_a > 0), src_a, 0.0)
                dst = np.zeros(shape, dtype="float64")
                reproject(source=src_a, destination=dst,
                          src_transform=src.transform, src_crs=src.crs or "EPSG:4326",
                          dst_transform=dst_transform, dst_crs="EPSG:4326",
                          resampling=Resampling.bilinear)
            if by_species:
                per_species[code] = (dst * ne)[::-1, :]   # lat ascending, Nex-weighted density
            acc += dst * ne * float((frac or {}).get(code, 1.0))
            found += 1
        if found == 0:
            return None
        if n_aglw:
            print(f"  livestock weight: AGLW {yr_aglw} ({n_aglw} spp)"
                  + (f" + GLW nearest-epoch ({n_glw} spp)" if n_glw else ""))
        if by_species:
            return per_species
        return acc[::-1, :]   # north-up -> lat ascending (match CropGRIDS mask)
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] livestock grid load failed ({e}); MAN will fall back to cropland weight")
        return None


def _applied_pasture_density(cfg, shape, lat, lon, year, mask, idx2iso):
    """Data-driven per-species applied/pasture livestock density split.

    Returns (d_app, d_grz): two 2d livestock-Nex-density grids where each species s is routed to
    the applied (cropland) weight by its country-varying FAOSTAT applied fraction a_s(i) and to
    the pasture (grazing) weight by (1-a_s(i)):
        d_app = Σ_s a_s(i) · D_s(m)      d_grz = Σ_s (1-a_s(i)) · D_s(m)
    a_s(i) = EMN_applied(s,i)/[EMN_applied+EMN_pasture] from manure.applied_fraction_by_species.
    Countries/species with no EMN split fall back to `manure.applied_fraction_fallback[code]`
    (else 1.0 for confined_species, 0.0 for grazing_species, 0.5 otherwise). Returns
    (None, None) if the per-species livestock grids are unavailable (caller falls back)."""
    from pathlib import Path as _P
    from . import manure as manuremod
    all_codes = (list(cfg["manure"].get("confined_species", []))
                 + list(cfg["manure"].get("grazing_species", [])))
    per_species = load_livestock_grid(cfg, shape, lat, lon, year, codes=all_codes, by_species=True)
    if not per_species:
        return None, None
    raw_dir = _P(cfgmod.get_path(cfg, "raw"))
    try:
        a_map = manuremod.applied_fraction_by_species(cfg, raw_dir, year)   # {code: {ISO3: a_s}}
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] applied_fraction_by_species failed ({e}); using fallback fractions")
        a_map = {}
    fallback = cfg["manure"].get("applied_fraction_fallback", {}) or {}
    conf = set(cfg["manure"].get("confined_species", []))
    graz = set(cfg["manure"].get("grazing_species", []))
    nidx = int(mask.max()) + 1
    clipped = mask.clip(min=0)
    d_app = np.zeros(shape, dtype="float64")
    d_grz = np.zeros(shape, dtype="float64")
    for code, dens in per_species.items():
        default = float(fallback.get(code, 1.0 if code in conf else (0.0 if code in graz else 0.5)))
        frac_iso = a_map.get(code, {})
        a_by_idx = np.array([frac_iso.get(idx2iso.get(i, ""), default) for i in range(nidx)],
                            dtype="float64")
        a_cell = a_by_idx[clipped]
        d_app += a_cell * dens
        d_grz += (1.0 - a_cell) * dens
    n_cov = sum(len(v) for v in a_map.values())
    print(f"  applied/pasture split: data-driven a_s from EMN "
          f"({len(a_map)} spp, {n_cov} country-values; fallback elsewhere)")
    return d_app, d_grz


def _cell_area_ha(latv, dlat, dlon):
    R = 6371000.0
    latr = np.deg2rad(latv)
    row = (R**2) * np.deg2rad(dlon) * (np.sin(latr + np.deg2rad(dlat)/2) - np.sin(latr - np.deg2rad(dlat)/2))
    return np.abs(row)


def _urban_fraction(pop_2d, latv, dlat, dlon, dens0=300.0, k=1.5):
    """Per-cell urban-ness u(c) in [0,1] from population DENSITY — a logistic centred on the GHS
    urban-cluster threshold (~300 persons km^-2). Used to place human-excreta recycling on RURAL
    cropland (weight = cropland x (1 - u)), reproducing the urban concentration of sewerage without a
    separate urban-population layer. density = population count / cell land area (km^2)."""
    R = 6371.0                                                  # km
    latr = np.deg2rad(np.asarray(latv, dtype="float64"))
    row_km2 = (R ** 2) * np.deg2rad(dlon) * (np.sin(latr + np.deg2rad(dlat) / 2)
                                             - np.sin(latr - np.deg2rad(dlat) / 2))
    area_km2 = np.abs(row_km2)[:, None]
    dens = np.divide(pop_2d, area_km2, out=np.zeros_like(pop_2d, dtype="float64"), where=area_km2 > 0)
    u = 1.0 / (1.0 + np.exp(-k * (np.log10(np.maximum(dens, 1e-6)) - np.log10(dens0))))
    return np.clip(u, 0.0, 1.0)


def _recycle_weights(cfg, cropland_ha, popw, lat, lon):
    """(w_ns, w_bio) — the two hum_crop pathways enter cropland by different routes and get different
    weights: NIGHT SOIL -> ALL cropland (cropland_ha, no urban/rural bias) because non-sewered excreta is
    recycled to cropland near where it is generated, both by rural households locally AND by the historical/
    present-day faecal-sludge trade that hauls collected excreta from population centres to nearby (often
    peri-urban) farmland. BIOSOLIDS -> PERI-URBAN cropland
    (cropland x urb), generated at centralized urban treatment plants. sewer_grid_mode='none' or no
    population grid -> both plain cropland. urb = urban-ness from population density (see _urban_fraction).
    allocate() conserves each pool to its national total regardless of the weight shape."""
    mode = str(cfg["human"].get("sewer_grid_mode", "rural")).lower()
    if mode == "none" or popw is None:
        return cropland_ha, cropland_ha
    dlat = abs(float(lat[1] - lat[0])); dlon = abs(float(lon[1] - lon[0]))
    urb = _urban_fraction(popw, lat, dlat, dlon)
    return cropland_ha, cropland_ha * urb


def load_hyde_grid(cfg, shape, lat, lon, year, layer):
    """HYDE 3.3 historical land base, regridded to the model grid for `year`.
    layer in {cropland, pasture, rangeland, population}. Returns per-cell quantity on the model
    grid (ha for land layers, head-count for population), or None if the file is absent.

    HYDE NetCDFs are 5 arcmin, annual. Land layers are area km2/cell; population is count/cell.
    We regrid the INTENSIVE quantity (area fraction, or population density) so it conserves
    across the 0.0833deg->0.05deg resolution change, then x model cell area.
    """
    f = Path(cfgmod.get_path(cfg, "raw")) / "hyde" / f"{layer}.nc"
    if not f.exists():
        return None
    try:
        import xarray as xr
        from rasterio.warp import reproject, Resampling
        from rasterio.transform import from_origin
        ds = xr.open_dataset(f, decode_times=False)
        # resolve var + coord names
        latn = next(c for c in ds.coords if c.lower() in ("lat", "latitude", "y"))
        lonn = next(c for c in ds.coords if c.lower() in ("lon", "longitude", "x"))
        var = next(v for v in ds.data_vars if ds[v].ndim == 3)
        hlat = ds[latn].values; hlon = ds[lonn].values
        # pick the time index nearest `year` (HYDE time = actual years, possibly via units)
        tc = ds[next(c for c in ds.coords if c.lower() == "time")]
        tv = np.atleast_1d(tc.values).astype(float)
        units = str(tc.attrs.get("units", "")); cal = str(tc.attrs.get("calendar", "standard"))
        try:                                            # proper calendar decoding -> integer years
            import cftime
            yrs = np.array([d.year for d in np.atleast_1d(cftime.num2date(tv, units, calendar=cal))])
        except Exception:                               # arithmetic fallback (handles "days since Y")
            import re
            m = re.search(r"since\s+(\d{1,4})", units.lower()); by = int(m.group(1)) if m else 0
            per = 365.2425 if "day" in units.lower() else (12.0 if "month" in units.lower() else 1.0)
            yrs = by + tv / per
        idx = int(np.argmin(np.abs(yrs - year)))
        if layer == "cropland" and abs(float(yrs[idx]) - year) > 0.5 and int(year) == HYDE_REF_YEAR:
            print(f"  [WARN] HYDE {layer} has no {year} time step; nearest {float(yrs[idx]):.0f} used as the "
                  f"reference year of the CropGRIDS scaling factor")
        a = np.asarray(ds[var].isel({"time": idx}).values, dtype="float64")
        if a.shape != (len(hlat), len(hlon)):
            a = a.T
        a = np.where(np.isfinite(a) & (a > 0), a, 0.0)
        # orient north-up (lat descending) for a clean source transform
        if hlat[0] < hlat[-1]:
            a = a[::-1, :]; hlat = hlat[::-1]
        hdlat = abs(float(hlat[0] - hlat[1])); hdlon = abs(float(hlon[1] - hlon[0]))
        carea = _cell_area_ha(hlat, hdlat, hdlon)[:, None]          # m2? -> use consistent: ha
        carea_ha = carea / 1e4
        if layer == "population":
            q = a / carea_ha                                        # head per ha (density)
        else:
            q = (a * 100.0) / carea_ha                              # km2->ha, /cell area = fraction
        west = float(hlon.min()) - hdlon / 2.0; nor = float(hlat.max()) + hdlat / 2.0
        src_tr = from_origin(west, nor, hdlon, hdlat)
        dlon = float(abs(lon[1] - lon[0])); dlat = float(abs(lat[1] - lat[0]))
        mwest = float(min(lon)) - dlon / 2.0; mnor = float(max(lat)) + dlat / 2.0
        dst = np.zeros(shape, dtype="float64")
        reproject(source=q, destination=dst, src_transform=src_tr, src_crs="EPSG:4326",
                  dst_transform=from_origin(mwest, mnor, dlon, dlat), dst_crs="EPSG:4326",
                  resampling=Resampling.bilinear)
        dst = dst[::-1, :]                                          # north-up -> model ascending
        marea_ha = (_cell_area_ha(lat, dlat, dlon) / 1e4)[:, None]
        return dst * marea_ha                                       # fraction/density -> ha or count
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] HYDE {layer} {year} load failed ({e})")
        return None


# ---------- Crop area in year t from a CropGRIDS / HYDE scaling factor  ----------
#
#   s_c(m)   = CG_c(m, 2020) / H(m, 2020)          static scaling factor (CropGRIDS reference year 2020)
#   A_c(m,t) = s_c(m) * H(m, t)                    crop area in year t
#
# CG_c is CropGRIDS harvested area of crop c, H is HYDE 3.3 cropland on the model grid. Crop areas
# reproduce CropGRIDS exactly in 2020 wherever H(m,2020) > 0. Edge rules:
#   - H(m,2020) = 0 but CG_c(m) > 0: s_c is undefined; HYDE is accepted, A_c = 0 (earlier
#     builds held the CropGRIDS area, A_c = CG_c).
# The cropland extent itself is HYDE, H(m,t) (`cropgrids_cropland_extent`); CropGRIDS only divides it
# among crops.
#   - growth cap (crop areas only): H(m,t) is limited to H(m,2020) * max(1, cell_area / CG_cropland(m)),
#     so Σ_c A_c never grows beyond max(CropGRIDS cropland, cell area) and is never cut below CropGRIDS.
HYDE_REF_YEAR = 2020           # CropGRIDS reference year
_HYDE_REF_MEMO = {}            # (raw path, ref_year) -> H(m, ref_year) on the model grid (kept for the run)


def _hyde_envelope_cfg(cfg):
    """`hyde_envelope` config block. Modes:
    - 'share' (default; alias of the original 'extent' calculation): cropland = H(m,t);
      crop areas A_c(m,t) = [CG_c(m) / CG_crop(m)] * H(m,t), i.e. the CropGRIDS crop composition of
      the cell applied to HYDE cropland of year t (CG_crop = CropGRIDS physical cropland).
    - 'scaling' : A_c(m,t) = [CG_c(m) / H(m,2020)] * H(m,t)."""
    h = cfg.get("hyde_envelope", {}) or {}
    mode = str(h.get("mode", "share")).lower()
    if mode == "ratio":                       # earlier name for the 'scaling' calculation
        mode = "scaling"
    if mode == "share":                       # same calculation as the original 'extent' mode
        mode = "extent"
    return mode, int(h.get("ref_year", HYDE_REF_YEAR))


def cropgrids_scaling_factor(cg_area, H_ref):
    """s_c(m) = CG_c(m, 2020) / H(m, 2020). NaN where H(m, 2020) = 0 (undefined)."""
    cg = np.asarray(cg_area, "float64"); H_ref = np.asarray(H_ref, "float64")
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(H_ref > 0, cg / np.where(H_ref > 0, H_ref, 1.0), np.nan)


def area_in_year(cg_area, H_ref, H_t, hold_unscaled=False):
    """A_c(m,t) = s_c(m) * H(m,t), with s_c from `cropgrids_scaling_factor`. Where s_c is undefined
    (H(m,2020) = 0) HYDE is accepted: A_c = 0, because HYDE defines where cropland is and CropGRIDS only
    divides it among crops. `hold_unscaled=True` restores the earlier rule (CropGRIDS held).
    Works on full grids or on matching 1-D slices."""
    cg = np.nan_to_num(np.asarray(cg_area, "float64"))
    s_c = cropgrids_scaling_factor(cg, H_ref)
    return np.where(np.isfinite(s_c), s_c * np.nan_to_num(np.asarray(H_t, "float64")),
                    cg if hold_unscaled else 0.0)


def hyde_scaling_inputs(cfg, shape, lat, lon, year, cg_cropland, ref_year=HYDE_REF_YEAR):
    """H(m,2020), the capped H(m,t) and the raw H(m,t) on the model grid, or (None, None, None) if HYDE
    is unavailable. The capped H(m,t) scales crop areas only; the raw H(m,t) is the cropland extent.

    Both come from `load_hyde_grid` (HYDE 3.3 cropland, ha per 0.05deg cell, annual time steps). H(m,t) is capped at H(m,2020) * max(1, cell_area/CG_cropland)
    where CropGRIDS has cropland, so cropland cannot grow beyond max(CropGRIDS, cell area)."""
    key = (str(cfgmod.get_path(cfg, "raw")), int(ref_year))
    H_ref = _HYDE_REF_MEMO.get(key)
    if H_ref is None:
        H_ref = load_hyde_grid(cfg, shape, lat, lon, ref_year, "cropland")
        if H_ref is None:
            return None, None, None
        H_ref = np.nan_to_num(np.asarray(H_ref, "float64"))
        _HYDE_REF_MEMO.clear(); _HYDE_REF_MEMO[key] = H_ref
    H_t = H_ref if int(year) == int(ref_year) else load_hyde_grid(cfg, shape, lat, lon, year, "cropland")
    if H_t is None:
        return None, None, None
    H_t = np.nan_to_num(np.asarray(H_t, "float64"))
    cg = np.nan_to_num(np.asarray(cg_cropland, "float64"))
    dlat = float(abs(lat[1] - lat[0])); dlon = float(abs(lon[1] - lon[0]))
    cell = (_cell_area_ha(np.asarray(lat, "float64"), dlat, dlon) / 1e4)[:, None] * np.ones((1, len(lon)))
    with np.errstate(divide="ignore", invalid="ignore"):
        cap = np.where(cg > 0, np.maximum(1.0, cell / np.where(cg > 0, cg, 1.0)), 1.0)   # no growth where CG = 0
    with np.errstate(invalid="ignore", over="ignore"):
        # cap can overflow to inf where CG cropland is a vanishing positive value; H(m,2020) = 0 there,
        # and area_in_year holds CropGRIDS wherever H(m,2020) = 0, so H_t is passed through unchanged.
        H_t_cap = np.where(H_ref > 0, np.minimum(H_t, H_ref * cap), H_t)
    return H_ref, H_t_cap, H_t


def cropgrids_cropland_extent(cfg, cg_cropland, shape, lat, lon, year):
    """Cropland of year t used by every cropland term, and the inputs that scale crop areas.

    Mode 'scaling' (default): cropland(m,t) = H(m,t), HYDE 3.3 cropland of year t. CropGRIDS only divides that cropland among crops:
    A_c(m,t) = s_c(m) * H(m,t), s_c(m) = CG_c(m,2020)/H(m,2020), with A_c = 0 where H(m,2020) = 0.
    Returns `(cropland_t, scaling)` where `scaling` is a dict {"H_ref", "H_t"} (2-D; H_t carries the
    growth cap) used to compute A_c(m,t) for every crop or crop group, and "factor": the per-cell
    multiplier Σ_c A_c(m,t)/CG(m) (used only by the legacy fertilizer branches).
    Mode 'extent' : HYDE cropland itself; scaling = {"factor": H/CG}.
    `(None, None)` if HYDE is unavailable."""
    mode, ref = _hyde_envelope_cfg(cfg)
    cg = np.nan_to_num(np.asarray(cg_cropland, "float64"))
    if mode == "extent":                 # config 'share': A_c = (CG_c / CG_crop) * H(m,t)
        h = load_hyde_grid(cfg, shape, lat, lon, year, "cropland")
        if h is None:
            return None, None
        h = np.clip(np.nan_to_num(np.asarray(h, "float64")), 0.0, None)
        with np.errstate(divide="ignore", invalid="ignore"):
            f = np.where(cg > 0, h / np.where(cg > 0, cg, 1.0), 0.0)
        drop = (cg > 0) & ~(h > 0)
        nocomp = (h > 0) & ~(cg > 0)
        print(f"  cropland ({year}): HYDE{year} [{h.sum()/1e6:.1f} Mha]; crop areas = CropGRIDS crop share "
              f"(CG_c / CG cropland) x HYDE{year}; {int(drop.sum()):,} cells with CropGRIDS but no HYDE "
              f"{year} cropland ({cg[drop].sum()/1e6:.2f} Mha CropGRIDS not used); {int(nocomp.sum()):,} cells "
              f"with HYDE cropland but no CropGRIDS cropland ({h[nocomp].sum()/1e6:.2f} Mha, no crop composition)")
        return h, {"H_ref": None, "H_t": None, "factor": f}
    H_ref, H_t, H_raw = hyde_scaling_inputs(cfg, shape, lat, lon, year, cg, ref)
    if H_ref is None:
        if (Path(cfgmod.get_path(cfg, "raw")) / "hyde" / "cropland.nc").exists():
            print(f"  [WARN] HYDE cropland.nc exists but could not be used for {year}/{ref}: this year runs "
                  f"WITHOUT the HYDE envelope (static CropGRIDS; no HYDE pasture/rangeland/population)")
        return None, None
    cl = np.clip(H_raw, 0.0, None)                       # cropland extent of year t = HYDE itself
    a_cg = area_in_year(cg, H_ref, H_t)                  # Σ_c A_c(m,t): CropGRIDS cropland scaled to year t
    with np.errstate(divide="ignore", invalid="ignore"):
        f = np.where(cg > 0, a_cg / np.where(cg > 0, cg, 1.0), 0.0)
    drop = (cg > 0) & ~(H_ref > 0)
    nocomp = (cl > 0) & ~(H_ref > 0)
    print(f"  cropland ({year}): HYDE{year} [{cl.sum()/1e6:.1f} Mha]; crop areas s_c(m)=CG_c/HYDE{ref} x HYDE{year} "
          f"[Σ {a_cg.sum()/1e6:.1f} Mha]; {int(drop.sum()):,} cells with CropGRIDS but no HYDE {ref} cropland "
          f"-> HYDE accepted ({cg[drop].sum()/1e6:.2f} Mha CropGRIDS dropped); {int(nocomp.sum()):,} cells with "
          f"HYDE {year} cropland but none in {ref} ({cl[nocomp].sum()/1e6:.2f} Mha, no crop composition)")
    hi = a_cg > 2.0 * cl
    print(f"  cropland ({year}): CropGRIDS cropland > 2 x HYDE{ref} in {int(hi.sum()):,} cells "
          f"({a_cg[hi].sum()/1e6:.1f} Mha of crop area on {cl[hi].sum()/1e6:.1f} Mha of HYDE cropland)")
    return cl, {"H_ref": H_ref, "H_t": H_t, "factor": f}

def load_lu_tian_fert(cfg, shape, lat, lon, year):
    """Lu & Tian (2016) annual 0.5deg synthetic-N fertilizer rate (g N m-2 cropland, 1900-2013)
    as the time-varying FERTILIZER spatial pattern. Returns the rate on the model grid (lat
    ascending) or None if the year/file is unavailable (caller then uses NPKGRIDS)."""
    if year < 1900 or year > 2013:
        return None
    f = Path(cfgmod.get_path(cfg, "raw")) / LUTIAN_SUBDIR / f"nfery{year}.asc"
    if not f.exists():
        return None
    try:
        import rasterio
        from rasterio.warp import reproject, Resampling
        from rasterio.transform import from_origin
        with rasterio.open(f) as src:
            a = src.read(1).astype("float64")
            nod = src.nodata if src.nodata is not None else -9999
            a = np.where(np.isfinite(a) & (a != nod) & (a > 0), a, 0.0)
            dlon = float(abs(lon[1] - lon[0])); dlat = float(abs(lat[1] - lat[0]))
            west = float(min(lon)) - dlon / 2.0; north = float(max(lat)) + dlat / 2.0
            dst = np.zeros(shape, dtype="float64")
            reproject(source=a, destination=dst, src_transform=src.transform, src_crs=src.crs or "EPSG:4326",
                      dst_transform=from_origin(west, north, dlon, dlat), dst_crs="EPSG:4326",
                      resampling=Resampling.bilinear)
        return dst[::-1, :]   # north-up -> lat ascending
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] Lu&Tian {year} load failed ({e}); FERT falls back to NPKGRIDS")
        return None


def load_adalibieke_fert(cfg, shape, lat, lon, year):
    """Adalibieke crop-specific synthetic-N applied (kg N/cell) as the FERTILIZER spatial WEIGHT,
    prebuilt on the model grid by scripts/12_build_adalibieke_fert.py. Annual 1961-2020 (years
    outside are clamped to the nearest available year). Returns the per-cell field (lat ascending)
    or None when disabled / the file is absent (caller then falls back to Lu&Tian, then NPKGRIDS).

    Enabled when config fertilizer.grid_weight == 'adalibieke' (default). allocate() renormalizes
    this weight within each country, so the fertilizer MASS still conserves to the FAOSTAT-RFN
    national total — Adalibieke only sets the within-country geography (a per-crop, 5-arcmin, annual
    refinement of the coarse 0.5deg Lu&Tian pattern, extended to 2020)."""
    gw = str(cfg.get("fertilizer", {}).get("grid_weight", "lutian")).lower()
    if gw != "adalibieke":
        return None
    y = int(min(max(year, ADA_Y0), ADA_Y1))                       # clamp to coverage
    f = Path(cfgmod.get_path(cfg, "raw")) / ADA_SUBDIR / f"adalibieke_Nfert_{y}_0p05.nc"
    if not f.exists():
        print(f"  [FERT][warn] Adalibieke file missing: {f} -> falling back to Lu&Tian/NPKGRIDS")
        return None
    try:
        import xarray as xr
        da = xr.open_dataset(f)["nfert_applied_kg"]
        a = np.nan_to_num(np.asarray(da.values, dtype="float64"), nan=0.0)
        # The build script writes on the EXACT model grid, so align by POSITION (a coordinate
        # reindex with a tight tolerance was fragile across xarray versions / float precision and
        # could silently void the field). Only reindex if the shape genuinely differs.
        if a.shape != shape:
            print(f"  [FERT] Adalibieke grid {a.shape} != model {shape}; reindexing by coordinate")
            a = np.nan_to_num(da.reindex(lat=lat, lon=lon, method="nearest").values, nan=0.0)
            if a.shape != shape:
                print("  [FERT][warn] Adalibieke reindex failed; falling back to Lu&Tian/NPKGRIDS")
                return None
        print(f"  FERT weight: Adalibieke crop-specific N ({y}{'' if y == year else f' clamped from {year}'}) "
              f"[global {a.sum()/1e9:.1f} Tg]")
        return a
    except Exception as e:  # noqa: BLE001
        import traceback
        print(f"  [FERT][warn] Adalibieke {year} load FAILED ({type(e).__name__}: {e}); "
              f"falling back to Lu&Tian/NPKGRIDS")
        traceback.print_exc()
        return None


def load_coello_fert(cfg, shape, lat, lon, year, nutrient="N"):
    """Coello crop-specific fertilizer applied (kg/cell) as the FERTILIZER spatial WEIGHT,
    prebuilt on the model grid by scripts/12b_build_coello_fert.py. Annual 1961-2019 (years
    outside are clamped to the nearest available year). Returns the per-cell field (lat ascending)
    or None when disabled / the file is absent (caller then falls back to Lu&Tian, then NPKGRIDS).

    Alternative to grid_weight 'coello_cropgrids' (load_coello_group_table +
    coello_cropgrids_weight), which is used for the released product.

    Enabled when config fertilizer.grid_weight == 'coello'. allocate() renormalizes this weight
    within each country, so the fertilizer MASS still conserves to FAOSTAT-RFN x our own CF —
    Coello only sets the within-country geography.

    Why Coello over Lu&Tian: annual and crop-specific at 5-arcmin across 1961-2019, so it replaces
    BOTH the Lu&Tian 0.5deg pattern (which ends in 2013) AND the frozen-2020 NPKGRIDS fallback,
    removing the 2013/14 splice. Only 2020-2023 are clamped (to 2019) — a 4-year freeze against
    the 10-year freeze the NPKGRIDS fallback imposed.

    NOTE: unlike the CropGRIDS-2020-area weights (Adalibieke, NPKGRIDS), the Coello field carries
    its OWN annual cropland extent, so the HYDE envelope (s_env) is deliberately NOT applied to it.
    Its units are a per-cell mass density x cell area (see scripts/12b), NOT a rate per cropland
    area — so it is used directly, never multiplied by cropland_ha (that would double-count area)."""
    gw = str(cfg.get("fertilizer", {}).get("grid_weight", "lutian")).lower()
    if gw != "coello":
        return None
    tag = "N" if nutrient == "N" else "P2O5"
    y = int(min(max(year, COE_Y0), COE_Y1))                       # clamp to coverage
    f = Path(cfgmod.get_path(cfg, "raw")) / COE_SUBDIR / f"coello_{tag}fert_{y}_0p05.nc"
    if not f.exists():
        print(f"  [FERT][warn] Coello file missing: {f} -> falling back to Lu&Tian/NPKGRIDS")
        return None
    try:
        import xarray as xr
        da = xr.open_dataset(f)["nfert_applied_kg"]
        a = np.nan_to_num(np.asarray(da.values, dtype="float64"), nan=0.0)
        # The build script writes on the EXACT model grid, so align by POSITION (a coordinate
        # reindex with a tight tolerance proved fragile across xarray versions and could silently
        # void the field — this bit us with Adalibieke). Only reindex if the shape truly differs.
        if a.shape != shape:
            print(f"  [FERT] Coello grid {a.shape} != model {shape}; reindexing by coordinate")
            a = np.nan_to_num(da.reindex(lat=lat, lon=lon, method="nearest").values, nan=0.0)
            if a.shape != shape:
                print("  [FERT][warn] Coello reindex failed; falling back to Lu&Tian/NPKGRIDS")
                return None
        if not np.isfinite(a).any() or a.sum() <= 0:
            print("  [FERT][warn] Coello field is empty; falling back to Lu&Tian/NPKGRIDS")
            return None
        print(f"  FERT weight: Coello crop-specific {tag} ({y}"
              f"{'' if y == year else f' clamped from {year}'}) [global {a.sum()/1e9:.1f} Tg]")
        return a
    except Exception as e:  # noqa: BLE001
        import traceback
        print(f"  [FERT][warn] Coello {year} load FAILED ({type(e).__name__}: {e}); "
              f"falling back to Lu&Tian/NPKGRIDS")
        traceback.print_exc()
        return None


# ---------- Coello group split x CropGRIDS placement (fertilizer.grid_weight: coello_cropgrids) ----------
# Coello sets ONLY each country's split of fertilizer among its 13 crop groups; each
# group is then placed on the SAME area basis as crop removal (CropGRIDS harvarea x s_env), so the
# fertilizer and removal maps share one crop map.
COE_GROUP_FILE = "coello_group_national_{tag}.csv"   # scripts/12d_build_coello_national_groups.py
_COE_GROUP_TBL_CACHE = {}       # csv path -> DataFrame; read once per process (26 loops 63 years)
_COE_GROUP_HA_MEMO = {}         # static group harvested area built from the weight cache
_COE_XWALK_REPORTED = set()     # crosswalk gaps already reported (print once per process)


def _fert_grid_weight(cfg):
    return str(cfg.get("fertilizer", {}).get("grid_weight", "lutian")).lower()


def load_coello_group_table(cfg, year, nutrient="N"):
    """National Coello fertilizer per crop group, F_C(g,k,t), for the clamped year.

    Returns a DataFrame [ISO3, coello_group, fert_kg] (kg N, or kg P2O5 when nutrient != 'N'),
    with ``.attrs`` = {year_used, year_req, tag}, or None when config fertilizer.grid_weight is not
    'coello_cropgrids', the table built by scripts/12d_build_coello_national_groups.py is missing or
    unreadable, or the year has no rows. On None the caller falls through to the existing
    coello -> Adalibieke -> Lu&Tian -> NPKGRIDS chain. Years outside 1961-2019 are clamped. The CSV
    is cached across calls (module-level dict) because 26_run_historical_grids loops years."""
    if _fert_grid_weight(cfg) != "coello_cropgrids":
        return None
    tag = "N" if nutrient == "N" else "P2O5"
    y = int(min(max(year, COE_Y0), COE_Y1))                       # clamp to Coello coverage
    f = Path(cfgmod.get_path(cfg, "raw")) / COE_SUBDIR / COE_GROUP_FILE.format(tag=tag)
    key = str(f)
    if key not in _COE_GROUP_TBL_CACHE:
        if not f.exists():
            print(f"  [FERT][warn] grid_weight=coello_cropgrids but the Coello national group table is "
                  f"missing: {f}\n               build it with: python scripts/12d_build_coello_national_groups.py "
                  f"--nutrient {tag}  -> falling back to the Lu&Tian/NPKGRIDS chain")
            return None
        try:
            t = pd.read_csv(f)
        except Exception as e:  # noqa: BLE001
            print(f"  [FERT][warn] Coello group table unreadable ({type(e).__name__}: {e}) -> "
                  f"falling back to the Lu&Tian/NPKGRIDS chain")
            return None
        need = ["ISO3", "year", "coello_group", "fert_kg"]
        if not set(need) <= set(t.columns):
            print(f"  [FERT][warn] Coello group table {f.name} lacks columns {need} -> falling back")
            return None
        t = t[need].copy()
        t["year"] = t["year"].astype(int)
        t["fert_kg"] = pd.to_numeric(t["fert_kg"], errors="coerce").fillna(0.0)
        _COE_GROUP_TBL_CACHE[key] = t
    t = _COE_GROUP_TBL_CACHE[key]
    sub = t[(t["year"] == y) & (t["fert_kg"] > 0)][["ISO3", "coello_group", "fert_kg"]].reset_index(drop=True)
    if sub.empty:
        print(f"  [FERT][warn] Coello group table has no rows for {y} -> falling back to the Lu&Tian/NPKGRIDS chain")
        return None
    sub.attrs.update(year_used=y, year_req=int(year), tag=tag)
    return sub


def _coello_crop_groups(cfg):
    """({CropGRIDS slug -> Coello group} without EXCLUDE_forage, {forage slugs}) from
    coefficients/coello_cropgrids_crosswalk.csv (same convention as scripts/12b)."""
    x = pd.read_csv(coef(cfg, "coello_cropgrids_crosswalk.csv"))
    forage = set(x.loc[x["coello_group"] == "EXCLUDE_forage", "cropgrids_crop"])
    x = x[x["coello_group"] != "EXCLUDE_forage"]
    return dict(zip(x["cropgrids_crop"], x["coello_group"])), forage


def _report_coello_xwalk_gaps(slug2grp, forage, present_slugs):
    """Print once: Coello-crosswalk crops absent from the removal crop set (they carry no group area),
    and removal crops absent from the Coello crosswalk (their area joins no group)."""
    present = set(present_slugs)
    no_area = tuple(sorted(set(slug2grp) - present))
    no_group = tuple(sorted(present - set(slug2grp) - forage))
    key = (no_area, no_group)
    if key in _COE_XWALK_REPORTED:
        return
    _COE_XWALK_REPORTED.add(key)
    if no_area:
        print(f"  [FERT][note] {len(no_area)} non-forage crop(s) in coello_cropgrids_crosswalk are not in the "
              f"CropGRIDS crop set used for removal (no file / not in cropgrids_item_crosswalk), so they add "
              f"no group area: {', '.join(no_area)}")
    if no_group:
        print(f"  [FERT][note] {len(no_group)} removal crop(s) missing from coello_cropgrids_crosswalk "
              f"(their harvarea joins no Coello group): {', '.join(no_group)}")


def _group_ha_from_parts(parts):
    """{group: [(cells, ha), ...]} -> {group: (cells, ha)} concatenated (duplicate cells allowed)."""
    out = {}
    for g, lst in parts.items():
        out[g] = (np.concatenate([c for c, _ in lst]),       # native dtypes (int32/float32) to keep
                  np.concatenate([v for _, v in lst]))       # the memoized stack small
    return out


def coello_group_ha_from_cache(cfg, cache):
    """Group harvested area H_g(m) = Σ_{c∈g} CropGRIDS harvarea_c(m) from the weight cache (fast path).

    The cache stores the per-crop harvarea CSR stack (cells, ha, ptr) keyed by FAO item code; codes map
    to CropGRIDS slugs via cropgrids_item_crosswalk.csv and slugs to Coello groups via
    coello_cropgrids_crosswalk.csv (EXCLUDE_forage dropped). The cache holds only crosswalk items with
    has_cropgrids_file, i.e. exactly the crops whose removal is gridded, so H_g uses the removal crop set.
    Returns {group: (flat cell index, ha)} in the cache's dtypes; memoized (static across years)."""
    cells = cache["cells"]; ha = cache["ha"]; ptr = cache["ptr"]; codes = cache["codes"]
    key = (int(len(codes)), int(ptr[-1]), int(cache["mask"].size))
    if key in _COE_GROUP_HA_MEMO:
        return _COE_GROUP_HA_MEMO[key]
    xw = pd.read_csv(coef(cfg, "cropgrids_item_crosswalk.csv"))
    code2slug = dict(zip(xw["fao_item_code"].astype(int), xw["slug"]))
    slug2grp, forage = _coello_crop_groups(cfg)
    parts, present = {}, []
    for j, code in enumerate(codes):
        slug = code2slug.get(int(code))
        if slug is None:
            continue
        present.append(slug)
        g = slug2grp.get(slug)
        if g is None:                                    # forage (EXCLUDE_forage) or not in crosswalk
            continue
        parts.setdefault(g, []).append((cells[ptr[j]:ptr[j + 1]], ha[ptr[j]:ptr[j + 1]]))
    _report_coello_xwalk_gaps(slug2grp, forage, present)
    out = _group_ha_from_parts(parts)
    _COE_GROUP_HA_MEMO[key] = out
    return out


def coello_cropgrids_weight(group_tbl, group_ha, s_env, cropland_ha, all_ha, mask, idx2iso):
    """Fertilizer weight W(m,t) = Σ_g F_C(g,k,t) · w_g(m,t)  [kg; Σ_{m∈k} W = Σ_g F_C(g,k)].

        w_g(m,t) = A_g(m,t) / Σ_{m'∈k} A_g(m',t),   A_g(m,t) = s_g(m)·H(m,t),  s_g(m) = H_g(m)/H(m,2020)

    group_tbl  : [ISO3, coello_group, fert_kg] national Coello totals F_C (load_coello_group_table)
    group_ha   : {group: (flat cells, ha)} sparse, or {group: 2-D array}; H_g = CropGRIDS harvarea
    s_env      : the scaling used for removal: a dict {"H_ref", "H_t"} from `cropgrids_cropland_extent`
                 (A_g = s_g·H(m,t), `area_in_year`), a 2-D per-cell multiplier (legacy 'extent' mode), or
                 None (then A_g = H_g, the CropGRIDS area)
    cropland_ha: 2-D cropland area (year-t extent when the envelope is on) for the last-resort fallback, or None
    all_ha     : 2-D all-crop (non-forage) harvarea for fallback (a), or None to use Σ_g H_g
    Fallback (a): a group with F_C(g,k) > 0 but Σ_{m∈k} H_g·s = 0 is spread over all-crop harvarea·s
    in k; if that is also zero, over cropland_ha. Mass with neither is reported and left out (if that
    leaves the whole country at zero, the caller's _fill_zero_weight_countries backfills it).
    Countries with no Coello rows get W = 0 here and are likewise backfilled by the caller."""
    shape = mask.shape
    flat_mask = mask.ravel()
    ncell = flat_mask.size
    nidx = int(mask.max()) + 1
    iso2idx = {v: k for k, v in idx2iso.items()}
    Href_flat = Ht_flat = s_flat = None
    if isinstance(s_env, dict) and s_env.get("H_ref") is not None:
        Href_flat = np.nan_to_num(np.asarray(s_env["H_ref"], "float64").ravel())
        Ht_flat = np.nan_to_num(np.asarray(s_env["H_t"], "float64").ravel())
    elif isinstance(s_env, dict):
        s_flat = np.nan_to_num(np.asarray(s_env["factor"], "float64").ravel())
    elif s_env is not None:
        s_flat = np.nan_to_num(np.asarray(s_env, "float64").ravel())

    def _to_year(c, h):
        """CropGRIDS area on cells c -> area in year t (same rule as crop removal)."""
        if Href_flat is not None:
            return area_in_year(h, Href_flat[c], Ht_flat[c])
        if s_flat is not None:
            return h * s_flat[c]
        return h

    def _sparse(a):
        if isinstance(a, tuple):                              # per-group temporaries only
            return np.asarray(a[0]), np.asarray(a[1], "float64")
        f = np.nan_to_num(np.asarray(a, "float64").ravel())
        c = np.nonzero(f > 0)[0]
        return c, f[c]

    W = np.zeros(ncell, "float64")
    resid = np.zeros(nidx, "float64")        # per-country F_C of groups with no group area (fallback a)
    tot_FC = unmatched = 0.0
    n_fb_pairs = 0
    fb_countries = set()
    for g, sub in group_tbl.groupby("coello_group"):
        Fc = np.zeros(nidx, "float64")
        for iso, v in zip(sub["ISO3"], sub["fert_kg"]):
            i = iso2idx.get(iso)
            if not (v > 0):
                continue
            if i is None or i < 1 or i >= nidx:
                unmatched += float(v); continue
            Fc[i] += float(v)
        tot_FC += float(Fc.sum())
        c, h = (_sparse(group_ha[g]) if g in group_ha
                else (np.zeros(0, "int64"), np.zeros(0, "float64")))
        h = _to_year(c, h)                                    # A_g(m,t): same scaling as removal
        k = flat_mask[c]
        land = (k >= 1) & (h > 0)
        c, h, k = c[land], h[land], k[land]
        den = np.bincount(k, weights=h, minlength=nidx)
        miss = (Fc > 0) & (den <= 0)
        if miss.any():
            resid[miss] += Fc[miss]
            n_fb_pairs += int(miss.sum()); fb_countries.update(np.nonzero(miss)[0].tolist())
        rate = np.where(den > 0, Fc / np.where(den > 0, den, 1.0), 0.0)
        W += np.bincount(c, weights=h * rate[k], minlength=ncell)

    m_all = m_cl = m_lost = 0.0
    if resid.any():
        kflat = flat_mask.clip(min=0)
        if all_ha is None:
            A = np.zeros(ncell, "float64")
            for a_g in group_ha.values():
                c, h = _sparse(a_g)
                A += np.bincount(c, weights=h, minlength=ncell)
        else:
            A = np.nan_to_num(np.asarray(all_ha, "float64").ravel()).copy()
        A = _to_year(np.arange(ncell), A)
        A[flat_mask < 1] = 0.0
        denA = np.bincount(kflat, weights=A, minlength=nidx)
        useA = (resid > 0) & (denA > 0)
        W += A * np.where(useA, resid / np.where(useA, denA, 1.0), 0.0)[kflat]
        m_all = float(resid[useA].sum())
        left = (resid > 0) & ~useA
        if left.any() and cropland_ha is not None:
            C = np.nan_to_num(np.asarray(cropland_ha, "float64").ravel()).copy()
            C[flat_mask < 1] = 0.0
            denC = np.bincount(kflat, weights=C, minlength=nidx)
            useC = left & (denC > 0)
            W += C * np.where(useC, resid / np.where(useC, denC, 1.0), 0.0)[kflat]
            m_cl = float(resid[useC].sum())
            left = left & ~useC
        m_lost = float(resid[left].sum())

    a = getattr(group_tbl, "attrs", {}) or {}
    yu, yr, tag = a.get("year_used"), a.get("year_req"), a.get("tag", "")
    ytxt = "" if yu is None else (f"{yu}" if yu == yr else f"{yu} clamped from {yr}")
    _sc = " x s_g(m)H(m,t)" if Href_flat is not None else (" x s_env" if s_flat is not None else "")
    print(f"  FERT weight: Coello {tag} group split, CropGRIDS harvarea{_sc}"
          f" ({ytxt}) [Coello national group total {tot_FC/1e9:.2f} Tg; {group_tbl['coello_group'].nunique()} groups]"
          f"  fallback(a): {n_fb_pairs} group-country pair(s) in {len(fb_countries)} country(ies), "
          f"{m_all/1e9:.4f} Tg -> all-crop harvarea, {m_cl/1e9:.4f} Tg -> cropland, {m_lost/1e9:.4f} Tg unplaced"
          + (f"; {unmatched/1e9:.4f} Tg on ISO3 not in mask" if unmatched > 0 else ""))
    return W.reshape(shape)


def load_zhang_manure(cfg, shape, lat, lon, year, pool):
    """Prebuilt gridded manure weight from Zhang et al. 2017 (ESSD 9:667), built by
    scripts/12c_build_zhang_manure.py. Returns (array, placed):

      pool 'crop'  -> var 'man_weight' (placed=True): cropland manure ALREADY distributed to 0.05deg
                      (area-conservative overlap, mass kept within each Zhang 5' cell). Placement basis
                      is the file attribute `placement`: 'hyde_cropland_overlap' (default;
                      HYDE 3.3 cropland H(m,t), the year-t extent of every other cropland term),
                      'cropgrids_hyde_scaling_overlap' or 'cropgrids_croparea_overlap'
                      (--placement cropgrids). The caller
                      uses it DIRECTLY as the allocate() weight — do NOT multiply by cropland_ha again.
                      Note: 2015-2023 use the 2014 field (clamp below), i.e. the 2014 extent, while
                      the other cropland terms use the extent of the model year. The slow path
                      (run_downscale, no HYDE envelope) uses CropGRIDS cropland for every other term, so
                      with hyde placement its manure extent differs from its other cropland terms; the same holds
                      for P runs, where the HYDE envelope is not applied to the other terms.
      pool 'grass' -> var 'man_density' (placed=False): 5-arcmin residual density; the caller
                      multiplies by the HYDE grazing base (CropGRIDS is cropland-only, cannot refine it).

    WEIGHT-ONLY: returns the within-country PATTERN only; the caller enforces the exact FAOSTAT EMN
    national totals via allocate() (applied 72381 -> cropland; pasture 72380 -> grassland). Gap years
    are clamped to the last data year (2014); FAOSTAT then sets the gap-year magnitude, so the pattern is
    frozen while the magnitude keeps evolving. Enabled when manure.grid_source=='zhang'; returns
    (None, False) if disabled/missing (caller keeps the EMN/AGLW livestock-density weight)."""
    if str(cfg.get("manure", {}).get("grid_source", "emn")).lower() != "zhang":
        return None, False
    if pool not in ("crop", "grass"):
        return None, False
    y = int(min(max(year, ZMAN_Y0), ZMAN_LAST[pool]))
    f = Path(cfgmod.get_path(cfg, "raw")) / ZMAN_SUBDIR / f"zhang_man_{pool}_{y}_0p05.nc"
    if not f.exists():
        print(f"  [MAN][warn] Zhang manure file missing: {f} -> keeping EMN/AGLW weight")
        return None, False
    try:
        import xarray as xr
        d = xr.open_dataset(f)
        placed = "man_weight" in d.data_vars                          # crop pool = pre-placed at 0.05deg
        var = "man_weight" if placed else "man_density"
        a = np.nan_to_num(np.asarray(d[var].values, dtype="float64"), nan=0.0)
        if a.shape != shape or a.sum() <= 0:
            print(f"  [MAN][warn] Zhang {pool} grid unusable ({a.shape}); keeping EMN/AGLW weight")
            return None, False
        basis = d[var].attrs.get("placement", "")
        if (pool == "crop" and _hyde_envelope_cfg(cfg)[0] in ("scaling", "extent")
                and basis != "hyde_cropland_overlap"):
            print(f"  [MAN][WARN] Zhang crop field {y} has placement '{basis or 'unset'}', not "
                  f"'hyde_cropland_overlap': rebuild with scripts/12c_build_zhang_manure.py --pool crop")
        print(f"  MAN weight: Zhang {pool} ({y}{'' if y == year else f' clamped from {year}'}"
              f"{', placed' if placed else ', density x land base'}{f' [{basis}]' if basis else ''})")
        return a, placed
    except Exception as e:  # noqa: BLE001
        print(f"  [MAN][warn] Zhang {pool} {year} load failed ({e}); keeping EMN/AGLW weight")
        return None, False


def _zhang_grass_weight(zg, lat, lon):
    """Grassland-manure placement weight = Zhang residual manure MASS per cell.

    Zhang's residual is a manure DENSITY per unit grid-cell (pixel) area (kg N km-2), so the mass per
    cell is density x cell area. That mass field IS the weight (allocate() then imposes the FAOSTAT
    pasture national total). The grazing land base is deliberately NOT used: HYDE reports only a grazing
    FRACTION per cell, not where the grazing is, and it is the same 5-arcmin resolution as Zhang, so
    re-weighting by grazing area adds no spatial information and would override Zhang's own
    livestock/production-system placement (which already encodes the grazing-manure distribution).
    Grassland manure is therefore represented at Zhang's native 5-arcmin resolution."""
    dlat = float(abs(lat[1] - lat[0])); dlon = float(abs(lon[1] - lon[0]))
    cellarea_km2 = (_cell_area_ha(lat, dlat, dlon) / 1e6)[:, None]     # m2 -> km2; density x this -> kg
    return zg * cellarea_km2                                           # manure mass per cell (kg)


GPW_SUBDIR = "gpw"                                                     # Global Pasture Watch 0.05deg exports
GPW_GY0, GPW_GY1 = 2000, 2022                                         # GPW coverage; model years clamped to this


def load_gpw_grazing(cfg, shape, lat, lon, year):
    """Year-matched Global Pasture Watch grazing-land AREA (ha) on the model grid, from the 2-band export
    GPW_grassland_frac_0p05_<year>.tif (band1 pasture=cultivated, band2 rangeland=natural/semi; fraction
    x 10000; built by scripts/gee_export_gpw_grassland_submit.py). Returns (pasture_ha, graze_ha):
      pasture_ha = band1 x cell area           -> NARROW managed-pasture base (grassland FERTILIZER)
      graze_ha   = (band1 + band2) x cell area -> BROAD pasture+rangeland base (PRP manure, forage, BNF)
    year clamped to [2000, 2022]; falls back to the newest GPW tif present (supports a single static-year
    export). (None, None) if no GPW export is in data/raw/gpw -> caller keeps the HYDE/MODIS grazing base."""
    import glob as _glob
    gdir = Path(cfgmod.get_path(cfg, "raw")) / GPW_SUBDIR
    yy = min(max(int(year), GPW_GY0), GPW_GY1)
    f = gdir / f"GPW_grassland_frac_0p05_{yy}.tif"
    if not f.exists():
        hits = _glob.glob(str(gdir / "GPW_grassland_frac_0p05_*.tif"))
        if not hits:
            return None, None
        f = Path(max(hits, key=lambda p: Path(p).stat().st_mtime))
    try:
        import rasterio
        with rasterio.open(f) as s:
            past = s.read(1).astype("float64") / 1e4
            rang = (s.read(2).astype("float64") / 1e4) if s.count > 1 else np.zeros_like(past)
    except Exception as e:  # noqa: BLE001
        print(f"  [GPW][warn] {f.name} read failed ({e}); keeping HYDE/MODIS grazing base")
        return None, None
    past = np.clip(np.where(np.isfinite(past), past, 0.0), 0.0, 1.0)[::-1, :]   # -> lat ascending
    rang = np.clip(np.where(np.isfinite(rang), rang, 0.0), 0.0, 1.0)[::-1, :]
    if past.shape != shape:
        print(f"  [GPW][warn] {f.name} shape {past.shape} != model {shape}; keeping HYDE/MODIS grazing base")
        return None, None
    dlat = float(abs(lat[1] - lat[0])); dlon = float(abs(lon[1] - lon[0]))
    ca = (_cell_area_ha(lat, dlat, dlon) / 1e4)[:, None]              # m2 -> ha per cell (row-broadcast)
    print(f"  GRAZE base: GPW {f.name} (pasture + pasture+rangeland)")
    return (past * ca).astype("float64"), ((past + rang) * ca).astype("float64")


def _p_graze_rates(cfg):
    """(p_managed, p_range) in kg N ha-1 yr-1 from coefficients/pasture_uptake.csv.

    Two rates keyed to the Global Pasture Watch bands: band 1 cultivated/managed pasture and
    band 2 natural or semi-natural rangeland. Prescribed from the measured pasture N-yield
    ranges cited in the methods (~100-300 managed, 10-50 extensive), not fitted to any
    comparison dataset, so TREND-nitrogen and the Eurostat/OECD balance stay independent."""
    pm, pr = 200.0, 30.0
    try:
        t = pd.read_csv(coef(cfg, "pasture_uptake.csv"))
        m = {str(k).strip().lower(): float(v) for k, v in zip(t["band"], t["p_graze_kg_ha"])}
        pm = m.get("managed", pm); pr = m.get("rangeland", pr)
    except Exception as e:  # noqa: BLE001
        print(f"  [FORAGE][warn] pasture_uptake.csv unreadable ({e}); using {pm:.0f}/{pr:.0f} kg N/ha")
    return pm, pr


def _forage_gridscale(cfg, raw_dir, year, man_past, weight, pasture_ha, graze_ha, mask, idx2iso):
    """Eq. 29 evaluated PER GRID CELL; the national total is the sum of the cells.

        FORAGE(m) = min[ MAN_graze(m) + N_prod,graze(m),
                         p_managed * A_pasture(m) + p_range * A_rangeland(m) ]

    MAN_graze is the gridded pasture-deposited manure passed in; N_prod,graze is the national
    grazing-animal-product N downscaled on the same grazing proxy. Evaluating the minimum on
    national aggregates instead over-credits removal by construction, since the minimum of sums
    is at least the sum of minima."""
    from . import forage as foragemod
    prod = foragemod.product_n(cfg, raw_dir, year)
    if prod is None or prod.empty:
        prod_grid = np.zeros_like(man_past)
        print("  [FORAGE][warn] no grazing-product N; demand = PRP only")
    else:
        prod_grid = allocate(prod.rename(columns={"prod_n_kg": "value_kg"})[["ISO3", "value_kg"]],
                             weight, mask, idx2iso, "value_kg")
    demand = man_past + prod_grid
    pm, pr = _p_graze_rates(cfg)
    p_ha = np.clip(np.asarray(pasture_ha, "float64"), 0.0, None)
    r_ha = np.clip(np.asarray(graze_ha, "float64") - p_ha, 0.0, None)
    supply = pm * p_ha + pr * r_ha
    forage = np.minimum(demand, supply)
    gz = (p_ha + r_ha) > 1.0
    nb = int((gz & (demand > supply)).sum())
    print(f"  FORAGE grid-scale Eq.29 (p_managed {pm:.0f} / p_range {pr:.0f} kg N/ha): "
          f"demand {demand.sum()/1e9:.2f} Tg -> removal {forage.sum()/1e9:.2f} Tg; "
          f"ceiling binds in {nb:,} of {int(gz.sum()):,} grazing cells "
          f"({100*nb/max(int(gz.sum()),1):.1f}%)")
    return forage


def _cap_grassland_fert(fert_grass, man_past, broad_area, modis_grass_area, mask, idx2iso, cap_rate=350.0):
    """Cap grassland FERTILIZER so that fertilizer + pasture-deposited manure stays at or below
    `cap_rate` kg N ha-1 of grazing land, spilling the excess fertilizer within the country.

    Pasture-deposited manure is NOT capped. Regulatory grassland-N limits bind on the
    discretionary input, and a high grazing-manure loading is a real signal of a
    feed-supplemented system rather than an error to be scaled away. The fertilizer national
    total is conserved; where a country has no headroom anywhere the remainder is accepted."""
    shape = fert_grass.shape
    flat = mask.ravel().clip(min=0); nidx = int(mask.max()) + 1

    def agg(a):
        return np.bincount(flat, weights=np.asarray(a, dtype="float64").ravel(), minlength=nidx)

    F = np.asarray(fert_grass, dtype="float64").copy()
    M = np.asarray(man_past, dtype="float64")
    head = np.clip(cap_rate * np.clip(broad_area, 0.0, None) - M, 0.0, None)   # room left for fertilizer
    exc = np.clip(F - head, 0.0, None)
    F -= exc
    E = agg(exc)
    if E.sum() <= 0:
        return F, np.asarray(man_past, dtype="float64")
    for _ in range(6):
        if E.sum() <= 1e-3:
            break
        room = np.clip(head - F, 0.0, None); Rc = agg(room)
        place = np.minimum(E, Rc)
        with np.errstate(divide="ignore", invalid="ignore"):
            rate = np.where(Rc > 0, place / np.maximum(Rc, 1e-12), 0.0)
        F += room * rate[flat].reshape(shape); E = E - place
    acc = 0.0
    if E.sum() > 1e-3:                                    # no headroom anywhere: accept over the cap
        w = np.clip(broad_area, 0.0, None); Wc = agg(w)
        with np.errstate(divide="ignore", invalid="ignore"):
            rate = np.where(Wc > 0, E / np.maximum(Wc, 1e-12), 0.0)
        F += w * rate[flat].reshape(shape); acc = E.sum()
    print(f"  [cap] grassland fertilizer held under {cap_rate:.0f} kg N/ha with PRP: "
          f"moved {agg(exc).sum()/1e9:.3f} Tg; {acc/1e9:.3f} Tg had no headroom and was accepted")
    return F, np.asarray(man_past, dtype="float64")


def _fill_zero_weight_countries(w, fallback, mask, label="FERT"):
    """Substitute `fallback` inside any country whose `w` sums to zero.

    allocate() SILENTLY DROPS a country's national value when its weight sums to zero (the
    `wsum[i] > 0` guard), which would leak that country's fertilizer out of the budget entirely.
    Coello maps 13 crop groups and has no coverage in ~25 small island/territory countries
    (~0.27 Mha, 0.02% of global cropland), so without this those countries would report zero
    fertilizer. Returns a copy of `w` with those countries filled from `fallback`."""
    if fallback is None:
        return w
    flat = mask.ravel().clip(min=0)
    nidx = int(mask.max()) + 1
    wsum = np.bincount(flat, weights=np.nan_to_num(w.ravel()), minlength=nidx)
    fsum = np.bincount(flat, weights=np.nan_to_num(fallback.ravel()), minlength=nidx)
    need = (wsum <= 0) & (fsum > 0)
    need[0] = False                                    # index <=0 is ocean/disputed
    if not need.any():
        return w
    out = w.copy()
    sub = need[flat].reshape(mask.shape)
    out[sub] = np.nan_to_num(fallback[sub])
    print(f"  [{label}] {int(need.sum())} country(ies) with no primary weight -> filled from "
          f"fallback (prevents allocate() dropping their national total)")
    return out


def cropland_dep_weight(dep_w, cropland_ha, mask, mode="isimip"):
    """Weight for distributing the national cropland deposition (Vishwakarma WL, W(i,t)*A_crop,i) over
    the cropland of each country.

    mode "isimip" (default): ISIMIP3a total-N deposition rate x cropland area, so the per-ha rate in each
    cell follows the ISIMIP within-country gradient while the national total stays W(i,t)*A_crop,i
    (the cropland-weighted mean rate of every country equals its WL rate). Cells with no valid ISIMIP
    rate take the cropland-weighted mean ISIMIP rate of their country; a country with no valid ISIMIP
    rate at all falls back to a uniform rate. mode "uniform" (or dep_w None): cropland area only."""
    if dep_w is None or mode == "uniform":
        return cropland_ha
    flat = mask.ravel().clip(min=0)
    n = int(flat.max()) + 1
    c = np.nan_to_num(cropland_ha.ravel(), nan=0.0)
    r = dep_w.ravel().astype("float64")
    ok = np.isfinite(r) & (r > 0)
    num = np.bincount(flat, weights=np.where(ok, r * c, 0.0), minlength=n)
    den = np.bincount(flat, weights=np.where(ok, c, 0.0), minlength=n)
    cmean = np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)
    fill = cmean[flat]
    rate = np.where(ok, r, fill)
    rate = np.where(np.isfinite(rate) & (rate > 0), rate, 1.0)      # no ISIMIP in the country: uniform
    return (rate * c).reshape(cropland_ha.shape)


def load_deposition_grid(cfg, shape, lat, lon, year):
    """Native-resolution TOTAL atmospheric N deposition (NOy+NHx) regridded to the model grid,
    as a per-cell RATE (kg N/ha). Used as the spatial WEIGHT for distributing the national
    deposition magnitude, so the within-country gradient is resolved while the national total
    still conserves to the country table. ISIMIP3a histsoc 0.5deg, monthly g N m-2; year clamped
    to 1901-2021. Returns the rate (lat ascending) or None (caller falls back to country-mean)."""
    dep_dir = Path(cfgmod.get_path(cfg, "raw")) / "deposition"
    if not dep_dir.exists():
        return None

    def _annual(species):
        hits = [p for p in dep_dir.glob(f"ndep-{species}*_1901_2021.nc")
                if "dry" not in p.name and "wet" not in p.name]
        if not hits:
            return None, None, None
        import xarray as xr
        ds = xr.open_dataset(sorted(hits)[0], decode_times=False)
        var = next(v for v in ds.data_vars if v != "time_bnds")
        t = ds["time"].values
        y = int(min(max(year, 1901), 2021))                         # clamp to coverage
        sel = np.where((t >= (y - 1850) * 12) & (t < (y - 1849) * 12))[0]
        if len(sel) == 0:
            return None, None, None
        a = ds[var].isel(time=sel).sum("time").values * 10.0        # g m-2 yr-1 -> kg ha-1
        return a, ds[var]["lat"].values, ds[var]["lon"].values

    try:
        noy, dla, dlo = _annual("noy")
        if noy is None:
            return None
        nhx, _, _ = _annual("nhx")
        tot = noy + (nhx if nhx is not None else 0.0)
        from rasterio.warp import reproject, Resampling
        from rasterio.transform import from_origin
        if dla[0] < dla[-1]:                                         # orient source north-up
            tot = tot[::-1, :]; dla = dla[::-1]
        sdlat = abs(float(dla[0] - dla[1])); sdlon = abs(float(dlo[1] - dlo[0]))
        swest = float(dlo.min()) - sdlon / 2.0; snor = float(dla.max()) + sdlat / 2.0
        dlon = float(abs(lon[1] - lon[0])); dlat = float(abs(lat[1] - lat[0]))
        mwest = float(min(lon)) - dlon / 2.0; mnor = float(max(lat)) + dlat / 2.0
        dst = np.zeros(shape, dtype="float64")
        reproject(source=tot, destination=dst, src_transform=from_origin(swest, snor, sdlon, sdlat),
                  src_crs="EPSG:4326", dst_transform=from_origin(mwest, mnor, dlon, dlat),
                  dst_crs="EPSG:4326", resampling=Resampling.bilinear)
        return dst[::-1, :]                                          # north-up -> model ascending
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] deposition grid {year} load failed ({e}); using country-mean dep")
        return None


def load_landcover_fractions(cfg, shape, year=2022):
    """Per-cell IGBP class fractions from MODIS MCD12C1 (native 0.05deg CMG), nearest epoch.

    Reads the 'Land_Cover_Type_1_Percent' subdataset (17 IGBP-class % layers per 0.05deg
    cell) and aggregates to our categories via coefficients/landcover_classes.csv. Returns
    {category: fraction_2d in 0..1} on the model grid (lat ascending) or None if absent.
    """
    from pathlib import Path as _P
    ldir = _P(cfgmod.get_path(cfg, "raw")) / cfg["manure"].get("landcover_dir", "landcover")
    if not ldir.exists():
        return None
    hits = list(ldir.glob("*.hdf")) + list(ldir.glob("*.tif")) + list(ldir.glob("*.nc"))
    if not hits:
        return None
    try:
        import rasterio
        cls = pd.read_csv(coef(cfg, cfg["manure"]["landcover_classes_file"]))
        pick, epoch = _nearest_epoch(hits, year, r"A(\d{4})\d{3}")   # MCD12C1.A<YYYY>001...
        path = pick or str(hits[0])
        with rasterio.open(path) as src:
            sds = src.subdatasets
        pct_name = next((s for s in (sds or []) if "Land_Cover_Type_1_Percent" in s), None)
        if pct_name is None and not sds:
            pct_name = path                       # plain raster (already the percent stack)
        with rasterio.open(pct_name) as ds:
            arr = ds.read().astype("float32")     # (bands, H, W)
        if arr.shape[0] != 17:
            print(f"  [warn] MCD12C1 percent stack has {arr.shape[0]} bands (expected 17); "
                  "check band order in landcover_classes.csv")
        cats = {}
        for category, grp in cls.groupby("category"):
            if category in ("forest", "other"):
                continue
            bands = [b for b in grp["band_index"].astype(int) if b < arr.shape[0]]
            cats[category] = np.clip(arr[bands].sum(axis=0) / 100.0, 0, 1)
        # orient to model grid (MCD12C1 is north-up, lat descending) -> ascending
        out = {}
        for k, a in cats.items():
            if a.shape != shape:
                print(f"  [warn] land-cover grid {a.shape} != model {shape}; skipping LC")
                return None
            out[k] = a[::-1, :]
        return out
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] land-cover load failed ({e}); pasture pool will be skipped")
        return None


def _boxsum(a, r):
    """Sum of `a` over a (2r+1) x (2r+1) window centred on each cell (zero outside the array)."""
    a = np.asarray(a, "float64")
    c = np.pad(a, ((r + 1, r), (r + 1, r))).cumsum(0).cumsum(1)
    n0, n1 = a.shape; w = 2 * r + 1
    return c[w:w + n0, w:w + n1] - c[0:n0, w:w + n1] - c[w:w + n0, 0:n1] + c[0:n0, 0:n1]


def _cap_cropland_manure(man_crop, cropland_ha, mask, idx2iso, cap_rate=350.0, radius=5, n_local=4,
                         verbose=True):
    """Cap applied manure on cropland at `cap_rate` kg N per ha of cropland, keeping manure local.

    Excess above the ceiling is moved, within the same country only:
      1. LOCAL: to cropland with headroom (cap_rate*A - M > 0) within a (2*radius+1)^2 window of cells
         (radius 5 at 0.05 deg ~ 25-30 km), in proportion to headroom; repeated `n_local` times so mass
         that overfills a receiver moves on locally;
      2. NATIONAL: what is left goes to the country's remaining headroom, in proportion to headroom;
      3. ACCEPTED: if the country has no headroom left, the remainder is spread over its cropland in
         proportion to cropland area (accepted above the ceiling, so no N is lost).
    Each country's applied-manure total is conserved exactly. Vectorized over the cells that carry
    excess (one pass per window offset), so the cost scales with the number of capped cells, not with
    country extents. Returns (capped man_crop, stats dict)."""
    shape = np.shape(man_crop); ny, nx = shape
    M = np.nan_to_num(np.asarray(man_crop, "float64")).ravel().copy()
    A = np.clip(np.nan_to_num(np.asarray(cropland_ha, "float64")), 0.0, None).ravel()
    K = np.asarray(mask).ravel()
    capm = cap_rate * A
    E = np.clip(M - capm, 0.0, None)
    E[K < 0] = 0.0                                                  # cells outside any country: untouched
    tot_excess = E.sum()
    stats = {"capped_kg": tot_excess, "local_kg": 0.0, "national_kg": 0.0, "accepted_kg": 0.0,
             "cells_capped": int((E > 0).sum())}
    if tot_excess <= 0:
        return M.reshape(shape), stats
    M -= E
    tol = 1e-3                                                      # kg; headroom below 1 g is ignored
    offs = [(di, dj) for di in range(-radius, radius + 1) for dj in range(-radius, radius + 1)]
    for _ in range(n_local):                                        # 1. local window, same country
        src = np.flatnonzero(E > 0)
        if src.size == 0:
            break
        R = np.clip(capm - M, 0.0, None); R[R <= tol] = 0.0
        si, sj = np.divmod(src, nx); ks = K[src]
        D = np.zeros(src.size)
        cache = []
        for di, dj in offs:                                         # headroom reachable from each source
            ri = si + di; rj = sj + dj
            ok = (ri >= 0) & (ri < ny) & (rj >= 0) & (rj < nx)
            r = np.where(ok, ri * nx + rj, 0)
            ok &= (K[r] == ks)
            w = np.where(ok, R[r], 0.0)
            D += w
            cache.append((r, w))
        D[D <= tol] = 0.0
        send = D > 0
        if not send.any():
            break
        f = np.where(send, E[src] / np.where(send, D, 1.0), 0.0)
        recv = np.zeros_like(M)
        for r, w in cache:                                          # r is unique per offset -> direct add
            recv[r] += w * f
        E[src[send]] = 0.0
        take = np.minimum(recv, R)
        M += take
        E += recv - take                                            # overfill stays as excess at the receiver
    nidx = int(K.max()) + 1
    kk = np.where(K >= 0, K, 0)
    E_c = np.bincount(kk, weights=E, minlength=nidx)               # per-country excess after the local step
    stats["local_kg"] = tot_excess - E_c.sum()
    R = np.clip(capm - M, 0.0, None); R[(R <= tol) | (K < 0)] = 0.0  # 2. national headroom
    R_c = np.bincount(kk, weights=R, minlength=nidx)
    nat_c = np.minimum(E_c, R_c)
    with np.errstate(divide="ignore", invalid="ignore"):
        M += R * np.where(R_c > 0, nat_c / np.where(R_c > 0, R_c, 1.0), 0.0)[kk]
    acc_c = E_c - nat_c                                             # 3. accepted above the ceiling
    Aw = np.where(K >= 0, A, 0.0)
    A_c = np.bincount(kk, weights=Aw, minlength=nidx)
    with np.errstate(divide="ignore", invalid="ignore"):
        M += Aw * np.where(A_c > 0, acc_c / np.where(A_c > 0, A_c, 1.0), 0.0)[kk]
    lost = acc_c[A_c <= 0].sum()                                    # country with excess but no cropland
    if lost > 0:                                                    # (not expected): return it where it was
        for k in np.flatnonzero((acc_c > 0) & (A_c <= 0)):
            cells = np.flatnonzero(K == k)
            M[cells] += acc_c[k] / cells.size
    stats["national_kg"] = nat_c.sum(); stats["accepted_kg"] = acc_c.sum()
    if verbose:
        print(f"  [cap] cropland manure held under {cap_rate:.0f} kg N/ha: {stats['capped_kg']/1e9:.3f} Tg "
              f"above the ceiling in {stats['cells_capped']:,} cells -> {stats['local_kg']/1e9:.3f} Tg local "
              f"(<= {radius} cells), {stats['national_kg']/1e9:.3f} Tg national, "
              f"{stats['accepted_kg']/1e9:.3f} Tg accepted (no headroom)")
    return M.reshape(shape), stats


def _cap_pasture_n(fert_grass, man_past, broad_area, modis_grass_area, mask, idx2iso, cap_rate=350.0):
    """Cap the COMBINED grassland fertilizer + pasture-deposited manure at `cap_rate` (kg N ha-1 of
    grazing land). Regulatory grassland-N limits (e.g. the Netherlands ~265-320, EU Nitrates) are on
    total N, so the ceiling is applied to fertilizer+manure jointly. Where a cell exceeds it, the excess
    cascades WITHIN the country: broad-grazing headroom -> MODIS grassland extent -> broad area (accepted,
    so no N is lost). Each pool's national total is conserved. This guards the implausible loadings that
    arise where a satellite grassland product under-detects managed pasture (e.g. NLD, JPN). Returns the
    capped (fert_grass, man_past)."""
    shape = fert_grass.shape
    flat = mask.ravel().clip(min=0); nidx = int(mask.max()) + 1

    def agg(a):
        return np.bincount(flat, weights=np.asarray(a, dtype="float64").ravel(), minlength=nidx)

    F = np.asarray(fert_grass, dtype="float64").copy()
    M = np.asarray(man_past, dtype="float64").copy()
    capm = cap_rate * np.clip(broad_area, 0.0, None)                  # per-cell capacity (kg)
    C = F + M
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.where(C > capm + 1e-6, capm / np.maximum(C, 1e-12), 1.0)   # scale-down at over-cap cells
    remF = F * (1.0 - s); remM = M * (1.0 - s)                        # removed mass (>=0)
    F -= remF; M -= remM
    EF = agg(remF); EM = agg(remM); E = EF + EM                       # per-country removed N
    if E.sum() <= 0:
        return F, M                                                  # nothing over the ceiling
    shF = np.where(E > 0, EF / np.maximum(E, 1e-12), 0.0)            # per-country fert / manure share of moved N
    shM = np.where(E > 0, EM / np.maximum(E, 1e-12), 0.0)

    def _spill(resid, fillw, capacity):                              # place resid over fillw up to capacity
        fw = np.clip(np.asarray(fillw, "float64"), 0.0, None)
        place = np.minimum(resid, capacity)
        FWc = agg(fw)
        with np.errstate(divide="ignore", invalid="ignore"):
            rate = np.where(FWc > 0, place / np.maximum(FWc, 1e-12), 0.0)   # mass per unit fillw, per country
        add = (fw.ravel() * rate[flat]).reshape(shape)
        return add, (resid - place)

    def _apply(add):
        F[...] = F + add * shF[flat].reshape(shape)
        M[...] = M + add * shM[flat].reshape(shape)

    head = np.clip(capm - (F + M), 0.0, None)                        # stage 1: broad-grazing headroom
    add, resid = _spill(E, head, agg(head)); _apply(add)
    resid_pre_modis = resid.copy()                                   # residual entering the MODIS fallback
    modis_placed = np.zeros(nidx)
    if modis_grass_area is not None and np.any(modis_grass_area > 0):  # stage 2: MODIS grassland extent
        add, resid = _spill(resid, modis_grass_area, cap_rate * agg(modis_grass_area)); _apply(add)
        modis_placed = resid_pre_modis - resid                       # N actually placed onto MODIS, per country
    accepted = resid.copy()                                          # stage 3: what neither GPW nor MODIS could hold
    if np.any(resid > 1e-6):                                          # spread remainder on broad (accept over-cap)
        add, resid = _spill(resid, broad_area, np.full(nidx, np.inf)); _apply(add)
    eff = np.maximum(broad_area, modis_grass_area if modis_grass_area is not None else 0.0)  # broad U MODIS extent
    fr = agg(F + M) / np.maximum(agg(eff), 1e-9)                      # country rate over the extent it can use
    hot = sorted(((idx2iso.get(i), fr[i]) for i in range(nidx) if i in idx2iso and fr[i] > cap_rate * 1.01),
                 key=lambda x: -x[1])[:6]
    print(f"  [cap] pasture-N ceiling {cap_rate:.0f} kg/ha: moved {E.sum()/1e9:.3f} Tg; "
          f"{len(hot)} country(ies) still over"
          + (": " + ", ".join(f"{iso} {r:.0f}" for iso, r in hot) if hot else ""))
    # MODIS-fallback vs accept-over-cap diagnostics (did the cascade ever actually need MODIS?)
    mh = sorted(((idx2iso.get(i), modis_placed[i]) for i in range(nidx) if i in idx2iso and modis_placed[i] > 1e6),
                key=lambda x: -x[1])
    ah = sorted(((idx2iso.get(i), accepted[i]) for i in range(nidx) if i in idx2iso and accepted[i] > 1e6),
                key=lambda x: -x[1])
    print(f"  [cap]   MODIS fallback: {modis_placed.sum()/1e9:.3f} Tg onto {len(mh)} country(ies)"
          + (" [" + ", ".join(f"{iso} {t/1e9:.3f}Tg" for iso, t in mh[:8]) + "]" if mh else "")
          + f"; accepted over-cap (GPW+MODIS both saturated): {accepted.sum()/1e9:.3f} Tg, {len(ah)} country(ies)"
          + (" [" + ", ".join(f"{iso} {t/1e9:.3f}Tg" for iso, t in ah[:8]) + "]" if ah else ""))
    return F, M


def place_national_residual(target, national, all_ha, cropland_ha, mask, idx2iso, label,
                            names=("all-crop harvarea", "cropland"), rel_tol=1e-9, abs_tol_kg=1e-3):
    """Place national mass that the per-crop allocation could not place .

    Crop removal and BNF are allocated item by item onto each crop's CropGRIDS harvested area
    (× `s_env` in the fast path when the HYDE envelope is on). National mass is silently dropped when
    (i) the FAO item has no CropGRIDS layer / is not in the weight cache, or (ii) the crop has zero
    (s-scaled) harvested area in that country. This helper restores per-country conservation:

        R(k) = Σ_items N(k, item) − Σ_{m∈k} target(m)

    and, for every country with R(k) > tol, spreads R(k) over

    1. `all_ha`: all-crop harvested area (× s when the envelope is on), i.e. the same total weight
       that removal itself uses, summed over all crops; if that is zero in k,
    2. `cropland_ha`: cropland area of the run (year-t extent, s(m)·H(m,t) with s = CropGRIDS/HYDE 2020, when the envelope is on; CropGRIDS otherwise);
       if that is also zero, the mass is left unplaced and reported.

    **Arguments**

    - `target`: 2-D (or flat) array of the mass already placed (kg per cell); not modified.
    - `national`: DataFrame with columns `ISO3`, `value_kg` (one row per country × item; rows are
      summed per country) or a dict ISO3 → kg. Non-finite values are ignored (as in `allocate()`).
    - `all_ha`: array (2-D or flat) or a zero-argument callable returning it (evaluated only when a
      residual exists).
    - `cropland_ha`: array (2-D or flat), or None to skip fallback level 2.
    - `mask`, `idx2iso`: country mask (cells ≤ 0 are ocean/disputed) and index → ISO3 map.
    - `label`: name printed in the one-line diagnostic (e.g. "CROP", "BNF").
    - `names`: display names of the two fallback weights for the diagnostic.

    **Returns** `(placed, stats)`: `placed` has the shape of `target` and equals `target` plus the
    residual placement; `stats` is a dict of masses (kg) and lists of ISO3 codes. Countries whose
    |R(k)| ≤ `rel_tol`·N(k) + `abs_tol_kg` are untouched, so the output is identical to `target`
    when nothing was dropped. Over-placed countries (R(k) < −tol, not expected) are reported only.
    National ISO3 codes absent from `idx2iso` (or mapped to index ≤ 0) cannot be gridded; their mass
    is reported as `off_mask_kg` and does not raise."""
    flat_mask = np.asarray(mask).ravel()
    nidx = int(flat_mask.max()) + 1
    kflat = flat_mask.clip(min=0)
    land = flat_mask >= 1
    tgt = np.asarray(target, "float64")
    out = tgt.ravel().copy()
    iso2idx = {v: k for k, v in idx2iso.items()}

    # national total per mask index (Σ over items); codes not on the mask are reported
    nat = np.zeros(nidx, "float64")
    off = {}
    pairs = (zip(national["ISO3"], national["value_kg"]) if isinstance(national, pd.DataFrame)
             else national.items())
    for iso, v in pairs:
        if v is None or pd.isna(v) or not np.isfinite(float(v)):
            continue
        i = iso2idx.get(iso)
        if i is None or i < 1 or i >= nidx:
            off[iso] = off.get(iso, 0.0) + float(v)
            continue
        nat[i] += float(v)

    placed_by_idx = np.bincount(kflat[land], weights=out[land], minlength=nidx)
    resid = nat - placed_by_idx
    tol = rel_tol * np.abs(nat) + abs_tol_kg
    need = resid > tol
    over = resid < -tol

    m_A = m_C = 0.0
    left = need.copy()
    if need.any():
        A = all_ha() if callable(all_ha) else all_ha
        A = np.nan_to_num(np.asarray(A, "float64").ravel())
        A[~land | (A < 0)] = 0.0
        denA = np.bincount(kflat, weights=A, minlength=nidx)
        useA = need & (denA > 0)
        out += A * np.where(useA, resid / np.where(useA, denA, 1.0), 0.0)[kflat]
        m_A = float(resid[useA].sum())
        left = need & ~useA
        if left.any() and cropland_ha is not None:
            C = np.nan_to_num(np.asarray(cropland_ha, "float64").ravel())
            C[~land | (C < 0)] = 0.0
            denC = np.bincount(kflat, weights=C, minlength=nidx)
            useC = left & (denC > 0)
            out += C * np.where(useC, resid / np.where(useC, denC, 1.0), 0.0)[kflat]
            m_C = float(resid[useC].sum())
            left = left & ~useC

    stats = dict(residual_kg=float(resid[need].sum()), n_countries=int(need.sum()),
                 to_all_ha_kg=m_A, to_cropland_kg=m_C, unplaced_kg=float(resid[left].sum()),
                 unplaced_iso=[idx2iso.get(int(i), str(i)) for i in np.nonzero(left)[0]],
                 over_kg=float(-resid[over].sum()),
                 over_iso=[idx2iso.get(int(i), str(i)) for i in np.nonzero(over)[0]],
                 off_mask_kg=float(sum(off.values())), off_mask_iso=sorted(off))

    def _lst(x, n=10):
        return ", ".join(x[:n]) + (f", ... (+{len(x) - n})" if len(x) > n else "")
    msg = (f"  {label} residual: {stats['residual_kg']/1e9:.4f} Tg in {stats['n_countries']} country(ies) "
           f"-> {m_A/1e9:.4f} Tg {names[0]}, {m_C/1e9:.4f} Tg {names[1]}, "
           f"{stats['unplaced_kg']/1e9:.4f} Tg unplaced")
    if stats["unplaced_iso"]:
        msg += f" [{_lst(stats['unplaced_iso'])}]"
    if off:
        msg += f"; {stats['off_mask_kg']/1e9:.4f} Tg on ISO3 not in mask [{_lst(stats['off_mask_iso'])}]"
    if stats["over_iso"]:
        msg += (f"; [warn] over-placed {stats['over_kg']/1e9:.4f} Tg in {len(stats['over_iso'])} "
                f"country(ies) left as is [{_lst(stats['over_iso'])}]")
    print(msg)
    return out.reshape(tgt.shape), stats


def run_downscale(cfg, year, nutrient="N", max_crops=None, only_slugs=None):
    """Single pass over CropGRIDS crops -> gridded crop removal, fertilizer weight, BNF,
    and cropland-area; then deposition (by cropland) and combine into a surplus raster.
    Returns dict of 2d arrays + lat/lon. Mass-conserving per country."""
    import xarray as xr
    from . import fertilizer
    cg = Path(cfgmod.get_path(cfg, "raw")) / CG_SUBDIR
    npk = Path(cfgmod.get_path(cfg, "raw")) / NPK_SUBDIR
    mask, idx2iso, lat, lon = load_country_grid(cfg)
    shape = mask.shape

    xw = pd.read_csv(coef(cfg, "cropgrids_item_crosswalk.csv"))
    xw = xw[xw["has_cropgrids_file"]] if "has_cropgrids_file" in xw else xw
    if only_slugs:
        xw = xw[xw["slug"].isin(only_slugs)]
    if max_crops:
        xw = xw.head(max_crops)
    crop_nat = national_crop_removal(cfg, year, nutrient)
    bnf_nat = national_bnf_by_item(cfg, year) if nutrient == "N" else pd.DataFrame(columns=["ISO3","fao_item_code","value_kg"])

    crop_grid = np.zeros(shape, "float64")
    bnf_grid = np.zeros(shape, "float64")
    fert_w = np.zeros(shape, "float64")     # Σ harvarea*Nrate (fertilizer intensity weight)
    cropland_ha = np.zeros(shape, "float64")
    allcrop_ha = np.zeros(shape, "float64")  # Σ_crops harvarea: residual-placement weight (fallback 1)
    nrate_var = {"N": "Nrate", "P": "P2O5rate"}[nutrient]
    p2o5 = cfg["conversions"]["p2o5_to_p"] if nutrient == "P" else 1.0
    # coello_cropgrids: collect CropGRIDS harvarea per Coello crop group from the SAME crop loop that
    # grids removal (sparse, only when that mode is selected). This path has no HYDE envelope, so the
    # group weight is H_g alone, exactly as removal here uses harvarea alone.
    coe_cg = _fert_grid_weight(cfg) == "coello_cropgrids"
    if coe_cg:
        coe_slug2grp, coe_forage = _coello_crop_groups(cfg)
    coe_parts, coe_present = {}, []

    for slug, code in zip(xw["slug"], xw["fao_item_code"]):
        ha = _layer(cg, slug, "harvarea")
        if ha is None:
            continue
        allcrop_ha += ha
        if coe_cg:
            coe_present.append(slug)
            _g = coe_slug2grp.get(slug)
            if _g is not None:
                _haf = ha.ravel(); _nz = np.nonzero(_haf > 0)[0]
                coe_parts.setdefault(_g, []).append((_nz.astype("int32"), _haf[_nz].astype("float32")))
        ca = _layer(cg, slug, "croparea")
        if ca is not None:
            cropland_ha += np.nan_to_num(ca)
        cs = crop_nat[crop_nat["fao_item_code"] == int(code)]
        if not cs.empty:
            crop_grid += allocate(cs[["ISO3", "value_kg"]], ha, mask, idx2iso, "value_kg")
        bs = bnf_nat[bnf_nat["fao_item_code"] == int(code)]
        if not bs.empty:
            bnf_grid += allocate(bs[["ISO3", "value_kg"]], ha, mask, idx2iso, "value_kg")
        nr = _layer(npk, slug, nrate_var)
        if nr is not None:
            fert_w += np.nan_to_num(ha) * np.nan_to_num(nr) * p2o5

    # Residual placement : national removal/BNF of items with no CropGRIDS layer, or of
    # crops with no harvarea in a country, would otherwise be dropped. Spread each country's residual
    # over its all-crop harvarea, then CropGRIDS cropland (this path has no HYDE envelope). Skipped for
    # partial debug runs (only_slugs / max_crops), where the residual would be mostly the excluded crops.
    if only_slugs or max_crops:
        print("  CROP/BNF residual placement skipped (partial crop set: only_slugs/max_crops)")
    else:
        crop_grid, _ = place_national_residual(crop_grid, crop_nat[["ISO3", "value_kg"]], allcrop_ha,
                                               cropland_ha, mask, idx2iso, "CROP",
                                               names=("all-crop harvarea", "CropGRIDS cropland"))
        if nutrient == "N":
            bnf_grid, _ = place_national_residual(bnf_grid, bnf_nat[["ISO3", "value_kg"]], allcrop_ha,
                                                  cropland_ha, mask, idx2iso, "BNF",
                                                  names=("all-crop harvarea", "CropGRIDS cropland"))
    del allcrop_ha

    # fertilizer weight, selected by config fertilizer.grid_weight. National magnitude is FAOSTAT-RFN
    # x our CF in every case (allocate() renormalizes the weight per country), so the choice only sets
    # WITHIN-country geography.
    #   coello_cropgrids (default): Coello et al. (2025) national split among 13 crop
    #     groups, F_C(g,k,t), each group placed by CropGRIDS harvarea H_g -- the SAME area basis as crop
    #     removal, so fertilizer and removal share one crop map: W = Σ_g F_C(g,k)·H_g/Σ_k H_g. Groups
    #     with no area in a country -> all-crop harvarea -> cropland; countries with no Coello ->
    #     NPKGRIDS (fert_w). If the 12d table is missing, falls through to the chain below.
    #   coello (alternative, not used): Coello 5-arcmin dasymetric field on Coello's own crop map.
    #   then Adalibieke (N) -> Lu&Tian rate x cropland (N, 1900-2013) -> NPKGRIDS intensity.
    fert_nat = fertilizer.compute(cfg, cfgmod.get_path(cfg, "raw"), year, nutrient).rename(columns={"fert_kg": "value_kg"})
    coe_tbl = load_coello_group_table(cfg, year, nutrient)
    coe = load_coello_fert(cfg, shape, lat, lon, year, nutrient) if coe_tbl is None else None
    ada = (load_adalibieke_fert(cfg, shape, lat, lon, year)
           if (nutrient == "N" and coe is None and coe_tbl is None) else None)
    lt = (load_lu_tian_fert(cfg, shape, lat, lon, year)
          if (nutrient == "N" and coe is None and ada is None and coe_tbl is None) else None)
    if coe_tbl is not None:
        _report_coello_xwalk_gaps(coe_slug2grp, coe_forage, coe_present)
        coe_ha = _group_ha_from_parts(coe_parts)
        W = coello_cropgrids_weight(coe_tbl, coe_ha, None, cropland_ha, None, mask, idx2iso)
        fert_weight = _fill_zero_weight_countries(W, fert_w, mask)     # no-Coello countries -> NPKGRIDS
    elif coe is not None:
        # Coello is already a per-cell MASS (density x cell area) carrying its own annual cropland
        # extent -> use directly, never x cropland_ha. NPKGRIDS backfills countries it misses.
        fert_weight = _fill_zero_weight_countries(coe, fert_w, mask)
    elif ada is not None:
        fert_weight = ada
    elif lt is not None:
        fert_weight = lt * cropland_ha
    else:
        fert_weight = fert_w
    fert_grid = allocate(fert_nat, fert_weight, mask, idx2iso, "value_kg")

    # deposition: distribute national total-N dep by the native ISIMIP gradient (conserves to the
    # country table); land-view deposition (all land) computed below with cell area.
    from . import deposition as depmod
    nidx = int(mask.max()) + 1
    dep_grid = np.zeros(shape, "float64")
    dep_w = None; dep_tbl = None
    if nutrient == "N":
        dep_tbl = depmod.dep_kgha_for_year(cfg, year)      # per-year (clamped) national rate (kg/ha)
        if dep_tbl is not None:
            dep_w = load_deposition_grid(cfg, shape, lat, lon, year)   # native gradient (kg/ha) or None
            cl_by_idx = np.bincount(mask.ravel().clip(min=0), weights=cropland_ha.ravel(), minlength=nidx)
            iso2cl = {idx2iso[i]: cl_by_idx[i] for i in idx2iso if i < len(cl_by_idx)}
            dnat = dep_tbl.copy()
            dnat["value_kg"] = dnat.apply(lambda r: r["dep_ox_kgha"] * iso2cl.get(r["ISO3"], 0.0), axis=1)
            # Cropland deposition: national MAGNITUDE from Vishwakarma WL, W(i,t)*A_crop,i (paper Eq. 2),
            # within-country PATTERN from the ISIMIP3a total-N rate (previously uniform W(i,t)).
            # The weight is I(m,t)*A_crop(m), so the national total is unchanged.
            w_crop = cropland_dep_weight(dep_w, cropland_ha, mask,
                                         cfg.get("deposition", {}).get("cropland_pattern", "isimip"))
            dep_grid = allocate(dnat[["ISO3", "value_kg"]], w_crop, mask, idx2iso, "value_kg")

    # human waste: allocate national HUM by gridded population (fallback: cropland area)
    from . import human as humanmod
    hum_grid = np.zeros(shape, "float64")            # hum_sewage: environmental discharge (total only)
    food_waste_grid = np.zeros(shape, "float64")     # consumer food-waste N (landscape/total input)
    hum_ns_grid = np.zeros(shape, "float64")         # night soil -> ALL cropland (no urban/rural bias)
    hum_bio_grid = np.zeros(shape, "float64")        # biosolids -> peri-urban cropland
    hum_landfill_grid = np.zeros(shape, "float64")   # landfilled sludge -> retained in landscape (near people)
    hum_water_grid = np.zeros(shape, "float64")      # sewered discharge -> surface water (total only)
    hum_land_grid = np.zeros(shape, "float64")       # non-sewered discharge -> land (landscape + total)
    try:
        _raw = cfgmod.get_path(cfg, "raw")
        hum_nat = humanmod.compute(cfg, _raw, year, nutrient)   # hum_sewage_kg + hum_sew_water/land_kg
        popw = load_population_grid(cfg, shape, year)
        wlayer = popw if popw is not None else cropland_ha
        # SEWERED discharge (treated effluent + untreated collected) -> URBAN population (people on sewers are
        # urban). NON-SEWERED land discharge (HUM_terr,nc) -> TOTAL population, like food waste and landfilled
        # sludge (Eq. 63): pit-latrine/septic/open-defecation N is deposited where people live, not only rurally.
        if popw is not None and str(cfg["human"].get("sewer_grid_mode", "rural")).lower() != "none":
            dla = abs(float(lat[1] - lat[0])); dlo = abs(float(lon[1] - lon[0]))
            urb = _urban_fraction(popw, lat, dla, dlo)
            w_water = popw * urb
        else:
            w_water = wlayer
        if "hum_sew_water_kg" in hum_nat:
            hum_water_grid = allocate(hum_nat.rename(columns={"hum_sew_water_kg": "value_kg"})[["ISO3", "value_kg"]],
                                      w_water, mask, idx2iso, "value_kg")
            hum_land_grid = allocate(hum_nat.rename(columns={"hum_sew_land_kg": "value_kg"})[["ISO3", "value_kg"]],
                                     wlayer, mask, idx2iso, "value_kg")     # HUM_terr,nc -> population
        else:                                                    # P/fallback: only the total, on population
            hum_water_grid = allocate(hum_nat.rename(columns={"hum_sewage_kg": "value_kg"})[["ISO3", "value_kg"]],
                                      wlayer, mask, idx2iso, "value_kg")
        hum_grid = hum_water_grid + hum_land_grid                # total discharge (reference/Beusen)
        print(f"  HUM(sewage) split: sewered->urban (surface water) + non-sewered->rural (land)")
        # consumer food-waste N also follows people (landfill/compost): allocate by population
        fw_nat = humanmod.food_waste(cfg, _raw, year, nutrient).rename(columns={"food_waste_kg": "value_kg"})
        if len(fw_nat):
            food_waste_grid = allocate(fw_nat[["ISO3", "value_kg"]], wlayer, mask, idx2iso, "value_kg")
        # recycled human excreta -> CROPLAND, split by pathway with opposite spatial signatures:
        # night soil -> RURAL cropland (1-urb); biosolids -> PERI-URBAN cropland (urb). Gridded separately.
        hr_nat = humanmod.human_recycle(cfg, _raw, year, nutrient)
        if len(hr_nat):
            w_ns, w_bio = _recycle_weights(cfg, cropland_ha, popw, lat, lon)
            hum_ns_grid = allocate(hr_nat.rename(columns={"hum_ns_kg": "value_kg"})[["ISO3", "value_kg"]],
                                   w_ns, mask, idx2iso, "value_kg")
            hum_bio_grid = allocate(hr_nat.rename(columns={"hum_bio_kg": "value_kg"})[["ISO3", "value_kg"]],
                                    w_bio, mask, idx2iso, "value_kg")
            if "hum_landfill_kg" in hr_nat:                       # landfilled sludge -> landscape (near people)
                hum_landfill_grid = allocate(hr_nat.rename(columns={"hum_landfill_kg": "value_kg"})[["ISO3", "value_kg"]],
                                             popw, mask, idx2iso, "value_kg")
            print(f"  hum_crop weight = night soil->rural, biosolids->peri-urban cropland "
                  f"(sewer_grid_mode={cfg['human'].get('sewer_grid_mode', 'rural')})")
    except utils.MissingSourceFile:
        pass
    human_recycle_grid = hum_ns_grid + hum_bio_grid   # hum_crop = night soil + biosolids (-> cropland)

    # cell land area (ha) from cos(latitude), for the per-land-ha (NANI) view
    R = 6371000.0
    dlat = float(abs(lat[1] - lat[0])); dlon = float(abs(lon[1] - lon[0]))
    latr = np.deg2rad(lat)
    row_area_m2 = (R**2) * np.deg2rad(dlon) * (np.sin(latr + np.deg2rad(dlat)/2)
                                               - np.sin(latr - np.deg2rad(dlat)/2))
    cell_area_ha = np.abs(np.repeat(row_area_m2[:, None], shape[1], axis=1) / 1e4)
    only_land = mask >= 1

    # ---- MANURE: two pools, each to its own land base, weighted by GLW excretion density ----
    #   applied-to-soils (72381) -> CROPLAND             (man_crop; View 1 + View 2)
    #   left-on-pasture (72380)  -> GRAZING land (MODIS)  (man_past_*; View 2 only)
    # Grazing land is built as two separate bases: pasture (grassland) and rangeland
    # (grassland+shrub+savanna) -> two alternative downscalings of the same pasture pool.
    from . import manure as manuremod
    raw = cfgmod.get_path(cfg, "raw")
    # Applied/pasture spatial split of livestock density -> two weights (lw_conf=applied/cropland,
    # lw_graz=pasture/grazing). Preferred: data-driven per-species applied fraction a_s(i,t) from
    # FAOSTAT EMN (applied_fraction_from_emn); each species is routed to the applied vs pasture
    # weight by its own country-varying applied share, superseding the single global cattle split.
    # Fallback: species-list split with a global cattle_confined_fraction.
    if cfg["manure"].get("applied_fraction_from_emn", False):
        lw_conf, lw_graz = _applied_pasture_density(cfg, shape, lat, lon, year, mask, idx2iso)
    else:
        lw_conf = lw_graz = None
    if lw_conf is None and lw_graz is None:
        # cattle split: fraction f_conf of cattle is confined (managed manure -> applied/cropland
        # weight), the remainder grazing. f_conf=0 reproduces the pigs+poultry-only behaviour.
        fconf = float(cfg["manure"].get("cattle_confined_fraction", 0.0))
        cattle = cfg["manure"].get("cattle_code", "CTL")
        conf_codes = list(cfg["manure"].get("confined_species", [])) + ([cattle] if fconf > 0 else [])
        graz_codes = list(cfg["manure"].get("grazing_species", []))
        lw_conf = load_livestock_grid(cfg, shape, lat, lon, year, codes=conf_codes,
                                      frac={cattle: fconf})          # pigs/poultry + f_conf*cattle -> applied
        lw_graz = load_livestock_grid(cfg, shape, lat, lon, year, codes=graz_codes,
                                      frac={cattle: 1.0 - fconf})    # ruminants (cattle x (1-f_conf)) -> pasture
    man_crop = np.zeros(shape, "float64")
    man_past = np.zeros(shape, "float64")           # pasture-deposited manure (PRP) -> BROAD grazing base
    forage = np.zeros(shape, "float64")             # grassland forage removal -> BROAD grazing base
    forage_weight = None                            # grazing proxy used for the product term
    fo_nat = None                                   # forage national totals, hoisted for the Zhang override
    bnf_grass_grid = np.zeros(shape, "float64")     # pasture/forage-legume BNF -> BROAD grazing land (pasture+rangeland)
    past_base = None                                # narrow grazing base (managed pasture): fertilizer only
    rng_base = None                                 # broad grazing base (pasture+rangeland): PRP + forage + BNF_grass

    try:                                            # applied-to-soils -> cropland (confined species)
        ap_nat = manuremod.applied(cfg, raw, year).rename(columns={"manure_kg": "value_kg"})
        man_w = (lw_conf * cropland_ha) if lw_conf is not None else cropland_ha
        man_crop = allocate(ap_nat[["ISO3", "value_kg"]], man_w, mask, idx2iso, "value_kg")
        print(f"  MAN applied->cropland ({'confined-GLWxcropland' if lw_conf is not None else 'cropland only'})")
    except utils.MissingSourceFile:
        pass

    use_pool = cfg["manure"].get("pasture_pool", True)
    gpw_pasture, gpw_graze = load_gpw_grazing(cfg, shape, lat, lon, year) if use_pool else (None, None)
    lc = load_landcover_fractions(cfg, shape) if use_pool else None
    if lc is not None or gpw_graze is not None:
        try:                                        # left-on-pasture -> grazing land
            pa_nat = manuremod.pasture(cfg, raw, year).rename(columns={"manure_kg": "value_kg"})

            def _base(cats):
                frac = np.zeros(shape, "float64")
                if lc is not None:
                    for c in cats:
                        if c in lc:
                            frac += lc[c]
                return frac * cell_area_ha          # grazing-land area (ha) per cell (MODIS fallback)

            # GPW (30 m) is the primary grazing base: narrow = managed pasture (cultivated), broad =
            # pasture + rangeland. Falls back to the MODIS land-cover base when no GPW export is present.
            past_base = gpw_pasture if gpw_pasture is not None else _base(cfg["manure"]["grazing_base_pasture"])
            rng_base = gpw_graze if gpw_graze is not None else _base(cfg["manure"]["grazing_base_rangeland"])
            wr = (lw_graz * rng_base) if lw_graz is not None else rng_base  # grazing weight on BROAD base
            man_past = allocate(pa_nat[["ISO3", "value_kg"]], wr, mask, idx2iso, "value_kg")   # PRP -> broad
            print(f"  MAN pasture->broad grazing land ({'GPW' if gpw_graze is not None else 'MODIS'} pasture+rangeland)")
            if nutrient == "N":                     # forage + pasture-legume BNF -> broad grazing land
                try:
                    from . import forage as foragemod, country_areas as ca_mod, bnf as bnfmod
                    fo_areas = ca_mod.load_areas(cfg, raw, year)
                    fo_nat = None            # forage is built per cell after the cap (grid-scale Eq. 29)
                    forage_weight = wr       # grazing proxy for the product term
                    bg_nat = bnfmod.grassland(cfg, raw, year, fo_areas).rename(columns={"bnf_grass_kg": "value_kg"})
                    # Pasture-legume BNF -> BROAD base: the national magnitude is PASTURE_HA(perm.
                    # meadows+pastures) x a whole-pasture-average rate calibrated to Herridge over that
                    # broad area, so it must be placed over pasture+rangeland (recovers the uniform rate).
                    bnf_grass_grid = allocate(bg_nat[["ISO3", "value_kg"]], rng_base, mask, idx2iso, "value_kg")
                except Exception as e:  # noqa: BLE001
                    print(f"  [warn] forage/grassland-BNF skipped ({e})")
        except utils.MissingSourceFile:
            pass
    else:
        print("  MAN pasture pool skipped (MODIS land cover not in data/raw/landcover)")

    # land/NANI view extras: deposition on ALL land + the (1-CF) grassland fertilizer.
    # PIECEWISE deposition for the landscape/total views (country-scale
    #   DEP_land_i = W_i * A_crop_i + Σ_c I_c * (a_c - g_c) ):
    # cropland keeps the Vishwakarma WL magnitude (= dep_grid, which conserves W_i*A_crop_i),
    # and NON-cropland receives the NATIVE ISIMIP rate I_c directly (NOT rescaled to the cropland
    # value). The only conserved national total is the Vishwakarma cropland deposition; the
    # non-cropland N is ISIMIP's own gridded field integrated over the non-cropland area (a_c - g_c).
    dep_land_grid = dep_grid.copy()                 # fallback = cropland deposition only
    fert_grass = np.zeros(shape, "float64")
    if nutrient == "N" and dep_tbl is not None:
        if dep_w is not None:                        # I_c available -> piecewise
            noncrop_ha = np.clip(cell_area_ha - cropland_ha, 0.0, None)     # a_c - g_c (non-cropland area)
            dep_land_grid = dep_grid + dep_w * noncrop_ha                   # Wang(cropland) + ISIMIP(non-crop)
            dep_land_grid = np.where(mask >= 0, dep_land_grid, 0.0)         # land of countries only 
        else:                                        # no ISIMIP grid -> degrade to cropland rate over all land
            la_by_idx = np.bincount(mask.ravel().clip(min=0), weights=cell_area_ha.ravel(), minlength=nidx)
            iso2la = {idx2iso[i]: la_by_idx[i] for i in idx2iso if i < len(la_by_idx)}
            dep_l = dep_tbl.copy()
            dep_l["value_kg"] = dep_l.apply(lambda r: r["dep_ox_kgha"] * iso2la.get(r["ISO3"], 0.0), axis=1)
            dep_land_grid = allocate(dep_l[["ISO3", "value_kg"]], cell_area_ha, mask, idx2iso, "value_kg")
    try:                                            # grassland fertilizer -> NARROW managed pasture (GPW cultivated)
        gf = fertilizer.grassland(cfg, raw, year, nutrient).rename(columns={"fert_kg": "value_kg"})
        base_narrow = past_base if past_base is not None else cropland_ha
        base_broad = rng_base if rng_base is not None else cropland_ha
        graz_w = _fill_zero_weight_countries(base_narrow, base_broad, mask, "FERTgrass")   # narrow -> broad if empty
        fert_grass = allocate(gf[["ISO3", "value_kg"]], graz_w, mask, idx2iso, "value_kg")
    except Exception:  # noqa: BLE001
        pass

    # synthetic-fertilizer NH3 volatilization loss (soil-surface basis): fert x FracGASF(country)
    fert_vol = np.zeros(shape, "float64"); fert_vol_land = np.zeros(shape, "float64")
    if nutrient == "N":
        try:
            from . import fert_volatilization as fvmod
            fg = fvmod.fracgasf_by_country(cfg, raw, year)
            iso2fg = dict(zip(fg["ISO3"], fg["fracgasf"]))
            fg_by_idx = np.array([iso2fg.get(idx2iso.get(i, ""), fvmod.DEFAULT_FRACGASF)
                                  for i in range(nidx)], dtype="float64")
            fg_cell = fg_by_idx[mask.clip(min=0)]
            fert_vol = fert_grid * fg_cell
            fert_vol_land = (fert_grid + fert_grass) * fg_cell
        except Exception as e:  # noqa: BLE001
            print(f"  [warn] fert volatilization skipped ({e})")

    # --- Optional: use the Tian-group gridded manure PATTERN (Zhang cropland-applied + Xu grassland
    # deposition/application) in place of the EMN/AGLW livestock-density weight. WEIGHT-ONLY: the
    # FAOSTAT EMN national totals are still enforced by allocate() (Zhang/Xu are FAOSTAT-anchored, so
    # their distribution is consistent with those magnitudes). Enabled by manure.grid_source=='zhang'.
    if nutrient == "N" and str(cfg.get("manure", {}).get("grid_source", "emn")).lower() == "zhang":
        _rawm = cfgmod.get_path(cfg, "raw")
        zc, zc_placed = load_zhang_manure(cfg, shape, lat, lon, year, "crop")   # crop = pre-placed 0.05deg
        zg, zg_placed = load_zhang_manure(cfg, shape, lat, lon, year, "grass")   # grass: placed (GPW) or 5' density
        if zc is None or zg is None:                                   # GUARD: don't silently fall back to AGLW
            raise RuntimeError(
                f"manure.grid_source='zhang' but the Zhang derived grids for {year} are missing "
                f"(expected data/raw/zhang_manure/derived/zhang_man_{{crop,grass}}_*_0p05.nc). "
                f"Download Zhang et al. 2017 (PANGAEA 10.1594/PANGAEA.871980: ManNitPro + ManNitProCrpRd) "
                f"and run scripts/12c_build_zhang_manure.py first. Refusing to fall back to the AGLW weight.")
        wcrop = zc if zc_placed else zc * cropland_ha                  # placed field used directly (no x cropland_ha)
        wcrop = _fill_zero_weight_countries(wcrop, cropland_ha, mask, "MAN")
        apn = manuremod.applied(cfg, _rawm, year).rename(columns={"manure_kg": "value_kg"})
        man_crop = allocate(apn[["ISO3", "value_kg"]], wcrop, mask, idx2iso, "value_kg")
        gbase = rng_base if rng_base is not None else cropland_ha      # broad grazing base (only for zero-weight fill)
        wgrass_raw = zg if zg_placed else _zhang_grass_weight(zg, lat, lon)     # GPW-placed used directly
        wgrass = _fill_zero_weight_countries(wgrass_raw, gbase, mask, "MAN")
        pan = manuremod.pasture(cfg, _rawm, year).rename(columns={"manure_kg": "value_kg"})
        man_past = allocate(pan[["ISO3", "value_kg"]], wgrass, mask, idx2iso, "value_kg")
        forage_weight = wgrass                    # product term shares the pasture-manure (Zhang) proxy

    # cap combined grassland fertilizer + pasture-deposited manure at the pasture-N ceiling (guards the
    # implausible loadings where GPW under-detects managed pasture); excess cascades broad -> MODIS -> broad.
    _cap = cfg.get("manure", {}).get("pasture_n_cap_kgha")
    if nutrient == "N" and _cap and rng_base is not None:
        _modis_g = (lc["grassland"] * cell_area_ha) if (lc is not None and "grassland" in lc) else None
        fert_grass, man_past = _cap_grassland_fert(fert_grass, man_past, rng_base, _modis_g, mask, idx2iso, float(_cap))

    # grid-scale Eq. 29: FORAGE per cell = min(PRP + grazing-product N, two-band pasture supply)
    if nutrient == "N" and rng_base is not None and past_base is not None:
        try:
            forage = _forage_gridscale(cfg, raw, year, man_past, forage_weight,
                                       past_base, rng_base, mask, idx2iso)
        except Exception as e:  # noqa: BLE001
            print(f"  [warn] grid-scale forage skipped ({e})")

    # manure NH3 volatilization loss (soil-surface basis): applied manure x FracGASM, plus the
    # pasture-deposited pool x FracGASM(pasture) for the land view only.
    man_vol = np.zeros(shape, "float64")
    man_vol_past = np.zeros(shape, "float64")
    if nutrient == "N":
        try:
            from . import man_volatilization as mvmod
            fga = mvmod.fracgasm(cfg, "applied"); fgp = mvmod.fracgasm(cfg, "pasture")
            man_vol = man_crop * fga                       # applied-to-cropland manure loss
            man_vol_past = man_past * fgp                  # grazing-deposited (PRP) loss, broad grazing base
        except Exception as e:  # noqa: BLE001
            print(f"  [warn] manure volatilization skipped ({e})")

    # VIEW 1 — cropland N surplus per ha of CROPLAND, net NH3. The FAO-comparable variant excludes the
    # human recycled term (matches the FAO CNB input set); the headline cropland surplus adds hum_crop
    # (night soil + biosolids applied to cropland).
    ag_surplus_mass_faocomp = fert_grid + bnf_grid + dep_grid + man_crop - crop_grid - fert_vol - man_vol
    ag_surplus_mass = ag_surplus_mass_faocomp + human_recycle_grid
    with np.errstate(divide="ignore", invalid="ignore"):
        surplus_cropland_kgha = np.where(cropland_ha > 0, ag_surplus_mass / cropland_ha, np.nan)
        surplus_cropland_faocomp_kgha = np.where(cropland_ha > 0, ag_surplus_mass_faocomp / cropland_ha, np.nan)

    # LAND-SURFACE view — per ha of LAND, everything but human waste: full fertilizer (cropland +
    # grassland) net of NH3, deposition on all land, applied + pasture manure net of NH3. Two grazing
    # variants. Adding human waste (hum_grid) gives the FULL view.
    # landscape now INCLUDES non-sewered human discharge (hum_land_grid, deposited on land); the total
    # additionally adds only the SEWERED discharge to surface water (hum_water_grid).
    land_base = ((fert_grid + fert_grass) + bnf_grid + bnf_grass_grid + dep_land_grid + man_crop
                 + food_waste_grid + human_recycle_grid + hum_land_grid + hum_landfill_grid
                 - crop_grid - fert_vol_land - man_vol)
    # single grazing base: fertilizer + pasture-legume BNF on NARROW managed pasture (in land_base
    # via fert_grass/bnf_grass_grid); pasture-deposited manure (PRP) + forage removal on BROAD
    # pasture+rangeland. One landscape surplus (the pasture/rangeland dual variants are retired).
    land_mass = land_base + man_past - man_vol_past - forage
    full_mass = land_mass + hum_water_grid
    with np.errstate(divide="ignore", invalid="ignore"):
        _ok = only_land & (cell_area_ha > 0)
        surplus_landscape_kgha = np.where(_ok, land_mass / cell_area_ha, np.nan)
        surplus_full_kgha = np.where(_ok, full_mass / cell_area_ha, np.nan)

    return dict(crop=crop_grid, bnf=bnf_grid, fert=fert_grid, dep=dep_grid,
                man_crop=man_crop, man_past=man_past,
                hum_sewage=hum_grid, hum_sew_water=hum_water_grid, hum_sew_land=hum_land_grid,
                food_waste=food_waste_grid, hum_crop=human_recycle_grid,
                hum_ns=hum_ns_grid, hum_bio=hum_bio_grid, hum_landfill=hum_landfill_grid,
                dep_land=dep_land_grid, fert_grass=fert_grass, fert_vol=fert_vol, man_vol=man_vol,
                man_vol_past=man_vol_past, forage=forage, bnf_grass=bnf_grass_grid,
                cropland_ha=cropland_ha, cell_area_ha=cell_area_ha,
                ag_surplus_mass=ag_surplus_mass,
                surplus_cropland_kgha=surplus_cropland_kgha,
                surplus_cropland_faocomp_kgha=surplus_cropland_faocomp_kgha,
                surplus_landscape_kgha=surplus_landscape_kgha, surplus_full_kgha=surplus_full_kgha,
                mask=mask, idx2iso=idx2iso, lat=lat, lon=lon)


# ============================================================================
# Weight cache + fast per-year path (for the 1961-2023 historical run)
# ============================================================================
# The CropGRIDS crop patterns, NPKGRIDS fertilizer intensity, and the country mask are
# IDENTICAL every year (only national magnitudes change). We read them ONCE into a compact
# cache, then each year does only the cheap allocation math + the (epoch-memoized) proxies.
# run_downscale_fast reproduces run_downscale numerically; it is purely a speedup.

def _cache_dir(cfg):
    return Path(cfgmod.get_path(cfg, "processed")) / "downscale_cache"


def build_weight_cache(cfg, nutrient="N", max_crops=None):
    """Read CropGRIDS/NPKGRIDS once; store crop harvested-area as a CSR sparse stack plus the
    fixed cropland_ha, fert_w, mask, lat, lon. ~one-time cost equal to a single year's run."""
    import json
    raw = cfgmod.get_path(cfg, "raw")
    cg = Path(raw) / CG_SUBDIR; npk = Path(raw) / NPK_SUBDIR
    mask, idx2iso, lat, lon = load_country_grid(cfg)
    shape = mask.shape
    nrate_var = {"N": "Nrate", "P": "P2O5rate"}[nutrient]
    xw = pd.read_csv(coef(cfg, "cropgrids_item_crosswalk.csv"))
    xw = xw[xw["has_cropgrids_file"]] if "has_cropgrids_file" in xw else xw
    if max_crops:
        xw = xw.head(max_crops)
    cropland_ha = np.zeros(shape, "float64")
    fert_w = np.zeros(shape, "float64")
    cells_l, ha_l, ptr, codes = [], [], [0], []
    for slug, code in zip(xw["slug"], xw["fao_item_code"]):
        ha = _layer(cg, slug, "harvarea")
        if ha is None:
            continue
        ca = _layer(cg, slug, "croparea")
        if ca is not None:
            cropland_ha += np.nan_to_num(ca)
        nr = _layer(npk, slug, nrate_var)
        if nr is not None:
            fert_w += np.nan_to_num(ha) * np.nan_to_num(nr)
        haf = ha.ravel()
        nz = np.nonzero(haf > 0)[0]
        cells_l.append(nz.astype("int32")); ha_l.append(haf[nz].astype("float32"))
        ptr.append(ptr[-1] + nz.size); codes.append(int(code))
    cd = _cache_dir(cfg); cd.mkdir(parents=True, exist_ok=True)
    np.savez(cd / "weights.npz",
             cells=np.concatenate(cells_l) if cells_l else np.array([], "int32"),
             ha=np.concatenate(ha_l) if ha_l else np.array([], "float32"),
             ptr=np.array(ptr, "int64"), codes=np.array(codes, "int64"),
             cropland_ha=cropland_ha.astype("float32"), fert_w=fert_w.astype("float32"),
             mask=mask.astype("int32"), lat=lat, lon=lon)
    (cd / "idx2iso.json").write_text(json.dumps({str(k): v for k, v in idx2iso.items()}))
    print(f"  built weight cache: {len(codes)} crops, {ptr[-1]:,} nonzero cells -> {cd}")
    return cd


def load_weight_cache(cfg):
    import json
    cd = _cache_dir(cfg)
    z = np.load(cd / "weights.npz", allow_pickle=False)
    idx2iso = {int(k): v for k, v in json.loads((cd / "idx2iso.json").read_text()).items()}
    return {k: z[k] for k in z.files} | {"idx2iso": idx2iso}


_PROXY_MEMO = {}


def _avail_epochs(files, pat):
    import re
    eps = set()
    for f in files:
        m = re.search(pat, Path(f).name)
        if m:
            eps.add(int(m.group(1)))
    return sorted(eps)


def _memo_proxy(kind, year, avail, fn):
    ep = min(avail, key=lambda e: abs(e - year)) if avail else None
    key = (kind, ep)
    if key not in _PROXY_MEMO:
        _PROXY_MEMO[key] = fn()
    return _PROXY_MEMO[key]


def run_downscale_fast(cfg, year, cache, nutrient="N"):
    """Per-year downscale using the prebuilt weight cache. Numerically reproduces
    run_downscale; proxies (GLW/MODIS/GHS) are memoized by epoch across calls."""
    import glob as _glob
    raw = cfgmod.get_path(cfg, "raw")
    mask = cache["mask"]; lat = cache["lat"]; lon = cache["lon"]; shape = mask.shape
    idx2iso = cache["idx2iso"]; iso2idx = {v: k for k, v in idx2iso.items()}
    flat_mask = mask.ravel(); nidx = int(mask.max()) + 1
    cells = cache["cells"]; ha = cache["ha"]; ptr = cache["ptr"]; codes = cache["codes"]
    cropland_ha = cache["cropland_ha"].astype("float64")   # CropGRIDS physical cropland (~2020);
                                        # replaced by HYDE H(m,t) below when the envelope is on.
                                        # Every term that is not
                                        # crop-specific (deposition, applied manure, night soil, biosolids,
                                        # fallbacks) and the per-hectare denominator use this one extent.
    fert_w = cache["fert_w"].astype("float64")

    # --- optional HYDE historical land base (gated on data/raw/hyde/*.nc present) ---
    # Crop areas in the model year (config hyde_envelope.mode = 'scaling'):
    #     s_c(m) = CG_c(m,2020) / H(m,2020),   A_c(m,t) = s_c(m) * H(m,t)
    # for every crop's CropGRIDS harvested area; the cropland extent (-> cropland_ha) is HYDE H(m,t) itself
    # (see area_in_year / cropgrids_cropland_extent for the edge rules). Legacy mode 'extent' uses HYDE cropland itself. Also loads grazing land and population.
    s_env = s_env_flat = hyde_pas = hyde_rng = hyde_pop = None
    scaling = None                                   # {"H_ref", "H_t", "factor"}: A_c(m,t) = s_c(m)*H(m,t)
    Href_flat = Ht_flat = None
    hyde_cl = None
    if nutrient == "N":
        hyde_cl, scaling = cropgrids_cropland_extent(cfg, cropland_ha, shape, lat, lon, year)
    if hyde_cl is not None:
        cropland_ha = hyde_cl                                                # year-t cropland = denom+weight
        s_env = scaling["factor"]                    # equivalent per-cell multiplier, legacy branches only
        s_env_flat = s_env.ravel()
        if scaling.get("H_ref") is not None:
            Href_flat = scaling["H_ref"].ravel(); Ht_flat = scaling["H_t"].ravel()
        hyde_pas = load_hyde_grid(cfg, shape, lat, lon, year, "pasture")
        hyde_rng = load_hyde_grid(cfg, shape, lat, lon, year, "rangeland")
        hyde_pop = load_hyde_grid(cfg, shape, lat, lon, year, "population")
        _m = _hyde_envelope_cfg(cfg)[0]
        print(f"  HYDE land base ({year}): cropland envelope ({'share' if _m == 'extent' else _m}) + grazing + population")

    def _maps(df):
        m = {}
        for iso, code, val in zip(df["ISO3"], df["fao_item_code"], df["value_kg"]):
            m.setdefault(int(code), {})[iso] = val
        return m
    crop_nat = national_crop_removal(cfg, year, nutrient)
    bnf_nat = national_bnf_by_item(cfg, year) if nutrient == "N" else None
    crop_map = _maps(crop_nat)
    bnf_map = _maps(bnf_nat) if bnf_nat is not None else {}

    crop_flat = np.zeros(flat_mask.size, "float64")
    bnf_flat = np.zeros(flat_mask.size, "float64")
    allcrop_flat = np.zeros(flat_mask.size, "float64")   # Σ_crops haj (x s_env): residual weight

    def _scatter(target, code_map, code, sl, haj, countries, wsum):
        if code not in code_map:
            return
        vbi = np.zeros(nidx, "float64")
        for iso, v in code_map[code].items():
            i = iso2idx.get(iso)
            if i is not None and pd.notna(v):
                vbi[i] = v
        safe = np.where(wsum > 0, wsum, 1.0)
        contrib = vbi[countries] * haj / safe[countries]
        contrib[countries < 1] = 0.0
        np.add.at(target, sl, contrib)

    for j, code in enumerate(codes):
        sl = cells[ptr[j]:ptr[j + 1]]; haj = ha[ptr[j]:ptr[j + 1]].astype("float64")
        if Href_flat is not None:                            # A_c(m,t) = s_c(m) * H(m,t),
            haj = area_in_year(haj, Href_flat[sl], Ht_flat[sl])   # s_c(m) = CG_c(m,2020)/H(m,2020)
        elif s_env_flat is not None:
            haj = haj * s_env_flat[sl]                       # legacy 'extent' mode
        countries = flat_mask[sl].clip(min=0)
        wsum = np.bincount(countries, weights=haj, minlength=nidx)
        _scatter(crop_flat, crop_map, int(code), sl, haj, countries, wsum)
        _scatter(bnf_flat, bnf_map, int(code), sl, haj, countries, wsum)
        allcrop_flat[sl] += haj                              # cells are unique within one crop
    # Residual placement : national removal/BNF of FAO items with no CropGRIDS layer (not
    # in the cache), or of crops with zero (s-scaled) harvarea in a country, would otherwise be dropped
    # (2023: 0.37 Tg of removal). Each country's residual goes to its all-crop harvarea x s_env (the
    # same total weight as removal), then to cropland_ha (year-t extent when the envelope is on), else is
    # reported as unplaced. Grid national totals then equal the national tables for mask countries.
    _cl_name = "year-t cropland" if s_env_flat is not None else "CropGRIDS cropland"
    _ah_name = "all-crop harvarea x s_env" if s_env_flat is not None else "all-crop harvarea"
    crop_flat, _ = place_national_residual(crop_flat, crop_nat[["ISO3", "value_kg"]], allcrop_flat,
                                           cropland_ha, mask, idx2iso, "CROP", names=(_ah_name, _cl_name))
    if bnf_nat is not None:
        bnf_flat, _ = place_national_residual(bnf_flat, bnf_nat[["ISO3", "value_kg"]], allcrop_flat,
                                              cropland_ha, mask, idx2iso, "BNF", names=(_ah_name, _cl_name))
    del allcrop_flat
    crop_grid = crop_flat.reshape(shape); bnf_grid = bnf_flat.reshape(shape)

    # fertilizer (cached weight), deposition (cropland), via the shared allocate()
    from . import fertilizer, human as humanmod, manure as manuremod
    fert_nat = fertilizer.compute(cfg, raw, year, nutrient).rename(columns={"fert_kg": "value_kg"})
    # FERTILIZER weight, selected by config fertilizer.grid_weight (national magnitude is always
    # FAOSTAT-RFN x CF; allocate() renormalizes per country, so this sets WITHIN-country geography):
    #   coello_cropgrids (default): Coello et al. (2025) only sets each country's split
    #     among 13 crop groups, F_C(g,k,t) (scripts/12d); each group is placed by CropGRIDS harvarea x
    #     s_env -- the SAME basis as crop removal above -- so fertilizer and removal share one crop map:
    #         W(m,t) = Σ_g F_C(g,k,t) · H_g(m)s(m,t) / Σ_{m'∈k} H_g(m')s(m',t).
    #     A group with no area in a country -> all-crop harvarea x s_env -> HYDE cropland; countries
    #     with no Coello -> NPKGRIDS x s_env. Missing 12d table -> falls through to the chain below.
    #   coello (alternative, not used): Coello 5-arcmin dasymetric field (12b), on Coello's own
    #     crop map. Not rescaled by s_env (it carries its own annual extent).
    #   then Adalibieke (N) -> Lu&Tian rate x cropland (N, 1900-2013) -> NPKGRIDS intensity (2020).
    #   When the HYDE envelope is on, the CropGRIDS-2020-area weights are rescaled by s_env.
    coe_tbl = load_coello_group_table(cfg, year, nutrient)
    coe = load_coello_fert(cfg, shape, lat, lon, year, nutrient) if coe_tbl is None else None
    ada = (load_adalibieke_fert(cfg, shape, lat, lon, year)
           if (nutrient == "N" and coe is None and coe_tbl is None) else None)
    lt = (load_lu_tian_fert(cfg, shape, lat, lon, year)
          if (nutrient == "N" and coe is None and ada is None and coe_tbl is None) else None)
    if coe_tbl is not None:
        coe_ha = coello_group_ha_from_cache(cfg, cache)              # static, memoized
        W = coello_cropgrids_weight(coe_tbl, coe_ha, scaling if scaling is not None else s_env,
                                    cropland_ha, None, mask, idx2iso)
        fb = fert_w * s_env if s_env is not None else fert_w
        fert_weight = _fill_zero_weight_countries(W, fb, mask)       # no-Coello countries -> NPKGRIDS
    elif coe is not None:
        # per-cell MASS already -> no s_env, no x cropland_ha. NPKGRIDS backfills missed countries.
        fb = fert_w * s_env if s_env is not None else fert_w
        fert_weight = _fill_zero_weight_countries(coe, fb, mask)
    elif ada is not None:
        fert_weight = ada * s_env if s_env is not None else ada
    elif lt is not None:
        fert_weight = lt * cropland_ha                       # cropland_ha is the year-t extent if envelope on
    elif s_env is not None:
        fert_weight = fert_w * s_env                          # NPKGRIDS intensity x year-t extent
    else:
        fert_weight = fert_w
    fert_grid = allocate(fert_nat, fert_weight, mask, idx2iso, "value_kg")

    from . import deposition as depmod
    dep_grid = np.zeros(shape, "float64")
    dep_w = None; dep_tbl = None                            # native dep-rate weight + country table
    if nutrient == "N":
        dep_tbl = depmod.dep_kgha_for_year(cfg, year)      # per-year (clamped) national rate (kg/ha)
        if dep_tbl is not None:
            dep_w = load_deposition_grid(cfg, shape, lat, lon, year)   # native gradient (kg/ha) or None
            cl_by_idx = np.bincount(flat_mask.clip(min=0), weights=cropland_ha.ravel(), minlength=nidx)
            iso2cl = {idx2iso[i]: cl_by_idx[i] for i in idx2iso if i < len(cl_by_idx)}
            dep_c = dep_tbl.copy()
            dep_c["value_kg"] = dep_c.apply(lambda r: r["dep_ox_kgha"] * iso2cl.get(r["ISO3"], 0.0), axis=1)
            # Cropland deposition: national MAGNITUDE from Vishwakarma WL, W(i,t)*A_crop,i (paper Eq. 2),
            # within-country PATTERN from the ISIMIP3a total-N rate (previously uniform W(i,t)).
            # The weight is I(m,t)*A_crop(m), so the national total is unchanged.
            w_crop = cropland_dep_weight(dep_w, cropland_ha, mask,
                                         cfg.get("deposition", {}).get("cropland_pattern", "isimip"))
            dep_grid = allocate(dep_c[["ISO3", "value_kg"]], w_crop, mask, idx2iso, "value_kg")

    # cell land area + epoch-memoized proxies
    R = 6371000.0
    dlat = float(abs(lat[1] - lat[0])); dlon = float(abs(lon[1] - lon[0]))
    latr = np.deg2rad(lat)
    row = (R**2) * np.deg2rad(dlon) * (np.sin(latr + np.deg2rad(dlat)/2) - np.sin(latr - np.deg2rad(dlat)/2))
    cell_area_ha = np.abs(np.repeat(row[:, None], shape[1], axis=1) / 1e4)
    only_land = mask >= 1

    glw = _glob.glob(str(Path(raw) / cfg["manure"].get("glw_dir", "gridded_livestock") / "*.tif"))
    lcf = (_glob.glob(str(Path(raw) / cfg["manure"].get("landcover_dir", "landcover") / "*.hdf")))
    pgf = _glob.glob(str(Path(raw) / "population_grid" / "*.tif"))
    glw_eps = _avail_epochs(glw, r"(19\d\d|20\d\d)")
    # Applied/pasture livestock-density split: data-driven per-species a_s(i,t) from EMN when
    # applied_fraction_from_emn, else the global cattle_confined_fraction species-list split.
    if cfg["manure"].get("applied_fraction_from_emn", False):
        # a_s(i,t) is per-YEAR (and so is AGLW density, when configured), so key the memo by year.
        appl_avail = [year]
        lw_conf, lw_graz = _memo_proxy(
            "glw_appl", year, appl_avail,
            lambda: _applied_pasture_density(cfg, shape, lat, lon, year, mask, idx2iso))
    else:
        lw_conf = lw_graz = None
    if lw_conf is None and lw_graz is None:
        fconf = float(cfg["manure"].get("cattle_confined_fraction", 0.0))   # cattle confinement split
        cattle = cfg["manure"].get("cattle_code", "CTL")
        conf_codes = list(cfg["manure"].get("confined_species", [])) + ([cattle] if fconf > 0 else [])
        graz_codes = list(cfg["manure"].get("grazing_species", []))
        lw_conf = _memo_proxy("glw_conf", year, glw_eps,
                              lambda: load_livestock_grid(cfg, shape, lat, lon, year, codes=conf_codes,
                                                          frac={cattle: fconf}))
        lw_graz = _memo_proxy("glw_graz", year, glw_eps,
                              lambda: load_livestock_grid(cfg, shape, lat, lon, year, codes=graz_codes,
                                                          frac={cattle: 1.0 - fconf}))
    popw = _memo_proxy("ghs", year, _avail_epochs(pgf, r"E(\d{4})"),
                       lambda: load_population_grid(cfg, shape, year))
    lc = _memo_proxy("modis", year, _avail_epochs(lcf, r"A(\d{4})\d{3}"),
                     lambda: load_landcover_fractions(cfg, shape, year)) if cfg["manure"].get("pasture_pool", True) else None
    if hyde_pop is not None:
        popw = hyde_pop                                       # HYDE population (head count/cell)

    hum_grid = np.zeros(shape, "float64")            # hum_sewage total (= water + land), reference
    food_waste_grid = np.zeros(shape, "float64")     # consumer food-waste N (landscape/total input)
    hum_ns_grid = np.zeros(shape, "float64")         # night soil -> ALL cropland (no urban/rural bias)
    hum_bio_grid = np.zeros(shape, "float64")        # biosolids -> peri-urban cropland
    hum_landfill_grid = np.zeros(shape, "float64")   # landfilled sludge -> retained in landscape (near people)
    hum_water_grid = np.zeros(shape, "float64")      # sewered discharge -> surface water (total only)
    hum_land_grid = np.zeros(shape, "float64")       # non-sewered discharge -> land (landscape + total)
    try:
        _pw = popw if popw is not None else cropland_ha
        hum_nat = humanmod.compute(cfg, raw, year, nutrient)
        if popw is not None and str(cfg["human"].get("sewer_grid_mode", "rural")).lower() != "none":
            dla = abs(float(lat[1] - lat[0])); dlo = abs(float(lon[1] - lon[0]))
            urb = _urban_fraction(popw, lat, dla, dlo)
            w_water = popw * urb                     # sewered discharge -> URBAN population (effluent)
        else:
            w_water = _pw
        if "hum_sew_water_kg" in hum_nat:
            hum_water_grid = allocate(hum_nat.rename(columns={"hum_sew_water_kg": "value_kg"})[["ISO3", "value_kg"]],
                                      w_water, mask, idx2iso, "value_kg")
            hum_land_grid = allocate(hum_nat.rename(columns={"hum_sew_land_kg": "value_kg"})[["ISO3", "value_kg"]],
                                     _pw, mask, idx2iso, "value_kg")        # HUM_terr,nc -> population
        else:
            hum_water_grid = allocate(hum_nat.rename(columns={"hum_sewage_kg": "value_kg"})[["ISO3", "value_kg"]],
                                      _pw, mask, idx2iso, "value_kg")
        hum_grid = hum_water_grid + hum_land_grid
        fw_nat = humanmod.food_waste(cfg, raw, year, nutrient).rename(columns={"food_waste_kg": "value_kg"})
        if len(fw_nat):
            food_waste_grid = allocate(fw_nat[["ISO3", "value_kg"]], _pw, mask, idx2iso, "value_kg")
        hr_nat = humanmod.human_recycle(cfg, raw, year, nutrient)
        if len(hr_nat):                              # night soil -> all cropland, biosolids -> peri-urban cropland
            # same cropland extent as every other cropland term (s(m)·H(m,t), s = CropGRIDS/HYDE 2020, when the
            # envelope is on, else CropGRIDS)
            w_ns, w_bio = _recycle_weights(cfg, cropland_ha, popw, lat, lon)
            # countries with no year-t cropland would otherwise lose these terms in allocate(): fall back to
            # CropGRIDS cropland (night soil), then to the night-soil weight (biosolids)
            w_ns = _fill_zero_weight_countries(w_ns, cache["cropland_ha"].astype("float64"), mask, "HUM_NS")
            w_bio = _fill_zero_weight_countries(w_bio, w_ns, mask, "HUM_BIO")
            hum_ns_grid = allocate(hr_nat.rename(columns={"hum_ns_kg": "value_kg"})[["ISO3", "value_kg"]],
                                   w_ns, mask, idx2iso, "value_kg")
            hum_bio_grid = allocate(hr_nat.rename(columns={"hum_bio_kg": "value_kg"})[["ISO3", "value_kg"]],
                                    w_bio, mask, idx2iso, "value_kg")
            if "hum_landfill_kg" in hr_nat:                       # landfilled sludge -> landscape (near people)
                hum_landfill_grid = allocate(hr_nat.rename(columns={"hum_landfill_kg": "value_kg"})[["ISO3", "value_kg"]],
                                             popw, mask, idx2iso, "value_kg")
    except utils.MissingSourceFile:
        pass
    human_recycle_grid = hum_ns_grid + hum_bio_grid   # hum_crop = night soil + biosolids (-> cropland)

    man_crop = np.zeros(shape, "float64")
    man_past = np.zeros(shape, "float64")                  # pasture-deposited manure (PRP) -> BROAD base
    forage = np.zeros(shape, "float64")                    # grassland forage removal -> BROAD base
    forage_weight = None                                   # grazing proxy used for the product term
    fo = None                                              # forage national totals, hoisted for the Zhang override
    bnf_grass_grid = np.zeros(shape, "float64")            # pasture/forage-legume BNF -> NARROW pasture
    pb = None; rb = None                                    # pb=narrow (managed pasture); rb=broad (pasture+rangeland)
    try:
        ap = manuremod.applied(cfg, raw, year).rename(columns={"manure_kg": "value_kg"})
        man_w = (lw_conf * cropland_ha) if lw_conf is not None else cropland_ha
        man_crop = allocate(ap[["ISO3", "value_kg"]], man_w, mask, idx2iso, "value_kg")
    except utils.MissingSourceFile:
        pass
    gpw_pasture, gpw_graze = (load_gpw_grazing(cfg, shape, lat, lon, year)
                              if cfg["manure"].get("pasture_pool", True) else (None, None))
    if lc is not None or hyde_pas is not None or gpw_graze is not None:
        try:
            pa = manuremod.pasture(cfg, raw, year).rename(columns={"manure_kg": "value_kg"})
            if gpw_graze is not None:                         # GPW (30 m): narrow pasture / broad pasture+rangeland
                pb = gpw_pasture; rb = gpw_graze
            elif hyde_pas is not None:                        # HYDE grazing land (ha/cell)
                pb = hyde_pas; rb = hyde_pas + (hyde_rng if hyde_rng is not None else 0.0)
            else:                                            # MODIS land-cover fractions x cell area
                def _base(cats):
                    frac = np.zeros(shape, "float64")
                    for c in cats:
                        if c in lc:
                            frac += lc[c]
                    return frac * cell_area_ha
                pb = _base(cfg["manure"]["grazing_base_pasture"]); rb = _base(cfg["manure"]["grazing_base_rangeland"])
            wr = (lw_graz * rb) if lw_graz is not None else rb    # grazing weight on BROAD base (pasture+rangeland)
            man_past = allocate(pa[["ISO3", "value_kg"]], wr, mask, idx2iso, "value_kg")   # PRP -> broad
            if nutrient == "N":                                # forage + pasture-legume BNF -> broad grazing land
                try:
                    from . import forage as foragemod, country_areas as ca_mod, bnf as bnfmod
                    fo_areas = ca_mod.load_areas(cfg, raw, year)
                    fo = None                # forage is built per cell after the cap (grid-scale Eq. 29)
                    forage_weight = wr       # grazing proxy for the product term
                    bg = bnfmod.grassland(cfg, raw, year, fo_areas).rename(columns={"bnf_grass_kg": "value_kg"})
                    # Pasture-legume BNF -> BROAD base (perm. meadows+pastures whole-pasture-average
                    # rate; place over pasture+rangeland to recover the calibrated uniform rate).
                    bnf_grass_grid = allocate(bg[["ISO3", "value_kg"]], rb, mask, idx2iso, "value_kg")
                except Exception as e:  # noqa: BLE001
                    print(f"  [warn] forage/grassland-BNF skipped ({e})")
        except utils.MissingSourceFile:
            pass

    # land/NANI view extras: deposition on ALL land + the (1-CF) grassland fertilizer.
    # PIECEWISE deposition (see run_downscale): cropland keeps the Vishwakarma WL magnitude
    # (= dep_grid, conserving W_i*A_crop_i); non-cropland gets the NATIVE ISIMIP rate I_c over its
    # (a_c - g_c) area. Only the Vishwakarma cropland total is conserved.
    dep_land_grid = dep_grid.copy()                         # fallback = cropland deposition only
    fert_grass = np.zeros(shape, "float64")
    if nutrient == "N" and dep_tbl is not None:
        if dep_w is not None:                              # I_c available -> piecewise
            noncrop_ha = np.clip(cell_area_ha - cropland_ha, 0.0, None)     # a_c - g_c
            dep_land_grid = dep_grid + dep_w * noncrop_ha                   # Wang(cropland) + ISIMIP(non-crop)
            dep_land_grid = np.where(mask >= 0, dep_land_grid, 0.0)         # land of countries only 
        else:                                              # no ISIMIP grid -> cropland rate over all land
            la_by_idx = np.bincount(flat_mask.clip(min=0), weights=cell_area_ha.ravel(), minlength=nidx)
            iso2la = {idx2iso[i]: la_by_idx[i] for i in idx2iso if i < len(la_by_idx)}
            dep_l = dep_tbl.copy()
            dep_l["value_kg"] = dep_l.apply(lambda r: r["dep_ox_kgha"] * iso2la.get(r["ISO3"], 0.0), axis=1)
            dep_land_grid = allocate(dep_l[["ISO3", "value_kg"]], cell_area_ha, mask, idx2iso, "value_kg")
    try:                                                    # grassland (1-CF) fertilizer -> NARROW managed pasture (GPW cultivated)
        gf = fertilizer.grassland(cfg, raw, year, nutrient).rename(columns={"fert_kg": "value_kg"})
        base_narrow = pb if pb is not None else cropland_ha
        base_broad = rb if rb is not None else cropland_ha
        graz_w = _fill_zero_weight_countries(base_narrow, base_broad, mask, "FERTgrass")   # narrow -> broad if empty
        fert_grass = allocate(gf[["ISO3", "value_kg"]], graz_w, mask, idx2iso, "value_kg")
    except Exception:  # noqa: BLE001
        pass

    # synthetic-fertilizer NH3 volatilization loss (soil-surface basis): fert x FracGASF(country)
    fert_vol = np.zeros(shape, "float64"); fert_vol_land = np.zeros(shape, "float64")
    if nutrient == "N":
        try:
            from . import fert_volatilization as fvmod
            fg = fvmod.fracgasf_by_country(cfg, raw, year)
            iso2fg = dict(zip(fg["ISO3"], fg["fracgasf"]))
            fg_by_idx = np.array([iso2fg.get(idx2iso.get(i, ""), fvmod.DEFAULT_FRACGASF)
                                  for i in range(nidx)], dtype="float64")
            fg_cell = fg_by_idx[mask.clip(min=0)]
            fert_vol = fert_grid * fg_cell                       # cropland-view loss
            fert_vol_land = (fert_grid + fert_grass) * fg_cell   # land-view loss (full fertilizer)
        except Exception as e:  # noqa: BLE001
            print(f"  [warn] fert volatilization skipped ({e})")

    # --- Optional Tian-group gridded manure PATTERN (see run_downscale): weight-only, FAOSTAT EMN
    # magnitudes enforced by allocate(). Enabled by manure.grid_source=='zhang'.
    if nutrient == "N" and str(cfg.get("manure", {}).get("grid_source", "emn")).lower() == "zhang":
        _rawm = cfgmod.get_path(cfg, "raw")
        zc, zc_placed = load_zhang_manure(cfg, shape, lat, lon, year, "crop")   # crop = pre-placed 0.05deg
        zg, zg_placed = load_zhang_manure(cfg, shape, lat, lon, year, "grass")   # grass: placed (GPW) or 5' density
        if zc is None or zg is None:                                   # GUARD: don't silently fall back to AGLW
            raise RuntimeError(
                f"manure.grid_source='zhang' but the Zhang derived grids for {year} are missing "
                f"(expected data/raw/zhang_manure/derived/zhang_man_{{crop,grass}}_*_0p05.nc). "
                f"Download Zhang et al. 2017 (PANGAEA 10.1594/PANGAEA.871980: ManNitPro + ManNitProCrpRd) "
                f"and run scripts/12c_build_zhang_manure.py first. Refusing to fall back to the AGLW weight.")
        wcrop = zc if zc_placed else zc * cropland_ha                  # placed field used directly (no x cropland_ha)
        wcrop = _fill_zero_weight_countries(wcrop, cropland_ha, mask, "MAN")
        apn = manuremod.applied(cfg, _rawm, year).rename(columns={"manure_kg": "value_kg"})
        man_crop = allocate(apn[["ISO3", "value_kg"]], wcrop, mask, idx2iso, "value_kg")
        gbase = rb if rb is not None else cropland_ha                 # broad grazing base (only for zero-weight fill)
        wgrass_raw = zg if zg_placed else _zhang_grass_weight(zg, lat, lon)     # GPW-placed used directly
        wgrass = _fill_zero_weight_countries(wgrass_raw, gbase, mask, "MAN")
        pan = manuremod.pasture(cfg, _rawm, year).rename(columns={"manure_kg": "value_kg"})
        man_past = allocate(pan[["ISO3", "value_kg"]], wgrass, mask, idx2iso, "value_kg")
        forage_weight = wgrass                    # product term shares the pasture-manure (Zhang) proxy

    # cap combined grassland fertilizer + pasture-deposited manure at the pasture-N ceiling (guards the
    # implausible loadings where GPW under-detects managed pasture); excess cascades broad -> MODIS -> broad.
    _cap = cfg.get("manure", {}).get("pasture_n_cap_kgha")
    if nutrient == "N" and _cap and rb is not None:
        _modis_g = (lc["grassland"] * cell_area_ha) if (lc is not None and "grassland" in lc) else None
        fert_grass, man_past = _cap_grassland_fert(fert_grass, man_past, rb, _modis_g, mask, idx2iso, float(_cap))

    # grid-scale Eq. 29: FORAGE per cell = min(PRP + grazing-product N, two-band pasture supply)
    if nutrient == "N" and rb is not None and pb is not None:
        try:
            forage = _forage_gridscale(cfg, raw, year, man_past,
                                       forage_weight, pb, rb, mask, idx2iso)
        except Exception as e:  # noqa: BLE001
            print(f"  [warn] grid-scale forage skipped ({e})")

    # cap applied manure on cropland : the excess above the ceiling moves to cropland with
    # headroom nearby, then within the country, so national totals are conserved and manure stays local.
    _mcap = cfg.get("manure", {}).get("cropland_n_cap_kgha")
    if nutrient == "N" and _mcap:
        man_crop, _ = _cap_cropland_manure(man_crop, cropland_ha, mask, idx2iso, float(_mcap),
                                           int(cfg.get("manure", {}).get("cropland_cap_radius_cells", 5)))

    # manure NH3 volatilization loss: applied manure x FracGASM (+ pasture pool for land view)
    man_vol = np.zeros(shape, "float64")
    man_vol_past = np.zeros(shape, "float64")
    if nutrient == "N":
        try:
            from . import man_volatilization as mvmod
            fga = mvmod.fracgasm(cfg, "applied"); fgp = mvmod.fracgasm(cfg, "pasture")
            man_vol = man_crop * fga
            man_vol_past = man_past * fgp                  # grazing-deposited (PRP) loss, broad grazing base
        except Exception as e:  # noqa: BLE001
            print(f"  [warn] manure volatilization skipped ({e})")

    ag_faocomp = fert_grid + bnf_grid + dep_grid + man_crop - crop_grid - fert_vol - man_vol  # FAO-comparable
    ag = ag_faocomp + human_recycle_grid                                      # cropland incl. hum_crop
    # landscape INCLUDES non-sewered human discharge on land (hum_land_grid); total adds only the
    # SEWERED discharge to surface water (hum_water_grid).
    land_base = ((fert_grid + fert_grass) + bnf_grid + bnf_grass_grid + dep_land_grid + man_crop
                 + food_waste_grid + human_recycle_grid + hum_land_grid + hum_landfill_grid
                 - crop_grid - fert_vol_land - man_vol)
    # single grazing base: fertilizer + pasture-legume BNF on NARROW managed pasture (in land_base via
    # fert_grass/bnf_grass_grid); PRP manure + forage removal on BROAD pasture+rangeland. One landscape surplus.
    land_mass = land_base + man_past - man_vol_past - forage
    with np.errstate(divide="ignore", invalid="ignore"):
        _ok = only_land & (cell_area_ha > 0)
        sc = np.where(cropland_ha > 0, ag / cropland_ha, np.nan)
        sc_faocomp = np.where(cropland_ha > 0, ag_faocomp / cropland_ha, np.nan)
        sl_ = np.where(_ok, land_mass / cell_area_ha, np.nan)                    # landscape (incl. non-sewered)
        sf = np.where(_ok, (land_mass + hum_water_grid) / cell_area_ha, np.nan)  # full (+ sewered)
    return dict(crop=crop_grid, bnf=bnf_grid, fert=fert_grid, dep=dep_grid, man_crop=man_crop,
                man_past=man_past, hum_sewage=hum_grid,
                hum_sew_water=hum_water_grid, hum_sew_land=hum_land_grid,
                dep_land=dep_land_grid, fert_grass=fert_grass, fert_vol=fert_vol, man_vol=man_vol,
                man_vol_past=man_vol_past, forage=forage, bnf_grass=bnf_grass_grid, food_waste=food_waste_grid,
                hum_crop=human_recycle_grid, hum_ns=hum_ns_grid, hum_bio=hum_bio_grid,
                hum_landfill=hum_landfill_grid,
                cropland_ha=cropland_ha, cell_area_ha=cell_area_ha,
                surplus_cropland_kgha=sc, surplus_cropland_faocomp_kgha=sc_faocomp,
                surplus_landscape_kgha=sl_, surplus_full_kgha=sf,
                mask=mask, idx2iso=idx2iso, lat=lat, lon=lon)
