#!/usr/bin/env python3
"""Gridded N-surplus uncertainty from the coefficient-level national Monte Carlo (revised Eq. 67).

Each cell holds a fixed share of its country's national total for every budget term, so the cell
surplus variance follows in closed form from the national ensemble:

    sigma^2_surplus(m) = sum_k sum_l s_k s_l x_k(m) x_l(m) C_kl(i)

x_k(m) is term k in cell m of country i, s_k is +1 for inputs and -1 for outputs, and C_kl(i) is the
relative covariance of terms k and l across the national realizations for country i (cov_kl divided
by the product of the term means; C_kk is the squared CV). The off-diagonal terms carry the
dependence between terms that share a coefficient or a partition, so the grid layer is consistent
with the national Monte Carlo.

Needs, per year: data/outputs/grids/surplus_N_<year>_0p05.nc (scripts/26) and
data/outputs/uncertainty_mc/cov_N_<year>.npz (scripts/46).

Outputs: <out>/uncertainty_N_<year>_0p05.nc with, per view (cropland / landscape / full):
  surplus_<view>_mean_kgha, _sd_kgha, _coef_sd_kgha, _act_sd_kgha, _ci95_kgha  (kg N/ha)
Default <out> = data/outputs/uncertainty_mc/grids (the scripts/37 outputs are not overwritten).

Usage:
  python scripts/47_gridded_uncertainty_cov.py --year 2000
  python scripts/47_gridded_uncertainty_cov.py --start 1961 --end 2023
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import config as cfgmod                              # noqa: E402
from src import uncertainty_mc as MC                           # noqa: E402
from src.downscale import load_country_grid                    # noqa: E402

# grid layer saved by scripts/26 for each term; None = reconstructed from a saved layer x the
# country's national ratio (central terms from the national ensemble)
LAYER = {"fert": "fert_kg", "fert_grass": "fert_grass_kg", "fert_vol": "fert_vol_kg",
         "fert_vol_full": None, "crop": "crop_kg", "bnf": "bnf_kg", "bnf_grass": "bnf_grass_kg",
         "dep": "dep_kg", "dep_land": "dep_land_kg", "man_crop": "man_crop_kg", "man_past": "man_past_kg",
         "man_vol": None, "man_vol_past": "man_vol_past_kg", "forage": "forage_kg",
         "food_waste": "food_waste_kg", "hum_ns": "hum_ns_kg", "hum_bio": "hum_bio_kg",
         "hum_landfill": None, "hum_sew_land": "hum_sew_land_kg", "hum_sew_water": "hum_sew_water_kg"}
# reconstruction: term -> (numerator term, [denominator terms], base layer terms)
RECON = {"fert_vol_full": ("fert_vol_full", ["fert", "fert_grass"]),
         "man_vol": ("man_vol", ["man_crop"]),
         "hum_landfill": ("hum_landfill", ["hum_bio"]),
         "man_vol_past": ("man_vol_past", ["man_past"])}
OUT_VIEW = {"cropland": "cropland", "landscape": "landscape", "total": "full"}
CHUNK = 2_000_000


def run(cfg, year, grids, covdir, out_dir):
    t0 = time.time()
    ncf = Path(grids) / f"surplus_N_{year}_0p05.nc"
    cvf = Path(covdir) / f"cov_N_{year}.npz"
    if not ncf.exists() or not cvf.exists():
        print(f"  {year}: missing {'grid' if not ncf.exists() else 'covariance'} file, skipped")
        return
    ds = xr.open_dataset(ncf).sortby("lat").sortby("lon")
    mask2d, idx2iso, clat, clon = load_country_grid(cfg)
    cg = xr.DataArray(mask2d, coords={"lat": clat, "lon": clon}, dims=("lat", "lon")).sortby("lat").sortby("lon")
    cg = cg.reindex(lat=ds.lat, lon=ds.lon, method="nearest")
    lat, lon = ds.lat.values, ds.lon.values
    ny, nx = len(lat), len(lon)
    z = np.load(cvf, allow_pickle=True)
    terms = [str(t) for t in z["terms"]]
    K = len(terms); tix = {t: k for k, t in enumerate(terms)}
    iso_nat = [str(c) for c in z["iso"]]
    pos = {c: k for k, c in enumerate(iso_nat)}
    central = z["central"]                                                  # (nC, K)

    # country of each cell -> row in the national arrays (-1 = not in the ensemble)
    cmask = np.nan_to_num(cg.values).astype(int)
    lut = np.full(int(cmask.max()) + 2, -1, int)
    for i, c in idx2iso.items():
        if 0 < int(i) < len(lut) and c in pos:
            lut[int(i)] = pos[c]
    crow = lut[np.clip(cmask, 0, len(lut) - 1)].ravel()

    def G(name):
        return np.nan_to_num(ds[name].values).astype("float64").ravel() if name in ds else np.zeros(ny * nx)

    X = {}
    for t in terms:
        if LAYER.get(t) and LAYER[t] in ds:
            X[t] = G(LAYER[t])
    for t, (num, dens) in RECON.items():
        if t in X:
            continue
        den_nat = sum(central[:, tix[d]] for d in dens)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(den_nat > 0, central[:, tix[num]] / np.where(den_nat > 0, den_nat, 1.0), 0.0)
        rr = np.where(crow >= 0, r[np.clip(crow, 0, None)], 0.0)
        X[t] = sum(X.get(d, G(LAYER[d])) for d in dens) * rr
    missing = [t for t in terms if t not in X]
    for t in missing:
        X[t] = np.zeros(ny * nx)
    if missing:
        print(f"  {year}: no grid layer for {missing}; treated as zero")

    valid = np.zeros(ny * nx, bool)
    for t in terms:
        valid |= X[t] != 0
    valid &= crow >= 0
    vidx = np.flatnonzero(valid)
    R = 6371000.0; dla = abs(float(lat[1] - lat[0])); dlo = abs(float(lon[1] - lon[0]))
    rowm = (R ** 2) * np.deg2rad(dlo) * (np.sin(np.deg2rad(lat) + np.deg2rad(dla) / 2)
                                         - np.sin(np.deg2rad(lat) - np.deg2rad(dla) / 2))
    cell_ha = np.abs(np.repeat((rowm / 1e4)[:, None], nx, axis=1)).ravel()
    crop_ha = G("cropland_ha")
    denom = {"cropland": crop_ha, "landscape": cell_ha, "total": cell_ha}
    if "cropland_frac" in ds:                                   # same reporting threshold as scripts/26
        mf = float(ds.attrs.get("min_cropland_frac", 0.01))
        denom["cropland"] = np.where(G("cropland_frac") >= mf, crop_ha, 0.0)

    C = {m: z[f"C_{m}"].astype("float64") for m in ("total", "coef", "act")}
    signs = {v: np.array([MC.VIEWS[v].get(t, 0) for t in terms], float) for v in MC.VIEWS}
    var = {(v, m): np.zeros(len(vidx)) for v in MC.VIEWS for m in C}
    mean = {v: np.zeros(len(vidx)) for v in MC.VIEWS}
    pairs = [(k, l) for k in range(K) for l in range(k, K)]
    for a in range(0, len(vidx), CHUNK):
        sl = slice(a, min(len(vidx), a + CHUNK))
        ci = crow[vidx[sl]]
        Xc = np.stack([X[t][vidx[sl]] for t in terms], axis=1)               # (n, K)
        for v in MC.VIEWS:
            mean[v][sl] = Xc @ signs[v]
        for k, l in pairs:
            xx = Xc[:, k] * Xc[:, l] * (1.0 if k == l else 2.0)
            if not xx.any():
                continue
            for m, Cm in C.items():
                p = xx * Cm[ci, k, l]
                for v in MC.VIEWS:
                    s = signs[v][k] * signs[v][l]
                    if s:
                        var[(v, m)][sl] += s * p

    out = {}

    def scatter(vals):
        arr = np.full(ny * nx, np.nan, "float32"); arr[vidx] = vals.astype("float32")
        return arr.reshape(ny, nx)

    for v in MC.VIEWS:
        d = denom[v][vidx]
        ok = d > 0
        per = lambda x: np.where(ok, x / np.where(ok, d, 1.0), np.nan)          # noqa: E731
        sd = np.sqrt(np.clip(var[(v, "total")], 0, None))
        name = OUT_VIEW[v]
        out[f"surplus_{name}_mean_kgha"] = (("lat", "lon"), scatter(per(mean[v])))
        out[f"surplus_{name}_sd_kgha"] = (("lat", "lon"), scatter(per(sd)))
        out[f"surplus_{name}_coef_sd_kgha"] = (("lat", "lon"), scatter(per(np.sqrt(np.clip(var[(v, "coef")], 0, None)))))
        out[f"surplus_{name}_act_sd_kgha"] = (("lat", "lon"), scatter(per(np.sqrt(np.clip(var[(v, "act")], 0, None)))))
        out[f"surplus_{name}_ci95_kgha"] = (("lat", "lon"), scatter(per(1.96 * sd)))
    o = xr.Dataset(out, coords={"lat": lat, "lon": lon})
    o.attrs["method"] = ("closed-form propagation of the per-country relative covariance of the budget terms "
                         "from the coefficient-level national Monte Carlo (scripts/46); revised Eq. 67")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    dest = Path(out_dir) / f"uncertainty_N_{year}_0p05.nc"
    o.to_netcdf(dest, encoding={k: {"zlib": True, "complevel": 1} for k in o.data_vars})
    # conservation check: grid terms summed by country vs the national central terms
    rel = []
    cv_ = crow[vidx]
    for t in ("fert", "crop", "man_crop", "forage", "hum_sew_water"):
        if t in X:
            gs = np.bincount(cv_, weights=X[t][vidx], minlength=len(iso_nat))
            nat = central[:, tix[t]]
            tot = nat.sum()
            if tot > 0:
                rel.append(f"{t} {100 * (gs.sum() - tot) / tot:+.2f}%")
    print(f"  {year}: grid vs national totals: " + ", ".join(rel))
    print(f"  {year}: {len(vidx):,} cells in {time.time()-t0:.0f}s -> {dest.name}; "
          + ", ".join(f"{OUT_VIEW[v]} median sd {np.nanmedian(out[f'surplus_{OUT_VIEW[v]}_sd_kgha'][1]):.1f} kg/ha"
                      for v in MC.VIEWS))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--year", type=int); ap.add_argument("--start", type=int); ap.add_argument("--end", type=int)
    ap.add_argument("--grids", default=None, help="folder with surplus_N_<year>_0p05.nc")
    ap.add_argument("--cov", default=None, help="folder with cov_N_<year>.npz (scripts/46 output)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    cfg = cfgmod.load_config()
    outputs = cfgmod.get_path(cfg, "outputs")
    grids = Path(a.grids) if a.grids else outputs / "grids"
    covdir = Path(a.cov) if a.cov else outputs / "uncertainty_mc"
    out = Path(a.out) if a.out else outputs / "uncertainty_mc" / "grids"
    years = [a.year] if a.year else list(range(a.start, a.end + 1))
    for y in years:
        run(cfg, y, grids, covdir, out)
    print("Done.")


if __name__ == "__main__":
    main()
