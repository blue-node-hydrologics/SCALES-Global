#!/usr/bin/env python3
"""Build per-YEAR country TOTAL N deposition (NOy + NHx) -> coefficients/dep_total_country_byyear.csv.

Option B basis: the soil-surface N budget pairs net (post-management) manure with TOTAL atmospheric
deposition (oxidized NOy + reduced NHx). This aggregates the ISIMIP3a NOy and NHx histsoc fields
to a cos-lat (area) weighted country mean for every year 1961-2021 (2022-23 reuse 2021 in the engine).

Uses the on-disk CropGRIDS country mask (no Natural Earth / regionmask download).
Long table: ISO3, year, dep_kgha, dep_noy_kgha, dep_nhx_kgha.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, xarray as xr

ROOT = Path(__file__).resolve().parents[1]
G_M2_TO_KG_HA = 10.0
START, END = 1961, 2021

def open_field(p):
    ds = xr.open_dataset(p, decode_times=False)
    v = [x for x in ds.data_vars if x != "time_bnds"][0]
    return ds, v

def main():
    dep = ROOT / "data/raw/deposition"
    noy_ds, noy_v = open_field(dep / "ndep-noy_histsoc_monthly_1901_2021.nc")
    nhx_ds, nhx_v = open_field(dep / "ndep-nhx_histsoc_monthly_1901_2021.nc")
    dlat = noy_ds[noy_v]["lat"].values; dlon = noy_ds[noy_v]["lon"].values
    t = noy_ds["time"].values

    cm = xr.open_dataset(ROOT / "data/raw/cropgrids/nc_maps/CROPGRIDSv1.08_NC_maps/Countries_2018.nc")
    mask = cm["country"].values.astype("int32"); mlat = cm["lat"].values; mlon = cm["lon"].values
    cw = pd.read_csv(ROOT / "coefficients/cropgrids_country_crosswalk.csv")
    idx2iso = dict(zip(cw["mask_index"].astype(int), cw["ISO3"]))
    # country index aligned to the 0.5deg dep grid (nearest fine cell)
    mlat_s = np.argsort(mlat); mlon_s = np.argsort(mlon)
    fi = mlat_s[np.searchsorted(mlat[mlat_s], dlat).clip(0, len(mlat)-1)]
    fj = mlon_s[np.searchsorted(mlon[mlon_s], dlon).clip(0, len(mlon)-1)]
    cidx = mask[np.ix_(fi, fj)].ravel()
    # Weight the national mean by CROPLAND AREA (Vishwakarma et al. 2023 method) rather than
    # grid-cell area, so the national deposition reflects the rate experienced by cropland. Block-
    # aggregate the model cropland_ha (0.05deg) to the 0.5deg deposition grid; fall back to cos-lat
    # area if no cropland grid is available.
    A = (np.cos(np.deg2rad(dlat))[:, None] * np.ones((1, len(dlon)))).ravel()
    W, wtype = A, "cos-lat area"
    cgf = ROOT / "data/outputs/grids/surplus_N_2020_0p05.nc"
    if cgf.exists():
        cl = xr.open_dataset(cgf)["cropland_ha"].values            # 3600x7200, lat ascending
        H, Wd = cl.shape; fL, fO = H // len(dlat), Wd // len(dlon)
        cl = np.nan_to_num(cl[:len(dlat)*fL, :len(dlon)*fO]).reshape(len(dlat), fL, len(dlon), fO).sum((1, 3))
        cl = cl[::-1, :]                                            # ascending -> descending to match dep lat
        W, wtype = cl.ravel(), "cropland area (Vishwakarma 2023 method)"
    nidx = int(cidx.max()) + 1
    wsum = np.bincount(cidx.clip(min=0), weights=W, minlength=nidx)

    rows = []
    for year in range(START, END + 1):
        sel = np.where((t >= (year-1850)*12) & (t < (year-1849)*12))[0]
        if len(sel) == 0:
            continue
        noy = (noy_ds[noy_v].isel(time=sel).sum("time").values * G_M2_TO_KG_HA).ravel()
        nhx = (nhx_ds[nhx_v].isel(time=sel).sum("time").values * G_M2_TO_KG_HA).ravel()
        sno = np.bincount(cidx.clip(min=0), weights=np.nan_to_num(noy)*W, minlength=nidx)
        snh = np.bincount(cidx.clip(min=0), weights=np.nan_to_num(nhx)*W, minlength=nidx)
        for i in range(1, nidx):
            if i in idx2iso and wsum[i] > 0:
                no = sno[i]/wsum[i]; nh = snh[i]/wsum[i]
                rows.append({"ISO3": idx2iso[i], "year": year,
                             "dep_kgha": no+nh, "dep_noy_kgha": no, "dep_nhx_kgha": nh})
    out = (pd.DataFrame(rows).groupby(["ISO3", "year"], as_index=False)
           [["dep_kgha", "dep_noy_kgha", "dep_nhx_kgha"]].mean())
    out["source"] = f"ISIMIP3a NOy+NHx histsoc per-year (DOI 10.48364/ISIMIP.759077); {wtype}-weighted country mean"
    print(f"  weighting: {wtype}")
    dest = ROOT / "coefficients/dep_total_country_byyear.csv"
    out.to_csv(dest, index=False)
    g = out.groupby("year")[["dep_noy_kgha", "dep_nhx_kgha", "dep_kgha"]].mean()
    print(f"wrote {dest}: {len(out)} rows, {out.ISO3.nunique()} countries, {START}-{END}")
    print(f"  global-mean total dep: {g['dep_kgha'].loc[START]:.2f} ({START}) -> {g['dep_kgha'].loc[END]:.2f} ({END}) kg N/ha")
    print(f"  {END}: NOy {g['dep_noy_kgha'].loc[END]:.2f} + NHx {g['dep_nhx_kgha'].loc[END]:.2f} = {g['dep_kgha'].loc[END]:.2f} kg N/ha")

if __name__ == "__main__":
    main()
