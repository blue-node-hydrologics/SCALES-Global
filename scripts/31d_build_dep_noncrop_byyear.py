#!/usr/bin/env python3
"""Build per-YEAR country NON-CROPLAND ISIMIP N deposition (NOy+NHx) mean
   -> coefficients/dep_noncrop_country_byyear.csv.

Companion to scripts/31b (which builds the CROPLAND-area-weighted ISIMIP mean). Under the
PIECEWISE landscape/total deposition scheme, cropland receives the Vishwakarma WL rate W_i and
NON-cropland receives the native ISIMIP rate. The national landscape/total deposition is

    DEP_land_i = W_i * A_crop_i  +  Ibar_noncrop_i * (A_land_i - A_crop_i)

where Ibar_noncrop_i is the ISIMIP deposition rate averaged over the country's NON-cropland area
(kg N ha-1 yr-1, total N). This script produces Ibar_noncrop_i for pipeline.py; the gridded engine
(downscale.py) computes the same non-cropland term cell-by-cell as I_c*(a_c - g_c).

Identical machinery to scripts/31b, but the aggregation weight is NON-cropland area
(cell_area_ha - cropland_ha at 0.5 deg) instead of cropland area.

Uses the on-disk CropGRIDS country mask + the 0.05deg cropland grid.
Long table: ISO3, year, dep_noncrop_kgha, dep_noncrop_noy_kgha, dep_noncrop_nhx_kgha.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, xarray as xr

ROOT = Path(__file__).resolve().parents[1]
G_M2_TO_KG_HA = 10.0
START, END = 1961, 2021
R_EARTH_M = 6371000.0


def open_field(p):
    ds = xr.open_dataset(p, decode_times=False)
    v = [x for x in ds.data_vars if x != "time_bnds"][0]
    return ds, v


def cell_area_ha(lat, dlat, dlon):
    latr = np.deg2rad(lat)
    row = (R_EARTH_M ** 2) * np.deg2rad(dlon) * (
        np.sin(latr + np.deg2rad(dlat) / 2) - np.sin(latr - np.deg2rad(dlat) / 2))
    return np.abs(row) / 1e4                                     # m2 -> ha, per lat row


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
    mlat_s = np.argsort(mlat); mlon_s = np.argsort(mlon)
    fi = mlat_s[np.searchsorted(mlat[mlat_s], dlat).clip(0, len(mlat) - 1)]
    fj = mlon_s[np.searchsorted(mlon[mlon_s], dlon).clip(0, len(mlon) - 1)]
    cidx = mask[np.ix_(fi, fj)].ravel()

    # NON-cropland area weight at 0.5deg = cell_area_ha - cropland_ha (clipped >= 0).
    d_lat = abs(float(dlat[1] - dlat[0])); d_lon = abs(float(dlon[1] - dlon[0]))
    cell05 = np.repeat(cell_area_ha(dlat, d_lat, d_lon)[:, None], len(dlon), axis=1)   # ha, 0.5deg
    cgf = ROOT / "data/outputs/grids/surplus_N_2020_0p05.nc"
    if not cgf.exists():
        sys.exit(f"need {cgf} (0.05deg cropland_ha) to form non-cropland weight; run scripts/26 first")
    cl = xr.open_dataset(cgf)["cropland_ha"].values                                    # 3600x7200, lat asc
    H, Wd = cl.shape; fL, fO = H // len(dlat), Wd // len(dlon)
    cl = np.nan_to_num(cl[:len(dlat) * fL, :len(dlon) * fO]).reshape(len(dlat), fL, len(dlon), fO).sum((1, 3))
    cl = cl[::-1, :]                                            # ascending -> descending to match dep lat
    noncrop = np.clip(cell05 - cl, 0.0, None)                   # ha of non-cropland per 0.5deg cell
    W = noncrop.ravel()
    nidx = int(cidx.max()) + 1
    wsum = np.bincount(cidx.clip(min=0), weights=W, minlength=nidx)

    rows = []
    for year in range(START, END + 1):
        sel = np.where((t >= (year - 1850) * 12) & (t < (year - 1849) * 12))[0]
        if len(sel) == 0:
            continue
        noy = (noy_ds[noy_v].isel(time=sel).sum("time").values * G_M2_TO_KG_HA).ravel()
        nhx = (nhx_ds[nhx_v].isel(time=sel).sum("time").values * G_M2_TO_KG_HA).ravel()
        sno = np.bincount(cidx.clip(min=0), weights=np.nan_to_num(noy) * W, minlength=nidx)
        snh = np.bincount(cidx.clip(min=0), weights=np.nan_to_num(nhx) * W, minlength=nidx)
        for i in range(1, nidx):
            if i in idx2iso and wsum[i] > 0:
                no = sno[i] / wsum[i]; nh = snh[i] / wsum[i]
                rows.append({"ISO3": idx2iso[i], "year": year, "dep_noncrop_kgha": no + nh,
                             "dep_noncrop_noy_kgha": no, "dep_noncrop_nhx_kgha": nh})
    out = (pd.DataFrame(rows).groupby(["ISO3", "year"], as_index=False)
           [["dep_noncrop_kgha", "dep_noncrop_noy_kgha", "dep_noncrop_nhx_kgha"]].mean())
    out["source"] = ("ISIMIP3a NOy+NHx histsoc per-year (DOI 10.48364/ISIMIP.759077); "
                     "NON-cropland-area-weighted country mean (piecewise landscape deposition)")
    dest = ROOT / "coefficients/dep_noncrop_country_byyear.csv"
    out.to_csv(dest, index=False)
    g = out.groupby("year")["dep_noncrop_kgha"].mean()
    print(f"wrote {dest}: {len(out)} rows, {out.ISO3.nunique()} countries, {START}-{END}")
    print(f"  global-mean NON-cropland dep: {g.loc[START]:.2f} ({START}) -> {g.loc[END]:.2f} ({END}) kg N/ha")


if __name__ == "__main__":
    main()
