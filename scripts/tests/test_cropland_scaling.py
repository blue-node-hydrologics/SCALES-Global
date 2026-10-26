#!/usr/bin/env python3
"""
# Synthetic unit test for crop areas in year t (CropGRIDS / HYDE scaling factor)

Checks, without real data, the functions in `src/downscale.py` that implement

    s_c(m)   = CG_c(m, 2020) / H(m, 2020)
    A_c(m,t) = s_c(m) * H(m, t)

- `cropgrids_scaling_factor`: the static factor, undefined (NaN) where H(m,2020) = 0;
- `area_in_year`: s_c * H(t); A_c = 0 where s_c is undefined (HYDE accepted; old hold rule optional);
- `hyde_scaling_inputs`: growth cap on crop areas at max(CropGRIDS, cell area); reference year read once;
- `cropgrids_cropland_extent`: the cropland extent is HYDE H(m,t) itself; crop areas reproduce CropGRIDS
  in 2020; worked example from the methods;
- `coello_cropgrids_weight` with the scaling dict gives the same weight as the equivalent multiplier.

HYDE loading (`load_hyde_grid`) is replaced by a fake so the test needs no data or rasterio.

Usage (from the model root): `python scripts/tests/test_cropland_scaling.py`
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src import downscale as ds  # noqa: E402


def main():
    # 1. worked example: CropGRIDS maize 30 ha, HYDE 2020 60 ha, HYDE 1980 90 ha -> s = 0.5, A(1980) = 45 ha
    s = ds.cropgrids_scaling_factor(np.array([30.]), np.array([60.]))
    assert np.allclose(s, 0.5)
    assert np.allclose(ds.area_in_year(np.array([30.]), np.array([60.]), np.array([90.])), 45.)
    # undefined factor where H(2020) = 0 -> HYDE accepted (A = 0); legacy hold rule still available
    s0 = ds.cropgrids_scaling_factor(np.array([30., 0.]), np.array([0., 0.]))
    assert np.isnan(s0).all()
    assert np.allclose(ds.area_in_year(np.array([30., 0.]), np.array([0., 0.]), np.array([90., 5.])), [0., 0.])
    assert np.allclose(ds.area_in_year(np.array([30., 0.]), np.array([0., 0.]), np.array([90., 5.]),
                                       hold_unscaled=True), [30., 0.])

    # 2. fake HYDE on a 4x4 model grid
    lat = np.array([0.025, 0.075, 0.125, 0.175]); lon = lat.copy(); shape = (4, 4)
    H = {2020: np.full(shape, 60.0), 1980: np.full(shape, 90.0)}
    H[2020][0, 0] = 0.0                                   # no HYDE cropland in 2020 in this cell
    calls = []

    def fake(cfg, shape_, lat_, lon_, year, layer):
        calls.append(year)
        return H[year].copy()
    ds.load_hyde_grid = fake
    ds._HYDE_REF_MEMO.clear()
    cfg = {"paths": {"raw": "/tmp/none"}, "_root": "/", "hyde_envelope": {"mode": "scaling", "ref_year": 2020}}
    cg = np.full(shape, 30.0)
    cl20, sc20 = ds.cropgrids_cropland_extent(cfg, cg, shape, lat, lon, 2020)
    assert np.allclose(cl20, H[2020])                              # cropland extent = HYDE 2020
    a20 = ds.area_in_year(cg, sc20["H_ref"], sc20["H_t"])
    exp20 = cg.copy(); exp20[0, 0] = 0.0                           # crop areas = CropGRIDS where H(2020) > 0
    assert np.allclose(a20, exp20)
    cl80, sc80 = ds.cropgrids_cropland_extent(cfg, cg, shape, lat, lon, 1980)
    assert np.allclose(cl80, H[1980])                              # cropland extent = HYDE 1980
    exp = np.full(shape, 45.0); exp[0, 0] = 0.0                    # no H(2020): HYDE accepted, no crops
    assert np.allclose(ds.area_in_year(cg, sc80["H_ref"], sc80["H_t"]), exp)
    assert calls.count(2020) == 1                                  # reference year read once
    assert np.allclose(sc80["factor"], exp / 30.0)

    # 3. growth cap: CropGRIDS near the cell area cannot grow beyond the cell; never cut below CropGRIDS
    cell = (ds._cell_area_ha(lat, 0.05, 0.05) / 1e4)[:, None] * np.ones((1, 4))
    big = cell * 0.9
    Hc = {2020: big.copy(), 1980: big * 3.0}
    ds.load_hyde_grid = lambda cfg_, s_, la_, lo_, year, layer: Hc[year].copy()
    ds._HYDE_REF_MEMO.clear()
    clb, scb = ds.cropgrids_cropland_extent(cfg, big, shape, lat, lon, 1980)
    ab = ds.area_in_year(big, scb["H_ref"], scb["H_t"])            # crop areas capped at the cell area
    assert np.allclose(ab, cell), (ab / cell)
    assert np.allclose(clb, big * 3.0)                             # the extent is HYDE as given
    over = cell * 1.2
    Ho = {2020: over.copy(), 1980: over * 2.0}
    ds.load_hyde_grid = lambda cfg_, s_, la_, lo_, year, layer: Ho[year].copy()
    ds._HYDE_REF_MEMO.clear()
    _, so20 = ds.cropgrids_cropland_extent(cfg, over, shape, lat, lon, 2020)
    _, so80 = ds.cropgrids_cropland_extent(cfg, over, shape, lat, lon, 1980)
    assert np.allclose(ds.area_in_year(over, so20["H_ref"], so20["H_t"]), over)
    assert np.allclose(ds.area_in_year(over, so80["H_ref"], so80["H_t"]), over)   # never cut below CG

    # 4. Coello weight: scaling dict == equivalent per-cell multiplier
    mask = np.array([[1, 1, 2, 2]] * 4, "int32"); idx2iso = {1: "AAA", 2: "BBB"}
    ha = np.zeros(shape); ha[1:, :] = 5.0; ha[0, 1] = 2.0
    tbl = pd.DataFrame({"ISO3": ["AAA", "BBB"], "coello_group": ["Maize", "Maize"], "fert_kg": [100., 50.]})
    Href = np.full(shape, 60.0); Href[0, 1] = 0.0
    Ht = np.full(shape, 90.0); Ht[2, 2] = 30.0
    scal = {"H_ref": Href, "H_t": Ht, "factor": None}
    W1 = ds.coello_cropgrids_weight(tbl, {"Maize": ha}, scal, None, None, mask, idx2iso)
    A = ds.area_in_year(ha, Href, Ht)
    with np.errstate(divide="ignore", invalid="ignore"):
        f = np.where(ha > 0, A / np.where(ha > 0, ha, 1.0), 0.0)
    W2 = ds.coello_cropgrids_weight(tbl, {"Maize": ha}, f, None, None, mask, idx2iso)
    assert np.allclose(W1, W2)
    assert np.isclose(W1[mask == 1].sum(), 100.) and np.isclose(W1[mask == 2].sum(), 50.)
    print("test_cropland_scaling: PASSED")


if __name__ == "__main__":
    main()
