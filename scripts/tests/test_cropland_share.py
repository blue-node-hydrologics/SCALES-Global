#!/usr/bin/env python3
"""
# Synthetic unit test for the 'share' crop-area formulation 

    A_c(m,t) = [CG_c(m) / CG_crop(m)] * H(m,t)

CG_c is CropGRIDS harvested area of crop c, CG_crop CropGRIDS physical cropland, H HYDE cropland. Checks:

- `cropgrids_cropland_extent` in mode 'share' returns H(m,t) as the cropland extent;
- the returned factor gives A_c = CG_c * factor = (CG_c / CG_crop) * H(m,t);
- summed crop area equals cropping intensity x H, so it never exceeds intensity x HYDE cropland;
- cells with CropGRIDS but no HYDE cropland get no crop area; cells with HYDE but no CropGRIDS get none;
- `coello_cropgrids_weight` with the factor conserves each country's group total.

Usage (from the model root): `python scripts/tests/test_cropland_share.py`
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src import downscale as ds  # noqa: E402


def main():
    lat = np.array([0.025, 0.075]); lon = np.array([0.025, 0.075]); shape = (2, 2)
    H = np.array([[80.0, 10.0], [0.0, 50.0]])          # HYDE cropland, year t
    ds.load_hyde_grid = lambda cfg, s, la, lo, year, layer: H.copy()
    cfg = {"paths": {"raw": "/tmp/none"}, "_root": "/", "hyde_envelope": {"mode": "share"}}
    cg_crop = np.array([[40.0, 1000.0], [30.0, 0.0]])   # CropGRIDS physical cropland
    cl, sc = ds.cropgrids_cropland_extent(cfg, cg_crop, shape, lat, lon, 1980)
    assert np.allclose(cl, H)                                           # extent = HYDE
    maize = np.array([[30.0, 900.0], [30.0, 0.0]]); rice = np.array([[30.0, 600.0], [0.0, 0.0]])
    A_m = maize * sc["factor"]; A_r = rice * sc["factor"]
    assert np.isclose(A_m[0, 0], 30 / 40 * 80) and np.isclose(A_r[0, 0], 30 / 40 * 80)   # 60 each
    assert np.isclose(A_m[0, 1], 900 / 1000 * 10)                       # CG >> HYDE: bounded by H
    assert np.isclose((A_m + A_r)[0, 0], (60 / 40) * 80)                # intensity 1.5 x H
    assert A_m[1, 0] == 0 and A_m[1, 1] == 0                            # no HYDE / no CropGRIDS
    mask = np.array([[1, 1], [1, 1]], "int32"); idx2iso = {1: "AAA"}
    tbl = pd.DataFrame({"ISO3": ["AAA"], "coello_group": ["Maize"], "fert_kg": [100.0]})
    W = ds.coello_cropgrids_weight(tbl, {"Maize": maize}, sc, None, None, mask, idx2iso)
    assert np.isclose(W.sum(), 100.0)
    assert np.isclose(W[0, 0] / W[0, 1], A_m[0, 0] / A_m[0, 1])
    print("test_cropland_share: PASSED")


if __name__ == "__main__":
    main()
