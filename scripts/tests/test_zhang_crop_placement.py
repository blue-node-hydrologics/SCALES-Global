#!/usr/bin/env python3
"""
# Synthetic unit test for the Zhang cropland-manure placement (12c, `--placement hyde`)

Uses `build_placed` from `scripts/12c_build_zhang_manure.py` with a HYDE-like primary weight and a
CropGRIDS-like fallback, on a reduced Zhang grid, without data files. Checks that:

- manure mass is conserved within each Zhang (coarse) cell, so it stays local;
- inside a coarse cell with HYDE cropland, mass follows the HYDE weight;
- with a fallback, a coarse cell with no HYDE cropland falls back to the CropGRIDS-like weight, then to
  cell area (legacy ladder);
- in the default hyde placement (no fallback), a coarse cell with no HYDE cropland places nothing.

Usage (from the model root): `python scripts/tests/test_zhang_crop_placement.py`
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("zh", ROOT / "scripts" / "12c_build_zhang_manure.py")
zh = importlib.util.module_from_spec(spec)
spec.loader.exec_module(zh)


def main():
    # reduced grid: 2x2 coarse cells of 0.1 deg; model cells 0.05 deg -> each coarse cell = 2x2 model cells
    zh.ZH_NROWS, zh.ZH_NCOLS, zh.ZH_CELLSIZE, zh.ZH_YLL, zh.ZH_XLL = 2, 2, 0.1, 0.0, 0.0
    lat = np.array([0.025, 0.075, 0.125, 0.175]); lon = lat.copy()
    shape = (4, 4)
    combos = zh.build_overlap(lat, lon)
    cell = np.ones(shape)
    area5 = np.ones(2)                                         # 1 km2 per coarse cell -> mass = density
    dens = np.array([[10., 20.], [30., 40.]])                  # coarse cells: (r0,c0)=10 ... (r1,c1)=40
    hyde = np.zeros(shape); hyde[0, 0] = 1; hyde[1, 1] = 3      # only coarse cell (0,0) has HYDE cropland
    cg = np.zeros(shape); cg[0, 2] = 2; cg[0, 3] = 2            # coarse cell (0,1) has CropGRIDS only
    out = zh.build_placed(dens, hyde, combos, cell, area5, shape, fallback_area=cg)

    # per coarse cell mass conservation
    for r in range(2):
        for c in range(2):
            assert np.isclose(out[2*r:2*r+2, 2*c:2*c+2].sum(), dens[r, c]), (r, c)
    # coarse (0,0): HYDE 1:3 -> 2.5, 7.5
    assert np.isclose(out[0, 0], 2.5) and np.isclose(out[1, 1], 7.5) and out[0, 1] == 0 and out[1, 0] == 0
    # coarse (0,1): no HYDE -> CropGRIDS 2:2 -> 10, 10 on row 0
    assert np.isclose(out[0, 2], 10) and np.isclose(out[0, 3], 10) and out[1, 2] == 0
    # coarse (1,*): neither -> cell area, uniform
    assert np.allclose(out[2:4, 0:2], 7.5) and np.allclose(out[2:4, 2:4], 10.0)
    # default hyde placement : HYDE only, no fallback -> mass of Zhang cells without HYDE unplaced
    out2 = zh.build_placed(dens, hyde, combos, cell, area5, shape, fallback_area=None, area_fallback=False)
    assert np.isclose(out2[0:2, 0:2].sum(), dens[0, 0]) and np.isclose(out2[0, 0], 2.5)
    assert np.isclose(out2.sum(), dens[0, 0])                  # only the HYDE-bearing coarse cell is placed
    print("test_zhang_crop_placement: PASSED")


if __name__ == "__main__":
    main()
