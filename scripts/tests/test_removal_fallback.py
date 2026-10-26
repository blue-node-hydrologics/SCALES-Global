#!/usr/bin/env python3
"""
# Synthetic unit test for crop-removal / BNF residual placement

Checks `src.downscale.place_national_residual` on a tiny hand-built grid, without data files:

- a country whose national total is already fully placed is left bit-identical;
- a residual (item with no CropGRIDS layer, or a crop with no area in the country) is spread over the
  country's all-crop harvested area, in proportion to that area;
- a country with no harvested area falls back to cropland; with neither, the mass is reported unplaced;
- national ISO3 codes absent from the mask are reported, not placed, and do not raise;
- after placement, every country with a placeable residual sums exactly to its national total.

Usage (from the model root): `python scripts/tests/test_removal_fallback.py`
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src import downscale as ds  # noqa: E402


def main():
    # 3x4 grid; countries: 1=AAA (cols 0-1), 2=BBB (col 2), 3=CCC (col 3, rows 0-1), 0=ocean (row 2, col 3)
    mask = np.array([[1, 1, 2, 3],
                     [1, 1, 2, 3],
                     [1, 1, 2, 0]], "int32")
    idx2iso = {1: "AAA", 2: "BBB", 3: "CCC"}
    placed = np.array([[10., 30., 0., 0.],
                       [0., 0., 0., 0.],
                       [0., 0., 0., 0.]])                       # AAA has 40 placed; others nothing
    all_ha = np.array([[1., 3., 0., 0.],
                       [0., 0., 0., 0.],
                       [0., 4., 0., 0.]])                       # BBB, CCC have no harvested area
    cropland = np.array([[1., 1., 2., 0.],
                         [1., 1., 6., 0.],
                         [1., 1., 0., 5.]])                     # CCC cells on land have no cropland
    national = pd.DataFrame({"ISO3": ["AAA", "AAA", "BBB", "CCC", "ZZZ", "AAA"],
                             "value_kg": [40., 16., 8., 3., 7., np.nan]})
    out, st = ds.place_national_residual(placed, national, all_ha, cropland, mask, idx2iso, "TEST")

    # AAA: residual 16 over all_ha (1,3,4 on its cells; total 8) -> +2, +6, +8
    exp = placed.copy(); exp[0, 0] += 2; exp[0, 1] += 6; exp[2, 1] += 8
    # BBB: no all_ha -> cropland (2, 6) -> +2, +6
    exp[0, 2] += 2; exp[1, 2] += 6
    assert np.allclose(out, exp), out
    assert np.isclose(st["to_all_ha_kg"], 16) and np.isclose(st["to_cropland_kg"], 8)
    assert np.isclose(st["unplaced_kg"], 3) and st["unplaced_iso"] == ["CCC"], st
    assert np.isclose(st["off_mask_kg"], 7) and st["off_mask_iso"] == ["ZZZ"], st
    assert out[2, 3] == 0.0                                       # ocean untouched
    for k, iso in idx2iso.items():
        if iso == "CCC":
            continue
        tot = national.loc[national.ISO3 == iso, "value_kg"].sum()
        assert np.isclose(out[mask == k].sum(), tot), iso

    # nothing to place -> bit-identical, and callable all_ha never evaluated
    full = pd.DataFrame({"ISO3": ["AAA"], "value_kg": [40.]})
    calls = []
    out2, st2 = ds.place_national_residual(placed, full, lambda: calls.append(1) or all_ha, cropland,
                                           mask, idx2iso, "TEST")
    assert np.array_equal(out2, placed) and not calls and st2["n_countries"] == 0

    # flat input keeps flat shape
    out3, _ = ds.place_national_residual(placed.ravel(), national, all_ha.ravel(), cropland, mask,
                                         idx2iso, "TEST")
    assert out3.shape == placed.ravel().shape and np.allclose(out3, exp.ravel())
    print("test_removal_fallback: PASSED")


if __name__ == "__main__":
    main()
