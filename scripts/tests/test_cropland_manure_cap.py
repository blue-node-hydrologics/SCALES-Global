#!/usr/bin/env python3
"""
# Synthetic unit test for the cropland manure ceiling (`_cap_cropland_manure`)

Checks, without data files, that:

- no cell ends above the ceiling when the country has enough headroom;
- each country's manure total is conserved exactly, and no manure crosses a national border;
- excess moves first to cropland within the local window (a near cell with headroom is filled before
  a distant one), and only the remainder is spread over the country;
- a country without headroom keeps its excess (accepted above the ceiling, no N lost);
- `_boxsum` equals a brute-force window sum.

Usage (from the model root): `python scripts/tests/test_cropland_manure_cap.py`
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src import downscale as ds  # noqa: E402


def main():
    rng = np.random.default_rng(1)
    a = rng.random((9, 13))
    bs = ds._boxsum(a, 2)
    for i, j in [(0, 0), (4, 6), (8, 12), (3, 1)]:
        assert np.isclose(bs[i, j], a[max(i - 2, 0):i + 3, max(j - 2, 0):j + 3].sum())

    # country 1 (cols 0-29), country 2 (cols 30-39); one row of cells
    mask = np.ones((1, 40), "int32"); mask[0, 30:] = 2
    idx2iso = {1: "AAA", 2: "BBB"}
    A = np.full((1, 40), 10.0)                  # 10 ha cropland everywhere -> capacity 3500 kg at 350
    M = np.full((1, 40), 1000.0)
    M[0, 5] = 10000.0                            # 6500 kg excess in country 1, col 5
    M[0, 29] = 8000.0                            # excess next to the border (col 30 is country 2)
    out, st = ds._cap_cropland_manure(M, A, mask, idx2iso, 350.0, radius=2, verbose=False)
    for k in (1, 2):
        assert np.isclose(out[mask == k].sum(), M[mask == k].sum())     # national totals conserved
    assert out.max() <= 3500.0 + 1e-6                                   # nothing left above the ceiling
    assert np.allclose(out[0, 30:], 1000.0)                             # nothing crossed the border
    assert out[0, 4] > out[0, 15] and out[0, 6] > out[0, 15]            # neighbours filled first
    assert np.isclose(st["capped_kg"], 6500 + 4500)
    assert np.isclose(st["local_kg"] + st["national_kg"] + st["accepted_kg"], st["capped_kg"])

    # country with no headroom: everything above the ceiling, excess accepted, total conserved
    M2 = np.full((1, 40), 5000.0)
    out2, st2 = ds._cap_cropland_manure(M2, A, mask, idx2iso, 350.0, radius=2, verbose=False)
    assert np.isclose(out2.sum(), M2.sum()) and st2["accepted_kg"] > 0
    assert np.isclose(out2[mask == 2].sum(), M2[mask == 2].sum())

    # cell with manure but no cropland: all of it moves
    M3 = np.full((1, 40), 100.0); A3 = A.copy(); A3[0, 10] = 0.0; M3[0, 10] = 2000.0
    out3, _ = ds._cap_cropland_manure(M3, A3, mask, idx2iso, 350.0, radius=2, verbose=False)
    assert out3[0, 10] == 0.0 and np.isclose(out3[mask == 1].sum(), M3[mask == 1].sum())
    print("test_cropland_manure_cap: PASSED")


if __name__ == "__main__":
    main()
