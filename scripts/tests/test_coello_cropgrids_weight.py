#!/usr/bin/env python3
"""
# Synthetic unit test for the `coello_cropgrids` fertilizer weight

Checks `src.downscale.coello_cropgrids_weight` (and its helpers) on a tiny hand-built grid,
without any data files:

- per-country weight sums equal Σ_g F_C(g,k) (the national Coello group total);
- a cell's weight equals the hand-computed Σ_g F_C(g,k)·H_g(m)s(m)/Σ_k H_g s;
- fallback (a): a group with no area in its country goes to all-crop harvarea·s, then to cropland;
- countries with no Coello rows get zero weight and are backfilled by `_fill_zero_weight_countries`;
- sparse `(cells, ha)` and dense 2-D `group_ha` inputs give identical results;
- `load_coello_group_table` gating, year clamping and CSV caching;
- `coello_group_ha_from_cache` maps cache FAO codes -> CropGRIDS slugs -> Coello groups.

Usage (from the model root):  `python scripts/tests/test_coello_cropgrids_weight.py`
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src import downscale as ds  # noqa: E402


def close(a, b, tol=1e-9):
    return abs(a - b) <= tol * max(1.0, abs(b))


def test_weight():
    # 4 x 7 grid: cols 0-1 AAA (idx 1), cols 2-3 BBB (idx 2), col 4 CCC (idx 3), col 5 DDD (idx 4),
    # col 6 ocean (idx 0)
    mask = np.zeros((4, 7), "int32")
    mask[:, 0:2] = 1; mask[:, 2:4] = 2; mask[:, 4] = 3; mask[:, 5] = 4
    idx2iso = {1: "AAA", 2: "BBB", 3: "CCC", 4: "DDD"}
    H_maize = np.zeros((4, 7)); H_wheat = np.zeros((4, 7))
    H_maize[0, 0] = 10; H_maize[1, 1] = 30; H_maize[2, 2] = 5; H_maize[3, 3] = 15
    H_maize[0, 6] = 99                                   # ocean cell: must be ignored
    H_wheat[0, 0] = 20; H_wheat[2, 1] = 20               # AAA only; BBB has NO wheat area
    s = np.ones((4, 7)); s[0, 0] = 0.5; s[1, 1] = 2.0; s[3, 3] = 3.0
    cropland = np.zeros((4, 7)); cropland[:, 4] = [1, 1, 2, 0]; cropland[:, 5] = 1.0
    tbl = pd.DataFrame({"ISO3": ["AAA", "AAA", "BBB", "BBB", "CCC", "ZZZ"],
                        "coello_group": ["Maize", "Wheat", "Maize", "Wheat", "Rice", "Maize"],
                        "fert_kg": [100.0, 60.0, 40.0, 25.0, 8.0, 7.0]})
    tbl.attrs.update(year_used=2019, year_req=2023, tag="N")
    dense = {"Maize": H_maize, "Wheat": H_wheat}
    W = ds.coello_cropgrids_weight(tbl, dense, s, cropland, None, mask, idx2iso)

    # per-country sums = Σ_g F_C
    nat = np.bincount(mask.ravel(), weights=W.ravel(), minlength=5)
    assert close(nat[1], 160.0), nat
    assert close(nat[2], 65.0), nat          # Wheat 25 via fallback (a) to all-crop harvarea
    assert close(nat[3], 8.0), nat           # Rice 8: no harvarea at all in CCC -> cropland
    assert nat[4] == 0.0 and nat[0] == 0.0, nat
    # hand-computed cells (AAA): maize den = 10*0.5 + 30*2 = 65; wheat den = 20*0.5 + 20 = 30
    assert close(W[0, 0], 100 * 5 / 65 + 60 * 10 / 30), W[0, 0]
    assert close(W[1, 1], 100 * 60 / 65), W[1, 1]
    assert close(W[2, 1], 60 * 20 / 30), W[2, 1]
    # BBB: maize den = 5*1 + 15*3 = 50; wheat fallback over all-crop (= maize here) x s, same shares
    assert close(W[2, 2], 40 * 5 / 50 + 25 * 5 / 50), W[2, 2]
    assert close(W[3, 3], 40 * 45 / 50 + 25 * 45 / 50), W[3, 3]
    # CCC: cropland share
    assert close(W[2, 4], 8 * 2 / 4), W[2, 4]
    assert W[0, 6] == 0.0
    print("  [ok] per-country sums and hand-computed cells (incl. fallback (a) both levels)")

    # sparse input identical
    sp = {g: (np.nonzero(a.ravel() > 0)[0], a.ravel()[a.ravel() > 0]) for g, a in dense.items()}
    W2 = ds.coello_cropgrids_weight(tbl, sp, s, cropland, None, mask, idx2iso)
    assert np.allclose(W, W2)
    # explicit all_ha equals the Σ_g H_g default
    W3 = ds.coello_cropgrids_weight(tbl, sp, s, cropland, H_maize + H_wheat, mask, idx2iso)
    assert np.allclose(W, W3)
    print("  [ok] sparse == dense input; explicit all_ha == default")

    # s_env None -> H_g alone
    W4 = ds.coello_cropgrids_weight(tbl, dense, None, cropland, None, mask, idx2iso)
    assert close(W4[0, 0], 100 * 10 / 40 + 60 * 20 / 40)
    print("  [ok] s_env=None uses H_g alone")

    # no-Coello country DDD -> filled from fallback by the caller's existing helper
    fb = np.ones((4, 7))
    Wf = ds._fill_zero_weight_countries(W, fb, mask)
    assert (Wf[:, 5] == 1.0).all() and np.allclose(Wf[:, :5], W[:, :5])
    # allocate() then conserves a national total exactly and keeps the group-split geography
    nat_tbl = pd.DataFrame({"ISO3": ["AAA", "BBB", "CCC", "DDD"], "v": [1000.0, 50.0, 3.0, 4.0]})
    g = ds.allocate(nat_tbl, Wf, mask, idx2iso, "v")
    gn = np.bincount(mask.ravel(), weights=g.ravel(), minlength=5)
    assert np.allclose(gn[1:], [1000.0, 50.0, 3.0, 4.0])
    assert close(g[0, 0] / 1000.0, W[0, 0] / 160.0)
    print("  [ok] zero-weight backfill + allocate() conservation")


def test_table_loader():
    with tempfile.TemporaryDirectory() as td:
        cfg = {"fertilizer": {"grid_weight": "coello_cropgrids"}, "paths": {"raw": td}, "_root": "/"}
        d = Path(td) / ds.COE_SUBDIR; d.mkdir(parents=True)
        pd.DataFrame({"ISO3": ["AAA", "AAA", "BBB"], "year": [2019, 2000, 2019],
                      "coello_group": ["Maize", "Maize", "Wheat"], "fert_kg": [5.0, 3.0, 0.0],
                      "unit": "kg N"}).to_csv(d / "coello_group_national_N.csv", index=False)
        t = ds.load_coello_group_table(cfg, 2023, "N")
        assert t is not None and list(t.columns) == ["ISO3", "coello_group", "fert_kg"]
        assert len(t) == 1 and t.attrs["year_used"] == 2019 and t.attrs["year_req"] == 2023
        assert ds.load_coello_group_table(cfg, 1950, "N") is None          # clamps to 1961: no rows
        assert ds.load_coello_group_table(cfg, 2000, "N")["fert_kg"].iloc[0] == 3.0
        assert ds.load_coello_group_table(cfg, 2019, "P") is None          # P2O5 table missing
        cfg2 = dict(cfg, fertilizer={"grid_weight": "coello"})
        assert ds.load_coello_group_table(cfg2, 2019, "N") is None         # mode gate
    print("  [ok] load_coello_group_table gating / clamping / caching")


def test_group_ha_from_cache():
    xw = pd.read_csv(ROOT / "coefficients/cropgrids_item_crosswalk.csv")
    code = dict(zip(xw["slug"], xw["fao_item_code"].astype(int)))
    cfg = {"paths": {"coefficients": str(ROOT / "coefficients")}, "_root": "/"}
    cache = {"cells": np.array([0, 1, 1, 2, 3], "int32"), "ha": np.array([1, 2, 4, 8, 16], "float32"),
             "ptr": np.array([0, 2, 4, 5]), "codes": np.array([code["maize"], code["wheat"], code["maize"]]),
             "mask": np.ones((2, 2), "int32")}
    gha = ds.coello_group_ha_from_cache(cfg, cache)
    assert set(gha) == {"Maize", "Wheat"}, gha.keys()
    c, h = gha["Maize"]
    assert sorted(zip(c.tolist(), h.tolist())) == [(0, 1.0), (1, 2.0), (3, 16.0)]
    print("  [ok] coello_group_ha_from_cache code -> slug -> group")


if __name__ == "__main__":
    test_weight()
    test_table_loader()
    test_group_ha_from_cache()
    print("ALL PASSED")
