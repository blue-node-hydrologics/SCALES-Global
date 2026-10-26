"""Cropland deposition weight: ISIMIP pattern, national total conserved, gaps filled, uniform fallback."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.downscale import cropland_dep_weight, allocate  # noqa: E402
import pandas as pd


def test_pattern_and_conservation():
    mask = np.array([[1, 1, 2, 2], [1, 1, 2, 2]])
    crop = np.array([[10., 10., 5., 5.], [10., 0., 5., 5.]])
    dep_w = np.array([[2., 8., np.nan, np.nan], [np.nan, 4., np.nan, np.nan]])   # country 2: no ISIMIP
    idx2iso = {1: "AAA", 2: "BBB"}
    w = cropland_dep_weight(dep_w, crop, mask, "isimip")
    # cell (1,0) has no ISIMIP rate -> country cropland mean (2*10+8*10)/20 = 5
    assert np.isclose(w[1, 0], 5 * 10)
    # country 2 uniform
    assert np.allclose(w[:, 2:] / crop[:, 2:], 1.0)
    nat = pd.DataFrame({"ISO3": ["AAA", "BBB"], "v": [30 * 7.0, 20 * 9.0]})      # WL rate x cropland
    g = allocate(nat, w, mask, idx2iso, "v")
    assert np.isclose(g[mask == 1].sum(), 210.0) and np.isclose(g[mask == 2].sum(), 180.0)
    rate = np.divide(g, crop, out=np.zeros_like(g), where=crop > 0)
    assert rate[0, 1] > rate[0, 0]                       # gradient follows ISIMIP
    assert np.allclose(rate[:, 2:], 9.0)                 # uniform where ISIMIP is missing


def test_uniform_mode():
    mask = np.ones((2, 2), int); crop = np.ones((2, 2))
    assert cropland_dep_weight(np.full((2, 2), 3.0), crop, mask, "uniform") is crop
    assert cropland_dep_weight(None, crop, mask) is crop
