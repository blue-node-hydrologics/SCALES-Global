"""Synthetic test for scripts/47: per-cell closed-form variance equals a brute-force Monte Carlo of
cells that are fixed shares of correlated national terms."""
import sys, importlib.util, tempfile
from pathlib import Path
import numpy as np, xarray as xr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src import uncertainty_mc as MC   # noqa: E402

spec = importlib.util.spec_from_file_location("s47", ROOT / "scripts/47_gridded_uncertainty_cov.py")
s47 = importlib.util.module_from_spec(spec); spec.loader.exec_module(s47)


def test_grid_variance_matches_bruteforce():
    rng = np.random.default_rng(2)
    ny, nx, nC, K, N = 20, 30, 3, len(MC.TERMS), 20000
    lat = np.linspace(-9.5, 9.5, ny); lon = np.linspace(-14.5, 14.5, nx)
    mask = rng.integers(1, nC + 1, (ny, nx)); mask[:2] = 0
    idx2iso = {i: f"C{i}" for i in range(1, nC + 1)}
    s47.load_country_grid = lambda cfg: (mask, idx2iso, lat, lon)
    # national ensemble: correlated terms
    A = rng.normal(size=(K, K)) * 0.1
    nat_draws = np.exp(rng.normal(size=(N, nC, K)) @ A.T * 0.5)          # (N, nC, K)
    nat_draws *= rng.uniform(1e6, 1e8, (1, nC, K))
    mu = nat_draws.mean(0); cov = np.einsum("nck,ncl->ckl", nat_draws - mu, nat_draws - mu) / N
    Crel = cov / (mu[:, :, None] * mu[:, None, :])
    # cell shares per term
    w = rng.uniform(0, 1, (K, ny, nx)) * (mask > 0)
    for k in range(K):
        for c in range(1, nC + 1):
            m = mask == c
            w[k][m] /= w[k][m].sum()
    layers = {}
    for k, t in enumerate(MC.TERMS):
        nat = np.zeros((ny, nx))
        for c in range(1, nC + 1):
            nat[mask == c] = mu[c - 1, k]
        layers[t] = w[k] * nat
    dvars = {s47.LAYER[t]: (("lat", "lon"), layers[t].astype("float64")) for t in MC.TERMS if s47.LAYER[t]}
    dvars["cropland_ha"] = (("lat", "lon"), np.full((ny, nx), 1000.0))
    tmp = Path(tempfile.mkdtemp())
    xr.Dataset(dvars, coords={"lat": lat, "lon": lon}).to_netcdf(tmp / "surplus_N_2000_0p05.nc")
    # unsaved layers: make them the national ratio x base layer so reconstruction is exact
    for t, (num, dens) in s47.RECON.items():
        if s47.LAYER.get(t):
            continue
        base = sum(layers[d] for d in dens)
        for c in range(1, nC + 1):
            m = mask == c
            r = mu[c - 1, MC.TERMS.index(num)] / sum(mu[c - 1, MC.TERMS.index(d)] for d in dens)
            layers[t][m] = base[m] * r
    np.savez(tmp / "cov_N_2000.npz", iso=np.array([idx2iso[i] for i in range(1, nC + 1)]),
             terms=np.array(MC.TERMS), central=mu, C_total=Crel, C_coef=Crel, C_act=Crel,
             mean_total=mu, mean_coef=mu, mean_act=mu)
    s47.run({}, 2000, tmp, tmp, tmp)
    o = xr.open_dataset(tmp / "uncertainty_N_2000_0p05.nc")
    # brute force for the total view: cell value = share x national draw
    sg = np.array([MC.VIEWS["total"].get(t, 0) for t in MC.TERMS])
    cells = np.argwhere(mask > 0)[:25]
    for (i, j) in cells:
        c = mask[i, j] - 1
        x = np.array([layers[t][i, j] for t in MC.TERMS]) / mu[c]          # share
        draws = (nat_draws[:, c, :] * x) @ sg
        cell_ha = float(o["surplus_full_sd_kgha"].values[i, j])
        R = 6371000.0; dla = lat[1] - lat[0]; dlo = lon[1] - lon[0]
        area = abs(R**2 * np.deg2rad(dlo) * (np.sin(np.deg2rad(lat[i] + dla / 2)) - np.sin(np.deg2rad(lat[i] - dla / 2)))) / 1e4
        assert abs(cell_ha * area - draws.std()) / draws.std() < 1e-6, (cell_ha * area, draws.std())


if __name__ == "__main__":
    test_grid_variance_matches_bruteforce(); print("passed")
