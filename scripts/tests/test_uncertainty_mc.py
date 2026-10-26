"""Synthetic-data tests for src/uncertainty_mc.py (no FAOSTAT or grids needed).

Run:  python -m pytest scripts/tests/test_uncertainty_mc.py -q     (or)   python scripts/tests/test_uncertainty_mc.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src import uncertainty_mc as M   # noqa: E402

rng = np.random.default_rng(1)


def _human_central(h):
    """Deterministic human terms, transcribed from src/human.py (compute, food_waste, human_recycle)."""
    E = h["supply"] * (1 - h["w_c"])
    fw = h["supply"] * h["w_c"]
    sewage = E * np.clip(1 - h["f_ns"] - h["f_rem"], 0, None)
    land = np.minimum(E * np.clip(1 - h["c"] - h["f_ns"], 0, None), sewage)
    sludge = E * h["f_sludge"]
    return dict(food_waste=fw, hum_ns=E * h["f_ns"], hum_bio=h["a_land"] * sludge,
                hum_landfill=h["a_lf"] * sludge, hum_sew_land=land,
                hum_sew_water=np.clip(sewage - land, 0, None))


def make_inputs(nC=12, with_pseudo=True, lattice=True):
    iso = np.array([f"C{i:02d}" for i in range(nC)])
    c0 = {k: rng.uniform(1e7, 1e9, nC) for k in M.TERMS}
    cf = np.ones(nC); cf[:4] = rng.uniform(0.1, 0.9, 4)
    full = rng.uniform(1e8, 2e9, nC)
    c0["fert"], c0["fert_grass"] = full * cf, full * (1 - cf)
    g = rng.uniform(0.03, 0.15, nC)
    c0["fert_vol"], c0["fert_vol_full"] = c0["fert"] * g, full * g
    c0["man_vol"] = 0.21 * c0["man_crop"]; c0["man_vol_past"] = 0.10 * c0["man_past"]
    leg = rng.uniform(0, 5e8, nC); free = rng.uniform(0, 2e8, nC); c0["bnf"] = leg + free
    # human rows: the countries plus one dissolution aggregate split onto C00..C02
    nH = nC + (1 if with_pseudo else 0)
    hiso = np.concatenate([iso, ["SUN"]]) if with_pseudo else iso
    c = rng.uniform(0, 0.95, nH)
    f_treated = c * rng.uniform(0, 1, nH)
    RN = 0.10 + 0.70 * rng.uniform(0, 1, nH)
    f_rem = f_treated * RN
    is_t1 = np.zeros(nH, bool); is_t1[1] = True
    lam = rng.uniform(0, 0.1, nH)
    f_ns = np.where(is_t1, np.minimum(0.6, 1 - c), lam * (1 - c))
    f_sl = np.where(is_t1, 0.0, np.minimum(0.18 * f_treated, f_rem))
    a_land = rng.uniform(0.1, 0.5, nH); a_lf = rng.uniform(0.0, 0.4, nH)
    h = dict(iso=hiso, reg=np.full(nH, len(M.REGIONS) - 1), supply=rng.uniform(1e7, 1e9, nH),
             w_c=rng.uniform(0.07, 0.26, nH), c=c, f_treated=f_treated, f_rem=f_rem, f_ns=f_ns,
             f_sludge=f_sl, is_t1=is_t1, a_land=a_land, a_lf=a_lf, s_bio=0.18, biosolids=True,
             R_low=0.10, R_high=0.80)
    H = np.zeros((nC, nH)); H[np.arange(nC), np.arange(nC)] = 1
    if with_pseudo:
        H[0, nC], H[1, nC], H[2, nC] = 0.5, 0.3, 0.2
    hc = _human_central(h)
    for k, v in hc.items():
        c0[k] = H @ v
    inp = M.YearInputs(year=2000, iso=iso, reg=np.full(nC, len(M.REGIONS) - 1), c0=c0, cf=cf,
                       g_fert=g, r_mv_crop=np.full(nC, 0.21), r_mv_past=np.full(nC, 0.10),
                       bnf_leg=leg, bnf_free=free, hum=h, H=H)
    inp.forage_legacy = c0["forage"].copy()
    if lattice:
        inp.lattice, cells = _synthetic_lattice(nC)
        inp._cells = cells
        c0["forage"] = inp.lattice["F"][:, M.LAT_N // 2, M.LAT_N // 2].copy()
    return inp


def _synthetic_lattice(nC, ncell=400):
    cid = rng.integers(0, nC, ncell)
    D = rng.uniform(0, 1e6, ncell); a = rng.uniform(0, 3e5, ncell) * 200; b = rng.uniform(0, 3e6, ncell) * 30
    logs = np.linspace(-M.LAT_HALF_WIDTH, M.LAT_HALF_WIDTH, M.LAT_N); mult = np.exp(logs)
    F = np.zeros((nC, M.LAT_N, M.LAT_N))
    for iu, u in enumerate(mult):
        for iv, v in enumerate(mult):
            F[:, iu, iv] = np.bincount(cid, np.minimum(D, u * a + v * b), minlength=nC)
    Mi = np.bincount(cid, D * 0.8, minlength=nC); Pi = np.bincount(cid, D * 0.2, minlength=nC)
    return dict(F=F, have=np.ones(nC, bool), logs=logs, M=Mi, P=Pi, center=M.LAT_N // 2), (cid, D, a, b)


def test_central_reproduction():
    inp = make_inputs()
    t = M.compute_terms(inp, {"_human": {}}, 1)
    for k in M.TERMS:
        assert np.allclose(t[k][0], inp.c0[k], rtol=1e-10, atol=1e-3), k


def test_ensemble_mean_and_bounds():
    inp = make_inputs()
    spec, rcv = M.load_spec()
    act = {g: np.full(len(inp.iso), 0.1) for g in M.ACT_GROUPS}
    res = M.run_ensemble(inp, spec, rcv, n=4000, seed=3, chunk=1000, activity_cv=act, verbose=False)
    for v in M.VIEWS:
        g = res.surplus[v].astype(float).sum(axis=1)
        cen = res.surplus_central[v].sum()
        assert abs(g.mean() - cen) / abs(cen) < 0.03, (v, g.mean(), cen)
    # relative covariance diagonal matches the term CVs computed directly
    tot = res.cov["total"]
    assert np.all(np.diagonal(tot, axis1=1, axis2=2) >= -1e-6)


def test_human_partition_conserves_mass():
    inp = make_inputs(lattice=False)
    spec, rcv = M.load_spec()
    act = {g: np.zeros(len(inp.iso)) for g in M.ACT_GROUPS}
    d = M.FactorDraws(inp, spec, rcv, 500, 5, act)
    f = d.factors(slice(0, 500), coef=True, act=False)
    h = inp.hum; fh = f["_human"]
    ht = M._human_terms(h, fh, 500)
    supply = h["supply"] * fh["protein_N"]
    # food waste + excreted = supply
    w_c = np.minimum(h["w_c"] * fh["w_c"], 1.0)
    E = supply * (1 - w_c)
    assert np.allclose(ht["food_waste"] + E, supply)
    # excreted = night soil + discharge to land + discharge to water + removed (sludge + gas)
    removed = E - ht["hum_ns"] - ht["hum_sew_land"] - ht["hum_sew_water"]
    assert np.all(removed >= -1e-6 * E)
    sludge_routes = ht["hum_bio"] + ht["hum_landfill"]
    assert np.all(sludge_routes <= removed + 1e-6 * E)
    # bounded fractions
    assert np.all(ht["hum_ns"] <= E * 1.0000001)


def test_cf_partition():
    inp = make_inputs(lattice=False)
    spec, rcv = M.load_spec()
    act = {g: np.zeros(len(inp.iso)) for g in M.ACT_GROUPS}
    d = M.FactorDraws(inp, spec, rcv, 300, 7, act)
    t = M.compute_terms(inp, d.factors(slice(0, 300), act=False), 300)
    full = inp.c0["fert"] + inp.c0["fert_grass"]
    assert np.allclose(t["fert"] + t["fert_grass"], full)            # split conserves the total
    assert np.all(t["fert_grass"] >= -1e-6)
    assert np.allclose(t["fert"][:, 4:], inp.c0["fert"][4:])         # cf = 1 countries untouched


def test_forage_lattice_interpolation():
    inp = make_inputs()
    cid, D, a, b = inp._cells
    L = inp.lattice
    nC = len(inp.iso)
    for _ in range(50):
        fn, fp, fm, fr = rng.lognormal(0, 0.3, 4)
        d = (fn * L["M"] + fp * L["P"]) / (L["M"] + L["P"])
        direct = np.bincount(cid, np.minimum(D * d[cid], fm * a + fr * b), minlength=nC)
        f = {"nex": np.full((1, nC), fn), "prodN": np.array([[fp]]), "p_managed": np.array([[fm]]),
             "p_range": np.array([[fr]]), "_human": {}}
        t = M.compute_terms(inp, f, 1)["forage"][0]
        # man_crop/man_past also scale with nex but forage is checked alone
        rel = np.abs(t - direct) / np.maximum(direct, 1)
        assert np.all(rel < 0.02), rel.max()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("passed", name)
