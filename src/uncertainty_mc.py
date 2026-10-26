"""SCALES-Global — coefficient-level Monte Carlo uncertainty (replaces the term-level shortcut).

Principle
---------
In every realization, each coefficient is drawn from its own distribution and every national budget
term is recomputed from the drawn coefficients with the model's own equations. A coefficient that
appears in several terms therefore takes the same value in each (cN(j) in crop removal and legume
fixation; Nex in applied and grazing manure; c(i,t) in every human pathway), and partitioned
quantities keep summing to their totals (fertilizer split by cf(i); excreted N split by c, f_treat,
R_N, sl_bio, the sludge routes and lambda_ns).

Implementation
--------------
The deterministic national components are built once per year by the same code as the deterministic
run (pipeline.build_components, bnf.components, human._human_budget). Each term is then written as its
central value times the relevant coefficient ratios. For terms that are products of coefficients and
activity data this is exact; the nonlinear parts (bounded fractions, the human-waste partition, the
grid-scale minimum in forage removal) are recomputed explicitly. With every factor equal to one the
code reproduces the deterministic national components (checked by `validate_central`).

  * Coefficients and CVs: coefficients/uncertainty_spec.csv (one row per coefficient) and
    coefficients/uncertainty_region_cv.csv (regional CVs by IPCC-Nex region).
  * Draws: Latin hypercube sampling of standard normal deviates, transformed to mean-one lognormal
    factors with the coefficient's CV, truncated to [0.2, 5]; fractions are bounded at one.
  * Global coefficients: one value per realization. Regional: one value per IPCC-Nex region.
  * Activity factors: unchanged from src/uncertainty.py (one per activity group per country, CV from
    the FAOSTAT-flag lookup, Table 9).
  * Forage: Eq. 29 is evaluated cell by cell. For each year the national forage total of every country
    is tabulated on a lattice of (p_managed, p_range) multipliers relative to a demand scale; because
    FORAGE is homogeneous of degree one in (demand, supply), F(d.D, pm, pr) = d.F(D, pm/d, pr/d), so a
    two-dimensional lattice covers the demand factor as well. Each realization interpolates the lattice.

Outputs per year (written by scripts/46_run_uncertainty_mc.py): per-country and global surplus
statistics for the three views, global statistics for each term, and per-country relative covariance
matrices of the terms (total, coefficient-only and activity-only), which the gridded layer uses.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SPEC_FILE = ROOT / "coefficients/uncertainty_spec.csv"
REGION_CV_FILE = ROOT / "coefficients/uncertainty_region_cv.csv"
REGION_CROSSWALK = ROOT / "coefficients/ipcc_nex_region_crosswalk.csv"

REGIONS = ["North America", "Western Europe", "Eastern Europe", "Oceania", "Latin America",
           "Africa", "Middle East", "Asia", "Indian Subcontinent", "OTHER"]
# FAOSTAT dissolution aggregates (pre-split human-budget rows) -> IPCC-Nex region
PSEUDO_REGION = {"SUN": "Eastern Europe", "YUG": "Eastern Europe", "SCG": "Eastern Europe",
                 "CSK": "Eastern Europe", "BLX": "Western Europe", "ETX": "Africa", "SDX": "Africa"}

# national budget terms carried through the ensemble (kg N per country)
TERMS = ["fert", "fert_grass", "fert_vol", "fert_vol_full", "crop", "bnf", "bnf_grass", "dep",
         "dep_land", "man_crop", "man_past", "man_vol", "man_vol_past", "forage", "food_waste",
         "hum_ns", "hum_bio", "hum_landfill", "hum_sew_land", "hum_sew_water"]

# surplus views (mass; signs) — mirror src/mass_balance.assemble
VIEWS = {
    "cropland": {"fert": 1, "bnf": 1, "dep": 1, "man_crop": 1, "hum_ns": 1, "hum_bio": 1,
                 "crop": -1, "fert_vol": -1, "man_vol": -1},
    "landscape": {"fert": 1, "fert_grass": 1, "bnf": 1, "bnf_grass": 1, "dep_land": 1, "man_crop": 1,
                  "man_past": 1, "food_waste": 1, "hum_ns": 1, "hum_bio": 1, "hum_sew_land": 1,
                  "hum_landfill": 1, "crop": -1, "forage": -1, "fert_vol_full": -1, "man_vol": -1,
                  "man_vol_past": -1},
}
VIEWS["total"] = {**VIEWS["landscape"], "hum_sew_water": 1}

ACT_GROUPS = ["fert", "crop", "bnf", "man", "forage", "pop"]
_FMIN, _FMAX = 0.2, 5.0

# forage lattice: multipliers of (p_managed, p_range) relative to the demand scale, log-spaced,
# symmetric so that the central value (1, 1) is a lattice node
LAT_HALF_WIDTH = 1.8          # exp(+-1.8) = 0.165 .. 6.05
LAT_N = 25


# --------------------------------------------------------------------------------------------- #
# Specification
# --------------------------------------------------------------------------------------------- #
@dataclass
class Coef:
    name: str
    cv: float
    scope: str                      # 'global' | 'regional'
    cv_class: str = ""
    upper: Optional[float] = None


def load_spec(spec_file: Path = SPEC_FILE, region_file: Path = REGION_CV_FILE):
    s = pd.read_csv(spec_file)
    spec = {}
    for _, r in s.iterrows():
        spec[str(r["coef"])] = Coef(
            name=str(r["coef"]),
            cv=float(r["cv"]) if pd.notna(r["cv"]) else float("nan"),
            scope=str(r["scope"]).strip().lower(),
            cv_class=str(r["cv_class"]) if pd.notna(r["cv_class"]) else "",
            upper=float(r["upper_bound"]) if pd.notna(r["upper_bound"]) else None)
    missing = [k for k in GLOBAL_KEYS + REGIONAL_KEYS if k not in spec]
    if missing:
        raise ValueError(f"{spec_file.name} has no row for: {', '.join(missing)}")
    rcv = pd.read_csv(region_file).set_index("region")
    for reg in REGIONS:
        if reg not in rcv.index:
            raise ValueError(f"{region_file.name} lacks a row for region '{reg}'")
    return spec, rcv


def region_index(iso_list) -> np.ndarray:
    """ISO3 -> index into REGIONS (dissolution aggregates mapped explicitly; unmatched -> OTHER)."""
    mp = {}
    if REGION_CROSSWALK.exists():
        cw = pd.read_csv(REGION_CROSSWALK)
        mp = dict(zip(cw["iso3"].astype(str), cw["ipcc_nex_region"].astype(str)))
    mp.update(PSEUDO_REGION)
    idx = {r: i for i, r in enumerate(REGIONS)}
    return np.array([idx.get(mp.get(str(c), "OTHER"), idx["OTHER"]) for c in iso_list], dtype=int)


# --------------------------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------------------------- #
def lhs_normals(n: int, ncols: int, rng: np.random.Generator) -> np.ndarray:
    """(n, ncols) standard normal deviates, Latin-hypercube stratified independently per column."""
    from scipy.special import ndtri
    if ncols == 0:
        return np.zeros((n, 0))
    perm = np.argsort(rng.random((n, ncols)), axis=0)
    u = (perm + rng.random((n, ncols))) / n
    return ndtri(u)


def lognormal(z: np.ndarray, cv) -> np.ndarray:
    """Mean-one lognormal factor(s) from standard normal deviates z and CV(s); CV 0 -> 1."""
    cv = np.asarray(cv, float)
    sig = np.sqrt(np.log1p(cv * cv))
    return np.clip(np.exp(-0.5 * sig * sig + sig * z), _FMIN, _FMAX)


# --------------------------------------------------------------------------------------------- #
# Year inputs
# --------------------------------------------------------------------------------------------- #
@dataclass
class YearInputs:
    year: int
    iso: np.ndarray                       # (nC,) country universe = assemble() rows
    reg: np.ndarray                       # (nC,) region index
    c0: Dict[str, np.ndarray]             # central terms (nC,)
    cf: np.ndarray                        # effective cropland fraction
    g_fert: np.ndarray                    # effective FracGASF
    r_mv_crop: np.ndarray                 # man_vol / man_crop
    r_mv_past: np.ndarray                 # man_vol_past / man_past
    bnf_leg: np.ndarray
    bnf_free: np.ndarray
    hum: Optional[dict] = None            # pre-redistribution human-budget arrays (nH,)
    H: Optional[np.ndarray] = None        # (nC, nH) dissolution redistribution matrix
    lattice: Optional[dict] = None        # forage lattice (aligned to iso) or None
    forage_legacy: Optional[np.ndarray] = None
    forage_grid: Optional[np.ndarray] = None
    notes: List[str] = field(default_factory=list)


def _aligned(iso: np.ndarray, df: Optional[pd.DataFrame], col: Optional[str] = None) -> np.ndarray:
    """Align a [ISO3, value] table onto the iso universe, missing -> 0 (as assemble() does)."""
    if df is None or len(df) == 0:
        return np.zeros(len(iso))
    col = col or [c for c in df.columns if c != "ISO3"][0]
    s = pd.to_numeric(df.groupby("ISO3")[col].sum(), errors="coerce")
    return pd.Series(iso).map(s).fillna(0.0).to_numpy(float)


def _ratio(num, den):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, num / np.where(den > 0, den, 1.0), 0.0)


def load_year_inputs(cfg, year: int, grids_dir: Optional[Path] = None,
                     lattice_dir: Optional[Path] = None, use_grid_forage: bool = True,
                     verbose: bool = True) -> tuple:
    """Build the national components for `year` with the deterministic code, and the extra
    intermediates the coefficient-level ensemble needs. Returns (YearInputs, comps, areas)."""
    from . import config as cfgmod, country_areas, pipeline, bnf, human, dissolution
    raw = cfgmod.get_path(cfg, "raw")
    t0 = time.time()
    areas = country_areas.load_areas(cfg, raw, year)
    comps = pipeline.build_components(cfg, raw, year, areas, "N")
    leg_df, free_df = bnf.components(cfg, raw, year, "N")
    iso = areas["ISO3"].astype(str).to_numpy()
    A = lambda name, col=None: _aligned(iso, comps.get(name), col)   # noqa: E731

    c0 = {
        "fert": A("fert"), "fert_grass": A("fert_grass"), "fert_vol": A("fert_vol"),
        "fert_vol_full": A("fert_vol_full"), "crop": A("crop"), "bnf": A("bnf"),
        "bnf_grass": A("bnf_grass"), "dep": A("dep"), "dep_land": A("dep_land"),
        "man_crop": A("man_crop"), "man_past": A("man_past"), "man_vol": A("man_vol"),
        "forage": A("forage"), "food_waste": A("food_waste"), "hum_ns": A("hum_ns"),
        "hum_bio": A("hum_bio"), "hum_landfill": A("hum_landfill"),
        "hum_sew_land": A("hum_sew_land"), "hum_sew_water": A("hum_sew_water"),
    }
    c0["man_vol_past"] = A("man_vol_full") - c0["man_vol"]
    full = c0["fert"] + c0["fert_grass"]
    inp = YearInputs(
        year=int(year), iso=iso, reg=region_index(iso), c0=c0,
        cf=_ratio(c0["fert"], full), g_fert=_ratio(c0["fert_vol_full"], full),
        r_mv_crop=_ratio(c0["man_vol"], c0["man_crop"]),
        r_mv_past=_ratio(c0["man_vol_past"], c0["man_past"]),
        bnf_leg=_aligned(iso, leg_df), bnf_free=_aligned(iso, free_df))
    inp.forage_legacy = c0["forage"].copy()

    # ---- human-N partition (pre-redistribution) ----
    b = human._human_budget(cfg, raw, year)
    hcfg = cfg.get("human", {})
    if b is not None and hcfg.get("method") == "protein_treated":
        hiso = b["ISO3"].astype(str).to_numpy()
        routes = human._bio_routes(cfg, list(hiso), year)
        anchors = set(human._recycle_country_anchor(cfg, year).keys())
        inp.hum = dict(
            iso=hiso, reg=region_index(hiso),
            supply=(b["population"].astype(float) * b["e_supply"].astype(float)).to_numpy(),
            w_c=b["w_c"].astype(float).to_numpy(), c=b["c"].astype(float).to_numpy(),
            f_treated=b["f_treated"].astype(float).to_numpy(), f_rem=b["f_rem"].astype(float).to_numpy(),
            f_ns=np.asarray(b["f_recycle"], float) * np.ones(len(b)),
            f_sludge=b["f_sludge"].astype(float).to_numpy(),
            is_t1=np.isin(hiso, list(anchors)),
            a_land=np.array([routes[i][0] + routes[i][1] for i in hiso], float),
            a_lf=np.array([routes[i][2] for i in hiso], float),
            s_bio=float(hcfg.get("sludge_biomass_fraction", 0.18)),
            biosolids=bool(hcfg.get("biosolids", True)),
            R_low=float(hcfg.get("treatment_R_low", 0.10)),
            R_high=float(hcfg.get("treatment_R_high", 0.80)))
        # redistribution matrix (dissolution aggregates -> successors), restricted to the universe
        pos = {c: k for k, c in enumerate(iso)}
        H = np.zeros((len(iso), len(hiso)))
        sh = None
        for h, c in enumerate(hiso):
            if c in dissolution.PSEUDOS:
                sh = sh or dissolution.shares(cfg, raw, "population")
                for succ, frac in sh[c].items():
                    if succ in pos:
                        H[pos[succ], h] += frac
            elif c in pos:
                H[pos[c], h] = 1.0
        inp.H = H
    else:
        inp.notes.append("human budget unavailable: human terms held at central values (activity only)")

    # ---- forage lattice (grid-scale Eq. 29) ----
    if use_grid_forage and grids_dir is not None:
        try:
            lat_ = forage_lattice(cfg, year, Path(grids_dir), lattice_dir, verbose=verbose)
        except FileNotFoundError as e:
            lat_ = None
            inp.notes.append(f"no forage lattice ({e}); forage uses the term-level fallback CV")
        if lat_ is not None:
            pos = pd.Series(np.arange(len(lat_["iso"])), index=lat_["iso"])
            k = pd.Series(iso).map(pos)
            have = k.notna().to_numpy()
            kk = k.fillna(0).astype(int).to_numpy()
            F = lat_["F"][kk] * have[:, None, None]
            inp.lattice = dict(F=F, have=have, logs=lat_["logs"], M=lat_["M"][kk] * have,
                               P=lat_["P"][kk] * have, center=lat_["center"])
            inp.forage_grid = lat_["F_nc"][kk] * have
            c0["forage"] = F[:, lat_["center"], lat_["center"]].copy()
    # ---- align with the reported national budget ----
    # The released mass_balance_N_<year>.csv files carry grid-reconciled values for some terms
    # (deposition over all land replaced by its grid sum). Deposition enters the ensemble only
    # through a multiplicative regional factor, so the reported values can be used as the central
    # values directly; every other term is reported in the log so any further mismatch is visible.
    try:
        rep = pd.read_csv(cfgmod.get_path(cfg, "outputs") / f"mass_balance_N_{year}.csv")
        rep = rep.set_index("ISO3").reindex(iso)
        for k in ("dep", "dep_land"):
            col = f"{k}_kg"
            if col in rep.columns:
                v = rep[col].fillna(0.0).to_numpy(float)
                if abs(v.sum() - c0[k].sum()) > 1e-6 * max(abs(v.sum()), 1.0):
                    inp.notes.append(f"{k}: using reported national values ({v.sum()/1e9:.3f} Tg; "
                                     f"computed {c0[k].sum()/1e9:.3f} Tg)")
                    c0[k] = v
        diffs = []
        for k in TERMS:
            col = f"{k}_kg"
            if col in rep.columns and k not in ("dep", "dep_land"):
                a, b = c0[k].sum(), rep[col].fillna(0.0).sum()
                if abs(a - b) > 1e-3 * max(abs(b), 1e6):
                    diffs.append(f"{k} {a/1e9:.3f} vs reported {b/1e9:.3f} Tg")
        if diffs:
            inp.notes.append("central terms differing from mass_balance CSV: " + "; ".join(diffs))
    except FileNotFoundError:
        inp.notes.append(f"mass_balance_N_{year}.csv not found; central values from the model code")
    if verbose:
        print(f"  [{year}] inputs ready in {time.time()-t0:.1f}s ({len(iso)} countries"
              f"{'' if inp.lattice is None else ', forage lattice'})")
    return inp, comps, areas


# --------------------------------------------------------------------------------------------- #
# Forage lattice
# --------------------------------------------------------------------------------------------- #
def forage_lattice(cfg, year: int, grids_dir: Path, cache_dir: Optional[Path] = None,
                   verbose: bool = True) -> dict:
    """National forage totals on a (p_managed, p_range) multiplier lattice for `year`.

    Uses the year's saved grid (pasture-deposited manure, forage), the Global Pasture Watch bands
    and the national grazing-product N. Grazing-product N is placed on the same proxy as pasture
    manure, as in downscale._forage_gridscale, so the cell demand is
        D(m) = MAN_graze(m) . (1 + P_i / M_i).
    Cached to <cache_dir>/forage_lattice_<year>.npz."""
    nc = Path(grids_dir) / f"surplus_N_{year}_0p05.nc"
    if not nc.exists():
        raise FileNotFoundError(nc.name)
    nc_mtime = nc.stat().st_mtime
    if cache_dir is not None:
        f = Path(cache_dir) / f"forage_lattice_{year}.npz"
        if f.exists():
            z = np.load(f, allow_pickle=True)
            if "nc_mtime" in z.files and float(z["nc_mtime"]) == nc_mtime:   # grid unchanged since
                return {k: z[k] for k in z.files} | {"center": int(z["center"])}
    import xarray as xr
    from . import config as cfgmod, downscale as ds, forage as foragemod
    t0 = time.time()
    raw = cfgmod.get_path(cfg, "raw")
    g = xr.open_dataset(nc)
    lat = g["lat"].values; lon = g["lon"].values
    man_past = np.nan_to_num(g["man_past_kg"].values.astype("float64"))
    forage_nc = np.nan_to_num(g["forage_kg"].values.astype("float64"))
    mask, idx2iso, clat, clon = ds.load_country_grid(cfg)
    if mask.shape != man_past.shape:
        raise ValueError(f"country mask {mask.shape} != grid {man_past.shape}")
    if (clat[0] - clat[-1]) * (lat[0] - lat[-1]) < 0:          # opposite latitude order
        mask = mask[::-1, :]
    shape = man_past.shape
    asc = lat[0] < lat[-1]
    p_ha, g_ha = ds.load_gpw_grazing(cfg, shape, lat if asc else lat[::-1], lon, year)
    if p_ha is None:
        raise FileNotFoundError(f"GPW grazing bands for {year}")
    if not asc:                                                # load_gpw_grazing returns lat-ascending
        p_ha, g_ha = p_ha[::-1, :], g_ha[::-1, :]
    r_ha = np.clip(g_ha - p_ha, 0.0, None)
    p_ha = np.clip(p_ha, 0.0, None)
    pm0, pr0 = ds._p_graze_rates(cfg)

    flat_mask = mask.ravel().astype(int)
    nidx = int(flat_mask.max()) + 1
    M_idx = np.bincount(flat_mask.clip(min=0), weights=man_past.ravel(), minlength=nidx)
    M_idx[0] = 0.0
    prod = foragemod.product_n(cfg, raw, year)
    iso2P = dict(zip(prod["ISO3"], prod["prod_n_kg"])) if prod is not None and len(prod) else {}
    P_idx = np.zeros(nidx)
    for i, c in idx2iso.items():
        if 0 < i < nidx:
            P_idx[i] = float(iso2P.get(c, 0.0))
    scale = np.where(M_idx > 0, 1.0 + P_idx / np.where(M_idx > 0, M_idx, 1.0), 1.0)
    D = man_past.ravel() * scale[flat_mask.clip(min=0)]
    sel = (D > 0) & (flat_mask > 0)
    D = D[sel]; a = p_ha.ravel()[sel] * pm0; b = r_ha.ravel()[sel] * pr0
    cid = flat_mask[sel]
    logs = np.linspace(-LAT_HALF_WIDTH, LAT_HALF_WIDTH, LAT_N)
    mult = np.exp(logs)
    F_idx = np.zeros((nidx, LAT_N, LAT_N))
    for iu, u in enumerate(mult):
        au = u * a
        for iv, v in enumerate(mult):
            F_idx[:, iu, iv] = np.bincount(cid, weights=np.minimum(D, au + v * b), minlength=nidx)
    Fnc_idx = np.bincount(flat_mask.clip(min=0), weights=forage_nc.ravel(), minlength=nidx)
    Fnc_idx[0] = 0.0
    keep = [i for i in sorted(idx2iso) if 0 < i < nidx]
    # several mask indices can map to one ISO3: sum them
    df = pd.DataFrame({"iso": [idx2iso[i] for i in keep], "i": keep})
    isos = df["iso"].unique()
    pos = {c: k for k, c in enumerate(isos)}
    F = np.zeros((len(isos), LAT_N, LAT_N)); Fnc = np.zeros(len(isos))
    M = np.zeros(len(isos)); P = np.zeros(len(isos))
    for i, c in zip(df["i"], df["iso"]):
        k = pos[c]
        F[k] += F_idx[i]; Fnc[k] += Fnc_idx[i]; M[k] += M_idx[i]
    for c in isos:
        P[pos[c]] = float(iso2P.get(c, 0.0)) if M[pos[c]] > 0 else 0.0
    center = LAT_N // 2
    out = dict(iso=np.array(isos), F=F, F_nc=Fnc, M=M, P=P, logs=logs, center=center,
               pm0=pm0, pr0=pr0, nc_mtime=nc_mtime)
    tot_c, tot_nc = F[:, center, center].sum() / 1e9, Fnc.sum() / 1e9
    if verbose:
        print(f"  [{year}] forage lattice {LAT_N}x{LAT_N} over {int(sel.sum()):,} cells in "
              f"{time.time()-t0:.0f}s; central {tot_c:.3f} Tg vs saved grid {tot_nc:.3f} Tg")
    if cache_dir is not None:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        np.savez_compressed(Path(cache_dir) / f"forage_lattice_{year}.npz", **out)
    return out


def _interp_lattice(F, logs, cidx, lu, lv):
    """Bilinear interpolation of F[c, iu, iv] at log-multipliers (lu, lv); shapes (n, nC)."""
    n = len(logs); step = logs[1] - logs[0]
    tu = np.clip((lu - logs[0]) / step, 0, n - 1)
    tv = np.clip((lv - logs[0]) / step, 0, n - 1)
    iu = np.minimum(tu.astype(int), n - 2); iv = np.minimum(tv.astype(int), n - 2)
    wu = tu - iu; wv = tv - iv
    c = np.broadcast_to(cidx, lu.shape)
    return ((1 - wu) * (1 - wv) * F[c, iu, iv] + wu * (1 - wv) * F[c, iu + 1, iv]
            + (1 - wu) * wv * F[c, iu, iv + 1] + wu * wv * F[c, iu + 1, iv + 1])


# --------------------------------------------------------------------------------------------- #
# Terms from factors
# --------------------------------------------------------------------------------------------- #
def _human_terms(h: dict, f: dict, n: int) -> Dict[str, np.ndarray]:
    """Recompute the human-N partition (Eqs. 34-57) with drawn coefficients. Arrays (n, nH)."""
    one = np.ones((n, 1))
    g = lambda k: f.get(k, one)                                      # noqa: E731
    w_c = np.minimum(h["w_c"] * g("w_c"), 1.0)
    supply = h["supply"] * g("protein_N")
    E = supply * (1.0 - w_c)
    food_waste = supply * w_c
    c0 = h["c"]
    c = np.minimum(c0 * g("c"), 1.0)
    # treated fraction of excreted N: c . tau, tau = f_treated / c (f_treat in the text)
    tau = _ratio(h["f_treated"], c0)
    f_tr = np.where(c0 > 0, c * np.minimum(tau * g("f_treat"), 1.0),
                    np.minimum(h["f_treated"] * g("f_treat"), 1.0))
    # removal efficiency R_N = f_rem / f_treated, perturbed through its two anchors (Eq. 47)
    R0 = _ratio(h["f_rem"], h["f_treated"])
    Rl, Rh = h["R_low"], h["R_high"]
    s = np.clip((R0 - Rl) / (Rh - Rl), 0.0, 1.0)
    dRl = np.minimum(Rl * g("R_low"), 1.0) - Rl
    dRh = np.minimum(Rh * g("R_high"), 1.0) - Rh
    RN = np.clip(R0 + dRl * (1.0 - s) + dRh * s, 0.0, 1.0)
    f_rem = np.where(h["f_treated"] > 0, f_tr * RN, h["f_rem"] * np.ones_like(f_tr))
    # night soil: Tier 1 = documented share of total excreta; Tier 2 = lambda_ns x (1 - c)
    lam = _ratio(h["f_ns"], 1.0 - c0)
    f_ns = np.where(h["is_t1"], h["f_ns"] * g("lambda_ns"),
                    np.minimum(lam * g("lambda_ns"), 1.0) * (1.0 - c))
    f_ns = np.clip(np.minimum(f_ns, 1.0 - c), 0.0, None)
    # sewage sludge: min(sl_bio . treated, removed); zero for Tier-1 recyclers or if biosolids off
    if h["biosolids"]:
        f_sl = np.where(h["is_t1"], 0.0, np.minimum(h["s_bio"] * g("sl_bio") * f_tr, f_rem))
    else:
        f_sl = np.zeros_like(f_tr)
    # sludge routes, renormalized to sum to one
    a_ag = h["a_land"] * g("a_ag"); a_lf = h["a_lf"] * g("a_lf")
    a_inc = np.clip(1.0 - h["a_land"] - h["a_lf"], 0.0, None)
    tot = a_ag + a_lf + a_inc
    a_ag = _ratio(a_ag, tot); a_lf = _ratio(a_lf, tot)
    sludge = E * f_sl
    sewage = E * np.clip(1.0 - f_ns - f_rem, 0.0, None)
    sew_land = np.minimum(E * np.clip(1.0 - c - f_ns, 0.0, None), sewage)
    return dict(food_waste=food_waste, hum_ns=E * f_ns, hum_bio=a_ag * sludge,
                hum_landfill=a_lf * sludge, hum_sew_land=sew_land,
                hum_sew_water=np.clip(sewage - sew_land, 0.0, None))


def compute_terms(inp: YearInputs, f: dict, n: int) -> Dict[str, np.ndarray]:
    """All national terms (n, nC) from coefficient/activity factors `f` (missing keys -> 1)."""
    one = np.ones((n, 1))
    g = lambda k: f.get(k, one)                                      # noqa: E731
    c0 = inp.c0
    full = (c0["fert"] + c0["fert_grass"]) * g("act_fert")
    cf = np.where(inp.cf < 1.0, np.minimum(inp.cf * g("cf"), 1.0), inp.cf)
    t = {}
    t["fert"] = full * cf
    t["fert_grass"] = full - t["fert"]
    gf = np.minimum(inp.g_fert * g("fracgasf"), 1.0)
    t["fert_vol"] = t["fert"] * gf
    t["fert_vol_full"] = full * gf
    t["crop"] = c0["crop"] * g("cN") * g("act_crop")
    t["bnf"] = (inp.bnf_leg * g("cN") * g("f_bnf") * g("wpf") + inp.bnf_free * g("r_free")) * g("act_bnf")
    t["bnf_grass"] = c0["bnf_grass"] * g("r_graze") * g("act_bnf")
    t["dep"] = c0["dep"] * g("dep")
    t["dep_land"] = c0["dep_land"] * g("dep")
    t["man_crop"] = c0["man_crop"] * g("nex") * g("act_man")
    t["man_past"] = c0["man_past"] * g("nex") * g("act_man")
    t["man_vol"] = t["man_crop"] * np.minimum(inp.r_mv_crop * g("fracgasm_crp"), 1.0)
    t["man_vol_past"] = t["man_past"] * np.minimum(inp.r_mv_past * g("fracgasm_graze"), 1.0)
    # forage
    if inp.lattice is not None:
        L = inp.lattice
        M, P = L["M"], L["P"]
        with np.errstate(divide="ignore", invalid="ignore"):
            d = np.where(M + P > 0, (g("nex") * M + g("prodN") * P) / np.where(M + P > 0, M + P, 1.0), 1.0)
        d = np.broadcast_to(d, (n, len(inp.iso)))
        lu = np.log(np.broadcast_to(g("p_managed"), d.shape) / d)
        lv = np.log(np.broadcast_to(g("p_range"), d.shape) / d)
        Fi = _interp_lattice(L["F"], L["logs"], np.arange(len(inp.iso))[None, :], lu, lv)
        # countries absent from the grid keep the national (legacy) value with the fallback factor
        fallback = inp.forage_legacy * g("forage_fallback")
        t["forage"] = np.where(L["have"][None, :], d * Fi, fallback) * g("act_forage")
    else:
        t["forage"] = c0["forage"] * g("forage_fallback") * g("act_forage")
    # human
    if inp.hum is not None:
        ht = _human_terms(inp.hum, f.get("_human", {}), n)
        for k, v in ht.items():
            t[k] = (v @ inp.H.T) * g("act_pop")
    else:
        for k in ("food_waste", "hum_ns", "hum_bio", "hum_landfill", "hum_sew_land", "hum_sew_water"):
            t[k] = c0[k] * g("act_pop")
    return {k: np.broadcast_to(t[k], (n, len(inp.iso))) for k in TERMS}


def surpluses(t: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {v: sum(s * t[k] for k, s in VIEWS[v].items()) for v in VIEWS}


# --------------------------------------------------------------------------------------------- #
# Factor generation
# --------------------------------------------------------------------------------------------- #
GLOBAL_KEYS = ["cf", "fracgasf", "cN", "f_bnf", "wpf", "r_free", "r_graze", "fracgasm_crp",
               "fracgasm_graze", "p_managed", "p_range", "prodN", "protein_N", "R_low", "R_high",
               "sl_bio", "forage_fallback"]
REGIONAL_KEYS = ["nex", "dep", "w_c", "c", "f_treat", "a_ag", "a_lf", "lambda_ns"]
HUMAN_KEYS = {"protein_N", "R_low", "R_high", "sl_bio", "w_c", "c", "f_treat", "a_ag", "a_lf",
              "lambda_ns"}


class FactorDraws:
    """Latin-hypercube normal deviates for every random quantity of one year, generated once and
    sliced into chunks, so the total, coefficient-only and activity-only ensembles share draws."""

    def __init__(self, inp: YearInputs, spec, rcv, n: int, seed: int, activity_cv: Dict[str, np.ndarray]):
        rng = np.random.default_rng(seed)
        self.inp, self.spec, self.rcv, self.n = inp, spec, rcv, n
        self.zg = {k: lhs_normals(n, 1, rng) for k in GLOBAL_KEYS}
        self.zr = {k: lhs_normals(n, len(REGIONS), rng) for k in REGIONAL_KEYS}
        nC = len(inp.iso)
        self.za = {gname: lhs_normals(n, nC, rng) for gname in ACT_GROUPS}
        self.act_cv = activity_cv

    def _regional_cv(self, key):
        c = self.spec[key]
        col = c.cv_class
        return np.array([float(self.rcv.loc[r, col]) if col in self.rcv.columns else c.cv
                         for r in REGIONS])

    def factors(self, sl: slice, coef: bool = True, act: bool = True) -> dict:
        n = sl.stop - sl.start
        f, fh = {}, {}
        if coef:
            for k in GLOBAL_KEYS:
                cv = self.spec[k].cv if k in self.spec and np.isfinite(self.spec[k].cv) else 0.0
                v = lognormal(self.zg[k][sl], cv)                         # (n, 1)
                (fh if k in HUMAN_KEYS else f)[k] = v
            for k in REGIONAL_KEYS:
                if k not in self.spec:
                    continue
                vr = lognormal(self.zr[k][sl], self._regional_cv(k)[None, :])   # (n, nR)
                if k in HUMAN_KEYS:
                    if self.inp.hum is not None:
                        fh[k] = vr[:, self.inp.hum["reg"]]
                else:
                    f[k] = vr[:, self.inp.reg]
        if act:
            for gname in ACT_GROUPS:
                cv = self.act_cv.get(gname)
                if cv is None:
                    continue
                f[f"act_{gname}"] = lognormal(self.za[gname][sl], cv[None, :])
        f["_human"] = fh
        return f


# --------------------------------------------------------------------------------------------- #
# Ensemble
# --------------------------------------------------------------------------------------------- #
@dataclass
class EnsembleResult:
    year: int
    iso: np.ndarray
    n: int
    central: Dict[str, np.ndarray]                   # term -> (nC,) central
    surplus_central: Dict[str, np.ndarray]           # view -> (nC,)
    surplus: Dict[str, np.ndarray]                   # view -> (n, nC) float32 (total ensemble)
    term_global: Dict[str, np.ndarray]               # term -> (n,) global Tg (total ensemble)
    mean: Dict[str, np.ndarray]                      # mode -> (nC, K) term means
    cov: Dict[str, np.ndarray]                       # mode -> (nC, K, K) covariances
    notes: List[str]


def run_ensemble(inp: YearInputs, spec, rcv, n: int = 10000, seed: int = 0, chunk: int = 1000,
                 activity_cv: Optional[Dict[str, np.ndarray]] = None, verbose: bool = True) -> EnsembleResult:
    from . import uncertainty as U
    t0 = time.time()
    nC, K = len(inp.iso), len(TERMS)
    if activity_cv is None:
        grp_map = {"fert": "fert", "crop": "crop", "bnf": "bnf", "man": "man", "forage": "forage", "pop": "pop"}
        activity_cv = {g: U.activity_cv_vector(grp_map[g], inp.iso, inp.year) for g in ACT_GROUPS}
    draws = FactorDraws(inp, spec, rcv, n, seed, activity_cv)
    cen = compute_terms(inp, {"_human": {}}, 1)
    central = {k: np.array(cen[k][0]) for k in TERMS}
    s_cen = {v: x[0] for v, x in surpluses(cen).items()}
    modes = {"total": (True, True), "coef": (True, False), "act": (False, True)}
    S1 = {m: np.zeros((nC, K)) for m in modes}
    S2 = {m: np.zeros((nC, K, K)) for m in modes}
    surplus = {v: np.empty((n, nC), "float32") for v in VIEWS}
    term_global = {k: np.empty(n) for k in TERMS}
    for a in range(0, n, chunk):
        sl = slice(a, min(n, a + chunk))
        m_ = sl.stop - sl.start
        for mode, (cf_, af_) in modes.items():
            t = compute_terms(inp, draws.factors(sl, coef=cf_, act=af_), m_)
            X = np.stack([t[k] for k in TERMS], axis=-1)              # (m, nC, K)
            S1[mode] += X.sum(axis=0)
            S2[mode] += np.einsum("nck,ncl->ckl", X, X, optimize=True)
            if mode == "total":
                for v, arr in surpluses(t).items():
                    surplus[v][sl] = arr
                for k in TERMS:
                    term_global[k][sl] = t[k].sum(axis=1) / 1e9
    mean = {m: S1[m] / n for m in modes}
    cov = {m: S2[m] / n - np.einsum("ck,cl->ckl", mean[m], mean[m]) for m in modes}
    if verbose:
        g = {v: surplus[v].sum(axis=1) / 1e9 for v in VIEWS}
        msg = "; ".join(f"{v} {s_cen[v].sum()/1e9:.1f} Tg (MC {g[v].mean():.1f}, 95% "
                        f"{np.percentile(g[v], 2.5):.1f}-{np.percentile(g[v], 97.5):.1f})" for v in VIEWS)
        print(f"  [{inp.year}] {n} draws in {time.time()-t0:.1f}s: {msg}")
    return EnsembleResult(year=inp.year, iso=inp.iso, n=n, central=central, surplus_central=s_cen,
                          surplus=surplus, term_global=term_global, mean=mean, cov=cov, notes=inp.notes)


# --------------------------------------------------------------------------------------------- #
# Summaries and outputs
# --------------------------------------------------------------------------------------------- #
def country_summary(res: EnsembleResult) -> pd.DataFrame:
    out = {"ISO3": res.iso}
    for v, arr in res.surplus.items():
        a = arr.astype("float64")
        mean = a.mean(axis=0); sd = a.std(axis=0)
        out[f"surplus_{v}_central_TgN"] = res.surplus_central[v] / 1e9
        out[f"surplus_{v}_mean_TgN"] = mean / 1e9
        out[f"surplus_{v}_sd_TgN"] = sd / 1e9
        out[f"surplus_{v}_p2.5_TgN"] = np.percentile(a, 2.5, axis=0) / 1e9
        out[f"surplus_{v}_p97.5_TgN"] = np.percentile(a, 97.5, axis=0) / 1e9
        with np.errstate(divide="ignore", invalid="ignore"):
            out[f"surplus_{v}_cv"] = np.where(mean != 0, sd / np.abs(mean), np.nan)
    return pd.DataFrame(out)


def global_summary(res: EnsembleResult) -> pd.DataFrame:
    rows = []
    for v, arr in res.surplus.items():
        g = arr.astype("float64").sum(axis=1) / 1e9
        rows.append(dict(year=res.year, scope=v, central_Tg=res.surplus_central[v].sum() / 1e9,
                         mc_mean_Tg=g.mean(), sd_Tg=g.std(), **{"p2.5_Tg": np.percentile(g, 2.5),
                         "p25_Tg": np.percentile(g, 25), "p50_Tg": np.percentile(g, 50),
                         "p75_Tg": np.percentile(g, 75), "p97.5_Tg": np.percentile(g, 97.5)},
                         rel_uncertainty_pct=100 * g.std() / abs(g.mean()) if g.mean() else np.nan))
    return pd.DataFrame(rows)


def term_summary(res: EnsembleResult) -> pd.DataFrame:
    rows = []
    for k in TERMS:
        g = res.term_global[k]
        rows.append(dict(year=res.year, term=k, central_Tg=res.central[k].sum() / 1e9, mc_mean_Tg=g.mean(),
                         sd_Tg=g.std(), **{"p2.5_Tg": np.percentile(g, 2.5), "p25_Tg": np.percentile(g, 25),
                                            "p50_Tg": np.percentile(g, 50), "p75_Tg": np.percentile(g, 75),
                                            "p97.5_Tg": np.percentile(g, 97.5)},
                         cv=g.std() / g.mean() if g.mean() else np.nan))
    return pd.DataFrame(rows)


def save_global_draws(res: EnsembleResult, path: Path):
    """Global totals (Tg N) for every realization: the three surplus views and the 20 budget terms.
    Used for figures that need percentiles of sums of terms (e.g. all manure, all fixation)."""
    out = {f"surplus_{v}": (arr.astype("float64").sum(axis=1) / 1e9).astype("float32")
           for v, arr in res.surplus.items()}
    out.update({f"term_{k}": res.term_global[k].astype("float32") for k in TERMS})
    out.update({f"central_{k}": np.float64(res.central[k].sum() / 1e9) for k in TERMS})
    out.update({f"central_surplus_{v}": np.float64(res.surplus_central[v].sum() / 1e9) for v in res.surplus})
    np.savez_compressed(path, **out)


def save_covariance(res: EnsembleResult, path: Path):
    """Per-country relative covariance C_kl = cov_kl / (mean_k mean_l) for each mode (total / coef /
    act), plus the central terms, for the gridded layer (revised Eq. 67)."""
    out = dict(iso=res.iso, terms=np.array(TERMS), central=np.stack([res.central[k] for k in TERMS], -1))
    for m, cov in res.cov.items():
        mu = res.mean[m]
        den = mu[:, :, None] * mu[:, None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            out[f"C_{m}"] = np.where(den != 0, cov / np.where(den != 0, den, 1.0), 0.0).astype("float32")
        out[f"mean_{m}"] = mu
    np.savez_compressed(path, **out)


# --------------------------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------------------------- #
def validate_central(inp: YearInputs, comps: dict, areas: pd.DataFrame, cfg=None) -> pd.DataFrame:
    """With every factor equal to one, compare each term with the deterministic component and each
    surplus with mass_balance.assemble(). Forage is compared on the national (legacy) basis used by
    assemble(); the grid-scale forage is reported separately."""
    from . import mass_balance
    lat_saved = inp.lattice
    inp.lattice = None                              # compare against the deterministic national forage
    c_save = inp.c0["forage"]
    inp.c0["forage"] = inp.forage_legacy
    t = compute_terms(inp, {"_human": {}}, 1)
    inp.lattice = lat_saved; inp.c0["forage"] = c_save
    rows = []
    exp = {k: _aligned(inp.iso, comps.get(k)) for k in TERMS if k in comps}
    exp["man_vol_past"] = _aligned(inp.iso, comps.get("man_vol_full")) - _aligned(inp.iso, comps.get("man_vol"))
    for k in TERMS:
        if k not in exp:
            continue
        a, b = t[k][0], exp[k]
        rows.append(dict(item=k, model_Tg=b.sum() / 1e9, mc_central_Tg=a.sum() / 1e9,
                         max_abs_diff_kg=float(np.max(np.abs(a - b))) if len(a) else 0.0))
    tab = mass_balance.assemble(comps, areas, "N").set_index("ISO3").reindex(inp.iso)
    s = surpluses(t)
    ref = {"cropland": tab["surplus_cropland_kgha"] * tab["AREA_HA"],
           "landscape": tab["surplus_landscape_kgha"] * tab["LAND_HA"],
           "total": tab["surplus_full_kgha"] * tab["LAND_HA"]}
    for v in VIEWS:
        r = np.nan_to_num(ref[v].to_numpy(float)); a = s[v][0]
        ok = np.isfinite(ref[v].to_numpy(float))
        rows.append(dict(item=f"surplus_{v}", model_Tg=r[ok].sum() / 1e9, mc_central_Tg=a[ok].sum() / 1e9,
                         max_abs_diff_kg=float(np.max(np.abs(a[ok] - r[ok]))) if ok.any() else 0.0))
    if inp.forage_grid is not None:
        rows.append(dict(item="forage (grid-scale, saved grid)", model_Tg=inp.forage_grid.sum() / 1e9,
                         mc_central_Tg=inp.c0["forage"].sum() / 1e9,
                         max_abs_diff_kg=float(np.max(np.abs(inp.c0["forage"] - inp.forage_grid)))))
    return pd.DataFrame(rows)
