"""SCALES-Global — Monte-Carlo uncertainty layer (Strategy A, IPCC-Nex regional correlation).

Philosophy
----------
  * The deterministic model keeps its validated central estimates. The *literature* defines only the
    RELATIVE SPREAD around each term (Strategy A). Every perturbation is a mean-1 multiplicative factor,
    so the ensemble mean sits on the deterministic value (mass-consistent).

  * Each N-budget term carries TWO error channels:
      - a COEFFICIENT / METHOD channel, whose spatial CORRELATION is set by how the coefficient is
        sourced:
          * GLOBAL-SHARED terms (one global value applied everywhere — crop N content, %Ndfa,
            IPCC FracGAS, forage) draw ONE factor per realization for the whole world (fully
            correlated; does not average out on aggregation).
          * REGION-RESOLVED terms (values defined by IPCC 2019 world region — livestock excretion Nex,
            atmospheric deposition, and the human-waste/sanitation terms) draw ONE factor per IPCC-Nex
            REGION per realization, shared by all countries in that region and independent across
            regions, so their error partially averages when the ~9 regions are summed.
      - an ACTIVITY / REPORTING channel (independent across countries): one factor per activity GROUP
        per country per draw (FAOSTAT national-statistics error). A flux and its NH3 loss share one
        activity factor per country; all population-derived human terms share one population factor.

    This replaces the earlier scalar systematic-fraction (rho) split with an explicit block-correlation
    keyed to the IPCC 2019 Nex regions (Vol.4 Ch.10 Table 10.19). See
    the regional uncertainty classes.

  * Distributions are mean-1 lognormals; CVs -> sigma = sqrt(ln(1+CV^2)); factors tail-bounded [0.2, 5].

Lightweight: perturb the already-computed per-country component totals (the *_kg columns of
``data/outputs/mass_balance_N_<year>.csv``) and re-sum the three surpluses (cropland / landscape /
total), rather than re-running the FAOSTAT pipeline per draw.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REGION_CROSSWALK = ROOT / "coefficients/ipcc_nex_region_crosswalk.csv"

# --------------------------------------------------------------------------- #
# IPCC 2019 Nex world regions + per-region coefficient CVs (region-resolved terms)
# --------------------------------------------------------------------------- #
IPCC_REGIONS = ["North America", "Western Europe", "Eastern Europe", "Oceania", "Latin America",
                "Africa", "Middle East", "Asia", "Indian Subcontinent", "OTHER"]

# Per-region CV by cv-class, from Regional_uncertainty_approach_methods.md Sec.4 (IPCC Table 10.19
# excretion regions; deposition inter-product spread after Vishwakarma et al.; human-term recycling
# classes after Moree et al.). "OTHER" (uncovered countries) falls back to the term's base coeff_cv.
REGION_CV: Dict[str, Dict[str, float]] = {
    "North America":       {"manure": 0.20, "deposition": 0.15, "human_recycle": 0.30, "biosolids": 0.25, "food_waste": 0.25, "human_effluent": 0.25},
    "Western Europe":      {"manure": 0.20, "deposition": 0.15, "human_recycle": 0.30, "biosolids": 0.25, "food_waste": 0.25, "human_effluent": 0.25},
    "Eastern Europe":      {"manure": 0.25, "deposition": 0.20, "human_recycle": 0.40, "biosolids": 0.35, "food_waste": 0.30, "human_effluent": 0.30},
    "Oceania":             {"manure": 0.22, "deposition": 0.20, "human_recycle": 0.30, "biosolids": 0.30, "food_waste": 0.25, "human_effluent": 0.25},
    "Latin America":       {"manure": 0.30, "deposition": 0.25, "human_recycle": 0.45, "biosolids": 0.40, "food_waste": 0.30, "human_effluent": 0.30},
    "Africa":              {"manure": 0.40, "deposition": 0.30, "human_recycle": 0.50, "biosolids": 0.45, "food_waste": 0.35, "human_effluent": 0.35},
    "Middle East":         {"manure": 0.35, "deposition": 0.30, "human_recycle": 0.45, "biosolids": 0.40, "food_waste": 0.30, "human_effluent": 0.30},
    "Asia":                {"manure": 0.30, "deposition": 0.40, "human_recycle": 0.45, "biosolids": 0.40, "food_waste": 0.30, "human_effluent": 0.30},
    "Indian Subcontinent": {"manure": 0.35, "deposition": 0.40, "human_recycle": 0.45, "biosolids": 0.40, "food_waste": 0.35, "human_effluent": 0.35},
}

# Activity (reporting) CV by activity group. A flux and its NH3 loss share a group; all population-based
# human terms share the "pop" group; deposition (a modeled field) has NO activity channel.
# These are now the GLOBAL-MEAN activity CVs; the per-country/year values come from the FAOSTAT-flag
# lookup (coefficients/activity_cv_by_flag.csv, built by scripts/38_build_activity_cv.py), anchored so
# each group's flux-weighted global mean equals the value below. Used as the fallback where a country-year
# has no flag record (e.g. back-cast successor states pre-1992) or the lookup file is absent.
ACT_CV = {"fert": 0.12, "man": 0.10, "crop": 0.10, "bnf": 0.10, "forage": 0.10, "dep": 0.00, "pop": 0.05}

# --- per-country/year activity CV from FAOSTAT provenance flags -------------------------------------- #
_ACT_CV_LOOKUP: Optional[dict] = None


def _load_activity_cv_lookup() -> dict:
    """(group, iso3, year) -> activity CV, from the FAOSTAT-flag table (built by scripts/38)."""
    global _ACT_CV_LOOKUP
    if _ACT_CV_LOOKUP is None:
        p = ROOT / "coefficients/activity_cv_by_flag.csv"
        if p.exists():
            d = pd.read_csv(p)
            _ACT_CV_LOOKUP = {(str(g), str(i), int(y)): float(c)
                              for g, i, y, c in zip(d["group"], d["iso3"], d["year"], d["activity_cv"])}
        else:
            _ACT_CV_LOOKUP = {}
    return _ACT_CV_LOOKUP


def activity_cv_vector(group: str, iso3_array, year: Optional[int]) -> np.ndarray:
    """Per-country activity CV for an activity group in `year`, from the FAOSTAT-flag lookup.
    Falls back to the scalar ACT_CV[group] where a country-year has no flag record or the lookup
    is unavailable, so behaviour is unchanged when the table is absent."""
    base = ACT_CV.get(group, 0.0)
    lut = _load_activity_cv_lookup()
    if not lut or year is None:
        return np.full(len(iso3_array), base, float)
    y = int(year)
    return np.array([lut.get((group, str(c), y), base) for c in iso3_array], float)


@dataclass
class TermSpread:
    """Relative uncertainty for one N-budget term."""
    name: str                       # *_kg column in the mass-balance table
    coeff_cv: float                 # global-shared coefficient CV (also the OTHER-region fallback)
    act_group: str                  # activity group (shared activity factor)
    region_resolved: bool = False   # True -> one coeff factor per IPCC-Nex region; False -> one global
    cv_class: str = ""              # key into REGION_CV when region_resolved
    source: str = ""
    activity_cv: float = field(default=0.0)   # kept for back-compat (= ACT_CV[act_group])

    def __post_init__(self):
        if not self.activity_cv:
            self.activity_cv = ACT_CV.get(self.act_group, 0.0)


# --------------------------------------------------------------------------- #
# Parameter table — all terms in the current three-scope budget.
# --------------------------------------------------------------------------- #
PARAM_TABLE: List[TermSpread] = [
    # ---- GLOBAL-SHARED coefficients (one factor for the whole world) ----
    TermSpread("fert_kg", 0.05, "fert", source="Synthetic fertilizer N (FAOSTAT RFN). Best-constrained input; small global coefficient term (cropland-fraction split). Activity ~+-12%."),
    TermSpread("fert_grass_kg", 0.05, "fert", source="Synthetic fertilizer N applied to grassland (same source/coefficient as fert; shares fertilizer activity)."),
    TermSpread("crop_kg", 0.18, "crop", source="Crop N-removal coefficients (Ludemann et al. 2024, the FAO-CNB set). Per-crop spread CV~15-25%; production-weighted 0.18, one global value. Activity=FAOSTAT QCL."),
    TermSpread("bnf_kg", 0.40, "bnf", source="Biological N fixation on cropland. %Ndfa + whole-plant/below-ground + free-living; agricultural BNF ~+-50% (Herridge 2008/2022; Peoples 2021). One global coefficient."),
    TermSpread("bnf_grass_kg", 0.40, "bnf", source="Biological N fixation on grassland (same coefficient basis as bnf; shares legume activity)."),
    TermSpread("fert_vol_kg", 0.40, "fert", source="Cropland fertilizer NH3 volatilization = fert N x FracGASF (IPCC 2019 Table 11.3; range ~0.016-0.225, CV~0.40). One global coefficient. LOSS."),
    TermSpread("fert_vol_full_kg", 0.40, "fert", source="Whole-country fertilizer NH3 volatilization (cropland+grassland) = FracGASF; landscape/total loss. Same coefficient as fert_vol."),
    TermSpread("man_vol_kg", 0.35, "man", source="Cropland manure NH3 volatilization = applied manure N x FracGASM (IPCC 2019 default 0.21, range 0-0.31, CV~0.35). One global coefficient. LOSS."),
    TermSpread("man_vol_full_kg", 0.35, "man", source="Whole-country manure NH3 volatilization; landscape/total loss. Same coefficient as man_vol."),
    TermSpread("forage_kg", 0.30, "forage", source="Grassland forage N removal by grazing (lesser of grazing demand and pasture supply). Coeff CV~0.30, one global value. OUTPUT (landscape/total)."),
    # ---- REGION-RESOLVED coefficients (one factor per IPCC-Nex region) ----
    TermSpread("dep_kg", 0.25, "dep", region_resolved=True, cv_class="deposition", source="Cropland N deposition (NOy+NHx), modeled field. Region-resolved inter-product spread (Vishwakarma et al.): small Europe/Americas, large Asia. No activity channel."),
    TermSpread("dep_land_kg", 0.25, "dep", region_resolved=True, cv_class="deposition", source="Whole-land N deposition (landscape/total). Region-resolved as for dep. No activity channel."),
    TermSpread("man_crop_kg", 0.30, "man", region_resolved=True, cv_class="manure", source="Manure N applied to cropland (FAOSTAT EMN, IPCC-2019 Tier-1 Nex). Nex is tabulated BY REGION (IPCC Table 10.19); region-resolved. Activity=animal stocks."),
    TermSpread("man_past_kg", 0.30, "man", region_resolved=True, cv_class="manure", source="Manure N on pasture (left/deposited). Region-resolved Nex as for man_crop; shares animal-stock activity."),
    TermSpread("hum_ns_kg", 0.45, "pop", region_resolved=True, cv_class="human_recycle", source="Night soil: human excreta N recycled directly to cropland. Reconstructed by region/country (Tier-1 trajectories + Moree 1900 classes); region-resolved, widest human term. Activity=population."),
    TermSpread("hum_bio_kg", 0.35, "pop", region_resolved=True, cv_class="biosolids", source="Biosolids N applied to cropland. Regional sludge-management/recovery statistics; region-resolved. Activity=population."),
    TermSpread("hum_landfill_kg", 0.35, "pop", region_resolved=True, cv_class="biosolids", source="Landfilled sewage-sludge N retained in the landscape (a_lf route). Region-resolved as for hum_bio."),
    TermSpread("hum_sew_land_kg", 0.30, "pop", region_resolved=True, cv_class="human_effluent", source="Non-sewered human excreta N deposited on land. Regional sanitation reconstruction; region-resolved. Activity=population."),
    TermSpread("hum_sew_water_kg", 0.30, "pop", region_resolved=True, cv_class="human_effluent", source="Sewered human excreta N discharged to surface water (TOTAL surplus only). Regional sanitation/treatment reconstruction; region-resolved. Activity=population."),
    TermSpread("food_waste_kg", 0.30, "pop", region_resolved=True, cv_class="food_waste", source="Consumer food-waste N to the landscape. Regional retail+household loss fractions (Gustavsson/Moree); region-resolved. Activity=population."),
    # Retained for back-compat (parent of hum_sew_land/water; referenced by scripts/37). NOT in any
    # national surplus scope (the total uses hum_sew_water + hum_sew_land), so it never enters ALL_TERMS.
    TermSpread("hum_sewage_kg", 0.30, "pop", region_resolved=True, cv_class="human_effluent", source="Total human sewage discharge N (parent of hum_sew_land + hum_sew_water). Retained for the gridded script; not summed in any national scope."),
]

# DEPRECATED back-compat: a scalar systematic fraction, retained only so older importers
# (scripts/37) do not crash. The national MC no longer uses it — correlation is set by
# region_resolved (region-block) vs global-shared. Region-resolved -> ~0.6, global -> 1.0.
SYS_FRAC = {t.name: (0.6 if t.region_resolved else 1.0) for t in PARAM_TABLE}
ACT_GROUP = {t.name: t.act_group for t in PARAM_TABLE}


# --------------------------------------------------------------------------- #
# Surplus scope definitions — MUST mirror src/mass_balance.py exactly.
# --------------------------------------------------------------------------- #
CROPLAND_IN = ["fert_kg", "bnf_kg", "dep_kg", "man_crop_kg", "hum_ns_kg", "hum_bio_kg"]
CROPLAND_OUT = ["crop_kg", "fert_vol_kg", "man_vol_kg"]

LANDSCAPE_IN = ["fert_kg", "fert_grass_kg", "bnf_kg", "bnf_grass_kg", "dep_land_kg", "man_crop_kg",
                "man_past_kg", "food_waste_kg", "hum_ns_kg", "hum_bio_kg", "hum_sew_land_kg", "hum_landfill_kg"]
LANDSCAPE_OUT = ["crop_kg", "forage_kg", "fert_vol_full_kg", "man_vol_full_kg"]

TOTAL_IN = LANDSCAPE_IN + ["hum_sew_water_kg"]
TOTAL_OUT = LANDSCAPE_OUT

SCOPES = {
    "cropland":  (CROPLAND_IN, CROPLAND_OUT),
    "landscape": (LANDSCAPE_IN, LANDSCAPE_OUT),
    "total":     (TOTAL_IN, TOTAL_OUT),
}
ALL_TERMS = sorted({t for io in SCOPES.values() for grp in io for t in grp})


# --------------------------------------------------------------------------- #
# Distribution helpers
# --------------------------------------------------------------------------- #
_FMIN, _FMAX = 0.2, 5.0


def _lognormal_factor(cv, rng: np.random.Generator, size):
    """Mean-1 lognormal multiplicative factor(s) with the given (scalar) CV."""
    cv = float(cv)
    if cv <= 0:
        return np.ones(size)
    sig = np.sqrt(np.log(1.0 + cv * cv))
    return np.clip(np.exp(-0.5 * sig * sig + sig * rng.standard_normal(size)), _FMIN, _FMAX)


def _lognormal_vec(cv_vec: np.ndarray, rng: np.random.Generator):
    """Mean-1 lognormal factors for a VECTOR of CVs (one draw each). cv=0 -> factor 1."""
    cv_vec = np.asarray(cv_vec, float)
    sig = np.sqrt(np.log(1.0 + cv_vec * cv_vec))
    f = np.exp(-0.5 * sig * sig + sig * rng.standard_normal(cv_vec.shape))
    return np.clip(f, _FMIN, _FMAX)


def _region_ids(iso3) -> np.ndarray:
    """Map each ISO3 to an index into IPCC_REGIONS; unmatched -> 'OTHER'."""
    idx = {r: i for i, r in enumerate(IPCC_REGIONS)}
    other = idx["OTHER"]
    mp = {}
    if REGION_CROSSWALK.exists():
        cw = pd.read_csv(REGION_CROSSWALK)
        mp = dict(zip(cw["iso3"].astype(str), cw["ipcc_nex_region"].astype(str)))
    return np.array([idx.get(mp.get(str(c), "OTHER"), other) for c in iso3], dtype=int)


def _term_region_cv(t: TermSpread) -> np.ndarray:
    """Per-region CV vector (len IPCC_REGIONS) for a region-resolved term; OTHER -> coeff_cv."""
    return np.array([REGION_CV.get(r, {}).get(t.cv_class, t.coeff_cv) for r in IPCC_REGIONS], float)


# --------------------------------------------------------------------------- #
# Core MC
# --------------------------------------------------------------------------- #
@dataclass
class MCResult:
    year: int
    n_draws: int
    iso3: np.ndarray
    surplus: Dict[str, np.ndarray]          # scope -> (n_draws, n_countries) kg N
    global_Tg: Dict[str, np.ndarray]        # scope -> (n_draws,) Tg N
    central_Tg: Dict[str, float]

    def country_summary(self) -> pd.DataFrame:
        out = {"ISO3": self.iso3}
        for scope, arr in self.surplus.items():
            mean = arr.mean(axis=0); sd = arr.std(axis=0)
            lo = np.percentile(arr, 2.5, axis=0); hi = np.percentile(arr, 97.5, axis=0)
            with np.errstate(divide="ignore", invalid="ignore"):
                cv = np.where(mean != 0, sd / np.abs(mean), np.nan)
            out[f"surplus_{scope}_mean_TgN"] = mean / 1e9
            out[f"surplus_{scope}_sd_TgN"] = sd / 1e9
            out[f"surplus_{scope}_p2.5_TgN"] = lo / 1e9
            out[f"surplus_{scope}_p97.5_TgN"] = hi / 1e9
            out[f"surplus_{scope}_cv"] = cv
        return pd.DataFrame(out)

    def global_summary(self) -> Dict[str, Dict[str, float]]:
        res = {}
        for scope, arr in self.global_Tg.items():
            mean = float(arr.mean()); sd = float(arr.std())
            res[scope] = {
                "central_Tg": float(self.central_Tg[scope]),
                "mc_mean_Tg": mean, "sd_Tg": sd,
                "p2.5_Tg": float(np.percentile(arr, 2.5)),
                "p50_Tg": float(np.percentile(arr, 50)),
                "p97.5_Tg": float(np.percentile(arr, 97.5)),
                "rel_uncertainty_pct": (sd / abs(mean) * 100.0) if mean else np.nan,
            }
        return res


def load_central_components(mass_balance_csv: str | Path) -> pd.DataFrame:
    df = pd.read_csv(mass_balance_csv)
    cols = ["ISO3"]
    for t in ALL_TERMS:
        if t not in df.columns:
            df[t] = 0.0
        cols.append(t)
    return df[cols].copy()


def _surplus(perturbed: Dict[str, np.ndarray], scope: str) -> np.ndarray:
    ins, outs = SCOPES[scope]
    return sum(perturbed[k] for k in ins) - sum(perturbed[k] for k in outs)


def _central_global(comp: pd.DataFrame) -> Dict[str, float]:
    g = {t: float(comp[t].sum()) for t in ALL_TERMS}
    out = {}
    for scope, (ins, outs) in SCOPES.items():
        out[scope] = (sum(g[k] for k in ins) - sum(g[k] for k in outs)) / 1e9
    return out


def run_mc_year(mass_balance_csv, n_draws: int = 10000, year: Optional[int] = None, seed: int = 0,
                spreads: Optional[List[TermSpread]] = None, activity_error: bool = True,
                coeff_error: bool = True) -> MCResult:
    """N-draw Monte-Carlo ensemble for one year's national budget, IPCC-Nex regional correlation."""
    spreads = spreads or PARAM_TABLE
    spreads = [t for t in spreads if t.name in ALL_TERMS]
    comp = load_central_components(mass_balance_csv)
    iso3 = comp["ISO3"].to_numpy()
    nC = len(iso3)
    rng = np.random.default_rng(seed)
    region_id = _region_ids(iso3)

    central = {t.name: comp[t.name].to_numpy(float) for t in spreads}
    reg_cv = {t.name: _term_region_cv(t) for t in spreads if t.region_resolved}
    groups = sorted({t.act_group for t in spreads})
    # per-country activity CV per group (FAOSTAT-flag lookup; scalar ACT_CV fallback)
    act_cv_vec = {g: activity_cv_vector(g, iso3, year) for g in groups}

    surplus = {s: np.empty((n_draws, nC)) for s in SCOPES}
    for i in range(n_draws):
        # activity: one factor per group per country, drawn from that country's flag-based CV
        gaf = {g: (_lognormal_vec(act_cv_vec[g], rng) if activity_error else np.ones(nC))
               for g in groups}
        perturbed = {}
        for t in spreads:
            base = central[t.name]
            if not coeff_error:
                cf = 1.0
            elif t.region_resolved:
                cf = _lognormal_vec(reg_cv[t.name], rng)[region_id]     # one draw per region -> per country
            else:
                cf = float(_lognormal_factor(t.coeff_cv, rng, 1)[0])    # one global draw
            perturbed[t.name] = base * cf * gaf[t.act_group]
        for s in SCOPES:
            surplus[s][i] = _surplus(perturbed, s)

    central_Tg = _central_global(comp)
    return MCResult(
        year=int(year) if year is not None else -1,
        n_draws=n_draws, iso3=iso3,
        surplus=surplus,
        global_Tg={s: surplus[s].sum(axis=1) / 1e9 for s in SCOPES},
        central_Tg=central_Tg,
    )


def run_mc_timeseries(outputs_dir, years: List[int], n_draws: int = 10000, seed: int = 0,
                      spreads: Optional[List[TermSpread]] = None) -> pd.DataFrame:
    """Global time-series band (cropland/landscape/total) — one row per year."""
    outputs_dir = Path(outputs_dir)
    rows = []
    for yi, year in enumerate(years):
        csv = outputs_dir / f"mass_balance_N_{year}.csv"
        if not csv.exists():
            continue
        res = run_mc_year(csv, n_draws=n_draws, year=year, seed=seed + yi, spreads=spreads)
        gs = res.global_summary()
        row = {"year": year}
        for scope in SCOPES:
            for k, v in gs[scope].items():
                row[f"{scope}_{k}"] = v
        rows.append(row)
    return pd.DataFrame(rows).sort_values("year").reset_index(drop=True)


def param_table_df() -> pd.DataFrame:
    return pd.DataFrame([
        {"term": t.name, "coeff_cv": t.coeff_cv, "activity_cv": t.activity_cv,
         "correlation": ("region" if t.region_resolved else "global"),
         "cv_class": t.cv_class, "act_group": t.act_group, "source": t.source}
        for t in PARAM_TABLE
    ])
