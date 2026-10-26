"""HUM — human-waste nitrogen input to the environment.

Two methods (config `human.method`):

  'percapita' (legacy): HUM = population x flat per-capita excretion
                        (N: 5.0 kg/cap/yr, Hong et al. 2011; P: 0.6 kg/cap/yr).

  'protein_treated' (N only, recommended): a diet-based, country/year-varying per-capita
      N excretion, corrected for post-retail food loss and netted for wastewater-treatment
      removal, following Van Drecht et al. (2009) and Moree et al. (2013):

          HUM(i,t) = P(i,t) . e_N(i,t) . (1 - f_rem(i,t))

      with
          e_N(i,t) = protein(i,t) . f_N . 365 . (1 - w_c(i,t)) / 1000        [kg N cap^-1 yr^-1]

      - protein(i,t) : FAOSTAT Food Balance Sheets per-capita protein SUPPLY (g cap^-1 d^-1,
        FBS 2010-present / FBSH 1961-2013), measured at the RETAIL level.
      - f_N = 0.16    : mass fraction of N in dietary protein.
      - w_c(i,t)      : post-retail (consumer + household + plate) food-loss fraction that
        converts retail SUPPLY to actual INTAKE (see `_consumer_loss`). FAOSTAT already
        removes farm->retail losses but NOT consumer-stage waste, and humans are in
        approximate N balance (excretion ~ intake), so this correction is required to avoid
        overstating excretion. Regional base fractions (Gustavsson et al. 2011 / FAO), scaled
        over time by economic development.
      - f_rem(i,t) = f_treated(i,t) x R_N : fraction of excreted N removed by wastewater
        treatment (only TREATED sewage removes N). Read from an optional treatment table;
        f_rem=0 (gross) where unavailable. Human N enters the TOTAL surplus only.

Population from coefficients/population_country.csv (FAOSTAT OA element 511, item 3010).
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
from . import utils
from . import faostat_api as fao
from . import faostat_prep as prep
from .config import coef

N_IN_PROTEIN = 0.16          # kg N per kg dietary protein
DAYS_PER_YEAR = 365.0

# FAOSTAT UN sub-region (country_converter 'UNregion') -> Gustavsson et al. (2011) food region.
_UNREGION_TO_FOOD = {
    "Northern Africa": "NAfrWCAsia",
    "Eastern Africa": "SubSaharanAfrica", "Middle Africa": "SubSaharanAfrica",
    "Western Africa": "SubSaharanAfrica", "Southern Africa": "SubSaharanAfrica",
    "Sub-Saharan Africa": "SubSaharanAfrica",
    "Northern America": "NorthAmericaOceania",
    "Australia and New Zealand": "NorthAmericaOceania",
    "Melanesia": "SouthSoutheastAsia", "Micronesia": "SouthSoutheastAsia",
    "Polynesia": "SouthSoutheastAsia",
    "Latin America and the Caribbean": "LatinAmerica",
    "Caribbean": "LatinAmerica", "Central America": "LatinAmerica",
    "South America": "LatinAmerica",
    "Central Asia": "NAfrWCAsia", "Western Asia": "NAfrWCAsia",
    "Eastern Asia": "IndustrializedAsia",                 # Gustavsson: Japan, China, Rep. Korea
    "South-Eastern Asia": "SouthSoutheastAsia", "Southern Asia": "SouthSoutheastAsia",
    "Eastern Europe": "Europe", "Northern Europe": "Europe",
    "Southern Europe": "Europe", "Western Europe": "Europe",
}

_CC = None
_REGION_CACHE: dict[str, str] = {}


def _food_region(iso3: str) -> str:
    """Map an ISO3 to a Gustavsson food-loss region via country_converter UN sub-region."""
    global _CC
    if iso3 in _REGION_CACHE:
        return _REGION_CACHE[iso3]
    try:
        if _CC is None:
            import logging, country_converter as cc
            logging.getLogger("country_converter").setLevel(logging.CRITICAL)
            _CC = cc.CountryConverter()
        un = _CC.convert(iso3, src="ISO3", to="UNregion", not_found=None)
        reg = _UNREGION_TO_FOOD.get(un, "DEFAULT")
    except Exception:  # noqa: BLE001
        reg = "DEFAULT"
    _REGION_CACHE[iso3] = reg
    return reg


def _protein_g_cap_day(cfg, raw_dir: Path, year: int) -> pd.DataFrame | None:
    """FAOSTAT per-capita protein SUPPLY (g cap^-1 d^-1) for `year`, per ISO3.

    Prefer the current FBS release; fill country gaps from the historic FBSH release
    (they overlap 2010-2013). Returns [ISO3, protein_g_cap_day] or None if neither domain
    is present."""
    codes = cfg["faostat"]["codes"]
    el, it = codes["fbs_element_protein"], codes["fbs_item_total"]
    parts = {}
    for dom in ("FBS", "FBSH"):
        try:
            df = fao.read_domain(dom, raw_dir)
        except Exception:  # noqa: BLE001
            continue
        df = fao.filter_rows(df, year=year, element=el, item=it)
        if df is None or len(df) == 0:
            continue
        df = prep.to_iso3(df, cfg)
        g = df.groupby("ISO3", as_index=False)["Value"].mean()
        parts[dom] = g.set_index("ISO3")["Value"]
    if not parts:
        return None
    base = parts.get("FBS")
    if base is None:
        base = parts["FBSH"]
    elif "FBSH" in parts:                                    # fill FBS gaps from FBSH
        base = base.combine_first(parts["FBSH"])
    return (base.rename("protein_g_cap_day").reset_index())


def _gdp_from_faostat_mk(cfg, year: int) -> dict[str, float] | None:
    """{ISO3: GDP per capita} from FAOSTAT Macro Indicators (MK), nearest available year. Auto-detects
    the 'Gross Domestic Product' item and a per-capita element (constant/2015 prices preferred), so no
    hard-coded codes. Returns None if MK is not downloaded/available."""
    from .config import get_path
    try:
        df = fao.read_domain("MK", get_path(cfg, "raw"))
    except Exception:  # noqa: BLE001
        return None
    d = df[df["Item"].astype(str).str.contains("Gross Domestic Product", case=False, na=False)
           & df["Element"].astype(str).str.contains("per capita", case=False, na=False)]
    const = d[d["Element"].astype(str).str.contains("2015|constant", case=False, na=False, regex=True)]
    d = const if len(const) else d
    if d.empty:
        return None
    d = prep.to_iso3(d, cfg)
    d = d[["ISO3", "Year", "Value"]].dropna()
    d["Year"] = d["Year"].astype(int)
    out = {}
    for k, sub in d.groupby("ISO3"):
        j = (sub["Year"] - int(year)).abs().idxmin()
        out[k] = float(sub.loc[j, "Value"])
    return out or None


def _gdp_percapita(cfg, year: int) -> dict[str, float] | None:
    """{ISO3: GDP per capita} for `year`. Prefers an explicit table (coefficients/<human.gdp_file>:
    ISO3,year,gdp_pc — e.g. Maddison), else falls back to FAOSTAT Macro Indicators (MK). Returns None
    if neither is available (callers then hold the GDP-driven trajectories time-constant)."""
    fname = cfg["human"].get("gdp_file", "gdp_percapita.csv")
    path = coef(cfg, fname)
    if not path.exists():
        return _gdp_from_faostat_mk(cfg, year)
    g = pd.read_csv(path)
    cols = {c.lower(): c for c in g.columns}
    iso, yr, val = cols.get("iso3"), cols.get("year"), cols.get("gdp_pc", cols.get("gdppc"))
    if not (iso and yr and val):
        return None
    g = g[[iso, yr, val]].dropna()
    g[yr] = g[yr].astype(int)
    out = {}
    for k, sub in g.groupby(iso):
        j = (sub[yr] - int(year)).abs().idxmin()             # nearest available year
        out[k] = float(sub.loc[j, val])
    return out


def _logistic(x, k=1.5, x0=4.0):
    """Saturating development weight; x = log10(GDP per capita). x0=4.0 (~$10,000 cap^-1)
    is the midpoint, so low-income economies sit low on the curve and high-income ones
    saturate toward 1."""
    import math
    return 1.0 / (1.0 + math.exp(-k * (x - x0)))


def _consumer_loss(cfg, raw_dir: Path, year: int, iso_list) -> dict[str, float]:
    """Post-retail (retail + household) food-loss fraction w_c(i,t) for each ISO3.

    From Moree et al. (2013) SI Table 2 (= Gustavsson / SIK-FAO 2011): a regional retail+household
    food-loss fraction given at 1900 and 2010, linearly interpolated between those endpoints and held
    flat outside 1900-2010. w_c is applied to FAOSTAT retail protein supply to obtain intake/excretion
    (Eq. 12). coefficients/food_consumer_loss.csv holds region, w_c_1900, w_c_2010."""
    import numpy as np
    tbl = pd.read_csv(coef(cfg, cfg["human"].get("consumer_loss_file", "food_consumer_loss.csv")),
                      comment="#")
    v1900 = {r["region"]: float(r["w_c_1900"]) for _, r in tbl.iterrows()}
    v2010 = {r["region"]: float(r["w_c_2010"]) for _, r in tbl.iterrows()}
    out = {}
    for iso in iso_list:
        reg = _food_region(iso)
        a = v1900.get(reg, v1900.get("DEFAULT", 0.067))
        b = v2010.get(reg, v2010.get("DEFAULT", 0.13))
        out[iso] = float(np.interp(int(year), [1900, 2010], [a, b]))
    return out


def _treatment_removal(cfg, year: int, iso_list) -> dict[str, float]:
    """Fraction of excreted N removed by wastewater treatment, f_rem(i,t) = f_treated x R_N.

    Read from optional coefficients/<human.treatment_file> (ISO3,year,f_rem). Returns 0 (gross,
    no removal) for any country/year not covered, so the model runs before the treatment layer
    (Jones 2021 + Van Drecht/Moree back-cast) is downloaded. TOTAL surplus only."""
    fname = cfg["human"].get("treatment_file", "wastewater_treatment.csv")
    path = coef(cfg, fname)
    if not path.exists():
        print("  [human] no treatment table; f_rem=0 (gross human N) — download to enable netting")
        return {iso: 0.0 for iso in iso_list}
    t = pd.read_csv(path)
    cols = {c.lower(): c for c in t.columns}
    iso, yr, val = cols.get("iso3"), cols.get("year"), cols.get("f_rem")
    if not (iso and yr and val):
        return {iso: 0.0 for iso in iso_list}
    t = t[[iso, yr, val]].dropna()
    t[yr] = t[yr].astype(int)
    out = {}
    for k, sub in t.groupby(iso):
        j = (sub[yr] - int(year)).abs().idxmin()
        out[k] = min(1.0, max(0.0, float(sub.loc[j, val])))
    return {i: out.get(i, 0.0) for i in iso_list}


def _treated_fraction(cfg, year: int, iso_list) -> dict[str, float]:
    """Fraction of excreted N that is TREATED, f_treated(i,t), from coefficients/<treatment_file>
    (ISO3,year,f_treated; built by scripts/45 alongside f_rem, where f_rem = f_treated x R_N so
    f_rem <= f_treated). Used to split the sewered water discharge into UNTREATED (collected but not
    treated, c - f_treated) and TREATED EFFLUENT (treated but not N-removed, f_treated - f_rem);
    both go to surface water. Returns 0 where unavailable (then untreated = collected)."""
    fname = cfg["human"].get("treatment_file", "wastewater_treatment.csv")
    path = coef(cfg, fname)
    if not path.exists():
        return {iso: 0.0 for iso in iso_list}
    t = pd.read_csv(path)
    cols = {c.lower(): c for c in t.columns}
    iso, yr, val = cols.get("iso3"), cols.get("year"), cols.get("f_treated")
    if not (iso and yr and val):
        return {iso: 0.0 for iso in iso_list}
    t = t[[iso, yr, val]].dropna()
    t[yr] = t[yr].astype(int)
    out = {}
    for k, sub in t.groupby(iso):
        j = (sub[yr] - int(year)).abs().idxmin()
        out[k] = min(1.0, max(0.0, float(sub.loc[j, val])))
    return {i: out.get(i, 0.0) for i in iso_list}


def _population(cfg, year: int) -> pd.DataFrame:
    path = coef(cfg, cfg["human"]["population_file"])
    pop = utils.load_coef(path, "national population",
                          required_cols=["ISO3", "year", "population"])
    pop = pop[pop["year"].astype(int) == int(year)]
    return pop[["ISO3", "population"]].copy()


def _recycle_country_anchor(cfg, year: int) -> dict[str, float]:
    """{ISO3: f_recycle} for documented high-recycling countries from
    coefficients/<human.recycle_country_file> (ISO3,year,f_recycle), linearly interpolated to `year`
    and held flat outside each country's anchor range. Empty if the file is absent."""
    import numpy as np
    path = coef(cfg, cfg["human"].get("recycle_country_file", "human_recycle_country.csv"))
    if not path.exists():
        return {}
    t = pd.read_csv(path, comment="#")
    cols = {c.lower(): c for c in t.columns}
    iso, yr, val = cols.get("iso3"), cols.get("year"), cols.get("f_recycle")
    if not (iso and yr and val):
        return {}
    t = t[[iso, yr, val]].dropna()
    t[yr] = t[yr].astype(int); t[val] = t[val].astype(float)
    out = {}
    for k, sub in t.groupby(iso):
        sub = sub.sort_values(yr)
        out[k] = float(np.interp(int(year), sub[yr].to_numpy(), sub[val].to_numpy()))  # flat outside range
    return out


def _recycle_fraction(cfg, raw_dir: Path, year: int, iso_list) -> dict[str, float]:
    """Fraction of TOTAL human excreta N (rural + urban) recycled onto cropland, f_recycle(i,t).

    Two-tier reconstruction (broader than Moree's urban-only model, which globally is small):
      1. **Documented country trajectories** (night-soil / biosolids reuse) for the well-studied
         high-recycling cases — China above all (near-universal pre-1980, collapsing to ~0.1 by 2020),
         plus any others listed in `recycle_country_file`. Linearly interpolated over time.
      2. **Regional default** for every other country: the Moree et al. (2013) 1900 recycling class
         (coefficients/human_recycle.csv), collapsed to ~15% of the 1900 value by 1950 (the recycling
         RATE of non-sewered excreta), applied to the NON-SEWERED share (1 - c):
             f_recycle(i,t) = f_recycle_1900(region) . 0.15 . (1 - c(i,t))
         where c is the COLLECTED (sewer-connection) fraction from Jones et al. 2021 back-cast
         (coefficients/sewer_connection.csv). If the sewer table is absent, (1 - c) falls back to a GDP
         proxy (1 - s)^p (s = logistic in log10 GDP/cap, p = recycle_decline_power).

    Captures the large, real collapse of Asian (esp. Chinese) human-waste recycling over 1961-2020 that
    Moree's urban model misses. The country-anchor values are literature estimates (China N-budget and
    night-soil compilations; sources in human_recycle_country.csv) and carry a wide CV in the Monte Carlo."""
    import math
    tbl = pd.read_csv(coef(cfg, cfg["human"].get("recycle_file", "human_recycle.csv")), comment="#")
    base = {r["region"]: float(r["f_recycle_1900"]) for _, r in tbl.iterrows()}
    p = float(cfg["human"].get("recycle_decline_power", 1.0))
    f1950 = float(cfg["human"].get("recycle_post1950_factor", 0.15))   # Moree: 1900->1950 collapse
    sewer = _sewer_connection(cfg, year)                 # explicit non-sewered share (Jones), else {}
    gdp = None if sewer else _gdp_percapita(cfg, year)   # GDP only for the proxy fallback
    anchors = _recycle_country_anchor(cfg, year)
    if not sewer and gdp is None:
        print("  [recycle] no sewer/GDP table; regional-default f_recycle held at post-1950 class "
              f"level (anchored countries: {sorted(anchors)})")
    out = {}
    for iso in iso_list:
        if iso in anchors:                                  # documented trajectory (rural + urban)
            out[iso] = max(0.0, min(1.0, anchors[iso]))
            continue
        # default tier: Moree 1900 class collapsed to ~15% by 1950 = the recycling rate of NON-SEWERED
        # excreta; multiplied by the non-sewered share (1 - c), c = Jones collected/sewer-connection back-cast.
        # Falls back to a GDP proxy (1 - s)^p for the non-sewered share if the sewer table is absent.
        b = base.get(_food_region(iso), base.get("DEFAULT", 0.2))
        if sewer:
            nonsewered = 1.0 - sewer.get(iso, 0.0)
        elif gdp is not None and iso in gdp and gdp[iso] > 0:
            nonsewered = (1.0 - _logistic(math.log10(gdp[iso]))) ** p
        else:
            nonsewered = 1.0
        out[iso] = max(0.0, b * f1950 * nonsewered)
    return out


def _sewer_connection(cfg, year: int) -> dict[str, float]:
    """{ISO3: c} COLLECTED (sewer-connection) fraction for `year` from coefficients/<recycle_sewer_file>
    (ISO3,year,c — legacy column 'D' also accepted; built by scripts/45 from Jones WWc + GDP back-cast).
    Empty if the file is absent."""
    path = coef(cfg, cfg["human"].get("recycle_sewer_file", "sewer_connection.csv"))
    if not path.exists():
        return {}
    t = pd.read_csv(path)
    cols = {c.lower(): c for c in t.columns}
    iso, yr, val = cols.get("iso3"), cols.get("year"), cols.get("c", cols.get("d"))
    if not (iso and yr and val):
        return {}
    t = t[[iso, yr, val]].dropna(); t[yr] = t[yr].astype(int)
    out = {}
    for k, sub in t.groupby(iso):
        j = (sub[yr] - int(year)).abs().idxmin()
        out[k] = min(1.0, max(0.0, float(sub.loc[j, val])))
    return out


def _beta_bio(cfg, iso_list) -> dict[str, float]:
    """{ISO3: beta_bio} — fraction of the treatment-REMOVED nitrogen (f_rem = c.tau.R_N) that is
    land-applied to cropland as biosolids, from coefficients/<human.biosolids_beta_file> (ISO3,beta_bio
    + a DEFAULT row). The sewered-biosolids term carves biosolids OUT of the removed pool:
        f_bio = beta_bio . f_rem                         (Tier-2 countries only; f_bio = 0 for Tier-1)
    so night soil, biosolids, true removal and discharge form a disjoint partition of excreted N. Central
    0.20 (land-applying regime; ~ N-to-sludge share of removed N x land-applied share, calibrated so
    France total recycling ~ Starck 2023's ~10%), overridden down for incineration-dominant countries."""
    default = float(cfg["human"].get("biosolids_beta_default", 0.20))
    path = coef(cfg, cfg["human"].get("biosolids_beta_file", "biosolids_beta.csv"))
    if not path.exists():
        return {i: default for i in iso_list}
    t = pd.read_csv(path, comment="#")
    cols = {c.lower(): c for c in t.columns}
    iso, val = cols.get("iso3"), cols.get("beta_bio")
    if not (iso and val):
        return {i: default for i in iso_list}
    m = {str(r[iso]): float(r[val]) for _, r in t.iterrows() if pd.notna(r[val])}
    d = m.get("DEFAULT", default)
    return {i: m.get(i, d) for i in iso_list}




def _bio_routes(cfg, iso_list, year: int | None = None) -> dict[str, tuple]:
    """{ISO3: (a_ag, a_other, a_lf, a_inc)} — sludge disposal-route fractions (share of sludge N sent to
    cropland / non-crop land / landfill / incineration), normalised to sum to 1 (NEBRA 2018 US; EurEau 2021
    / Collivignarelli 2019 EU; Wei 2020 / Yue 2024 China; incineration-dominant NL/BE/CH/DE/AT/JP).

    TIME-VARYING (default): if `biosolids_routes_byyear_file` is set and `year` is given, each fraction is
    linearly interpolated between anchor years per country (held flat outside the anchor range, same
    convention as human_recycle_country.csv). This captures documented regime shifts — DE 2017 AbfKlaerV
    incineration mandate, NL ~1995 farmland ban, UK 1999 end-of-sea-disposal, US 1988 ocean-dumping ban —
    rather than back-projecting the present-day split onto the whole 1961-2023 record. Falls back to the
    static `biosolids_routes_file` slice when the byyear file is absent or `year` is None. Countries not
    listed use DEFAULT."""
    keys = ("a_ag", "a_other", "a_lf", "a_inc")
    dflt = {"a_ag": 0.35, "a_other": 0.10, "a_lf": 0.25, "a_inc": 0.30}
    rows = {}
    byfile = cfg["human"].get("biosolids_routes_byyear_file")
    if year is not None and byfile:                                     # time-varying (anchor-year interp)
        bpath = coef(cfg, byfile)
        if bpath.exists():
            import numpy as np
            t = pd.read_csv(bpath, comment="#")
            cols = {c.lower(): c for c in t.columns}
            ic, yc = cols.get("iso3"), cols.get("year")
            if ic and yc and all(k in cols for k in keys):
                t = t[[ic, yc] + [cols[k] for k in keys]].dropna()
                t[yc] = t[yc].astype(int)
                for iso, sub in t.groupby(ic):
                    sub = sub.sort_values(yc)
                    yy = sub[yc].to_numpy()
                    rows[str(iso)] = {k: float(np.interp(int(year), yy, sub[cols[k]].astype(float).to_numpy()))
                                      for k in keys}
    if not rows:                                                        # static present-day slice (fallback)
        path = coef(cfg, cfg["human"].get("biosolids_routes_file", "biosolids_routes.csv"))
        if path.exists():
            t = pd.read_csv(path, comment="#")
            cols = {c.lower(): c for c in t.columns}
            ic = cols.get("iso3")
            if ic and all(k in cols for k in keys):
                for _, r in t.iterrows():
                    rows[str(r[ic])] = {k: float(r[cols[k]]) for k in keys}
    base = rows.get("DEFAULT", dflt)

    def norm(rt):
        v = [max(0.0, float(rt[k])) for k in keys]
        s = sum(v)
        return tuple(x / s for x in v) if s > 0 else (0.35, 0.10, 0.25, 0.30)
    return {i: norm(rows.get(i, base)) for i in iso_list}


_HB_CACHE: dict = {}


def _human_budget(cfg, raw_dir: Path, year: int):
    """Memoized wrapper : compute(), food_waste() and human_recycle() each call the human
    budget for the same year; computing it once removes ~30 s of repeated FBS reads per model year.
    Returns a fresh copy each time, so callers that add columns cannot alter the cached table."""
    key = (int(year), str(raw_dir), repr(sorted((cfg.get("human") or {}).items())),
           repr(cfg.get("faostat", {}).get("codes", {})))
    if key not in _HB_CACHE:
        _HB_CACHE[key] = _human_budget_impl(cfg, raw_dir, year)
    v = _HB_CACHE[key]
    return None if v is None else v.copy()


def _human_budget_impl(cfg, raw_dir: Path, year: int):
    """Shared per-capita human-N terms for method='protein_treated'. Returns a DataFrame
    [ISO3, population, e_supply, e_N, w_c, f_rem, f_treated, c, f_recycle, f_sludge] (per-capita rates in
    kg N cap^-1 yr^-1), or None if FBS/FBSH protein is unavailable. e_supply = retail food-supply N;
    e_N = excreted N (= e_supply x (1-w_c), food loss removed). f_recycle = non-sewered night-soil
    recycled fraction; f_sludge = sewage-sludge fraction of excreted N (= s_bio x f_treated, capped at
    f_rem) routed to disposal by the a_x fractions in human_recycle()."""
    pop = _population(cfg, year)
    prot = _protein_g_cap_day(cfg, raw_dir, year)
    if prot is None or len(prot) == 0:
        return None
    df = pop.merge(prot, on="ISO3", how="left")
    if df["protein_g_cap_day"].isna().any():           # gap-fill with population-weighted global mean
        gmean = (df["protein_g_cap_day"] * df["population"]).sum() / \
                df.loc[df["protein_g_cap_day"].notna(), "population"].sum()
        df["protein_g_cap_day"] = df["protein_g_cap_day"].fillna(gmean)
    iso_list = df["ISO3"].tolist()
    f_N = float(cfg["human"].get("protein_n_fraction", N_IN_PROTEIN))
    df["w_c"] = df["ISO3"].map(_consumer_loss(cfg, raw_dir, year, iso_list)).astype(float)
    df["f_rem"] = df["ISO3"].map(_treatment_removal(cfg, year, iso_list)).astype(float)
    df["f_treated"] = df["ISO3"].map(_treated_fraction(cfg, year, iso_list)).astype(float)   # treated share
    _c = _sewer_connection(cfg, year)                                   # collected (sewer-connection) fraction c(i,t)
    df["c"] = df["ISO3"].map(_c).astype(float).fillna(0.0)
    df["f_recycle"] = (df["ISO3"].map(_recycle_fraction(cfg, raw_dir, year, iso_list)).astype(float)
                       if cfg["human"].get("recycle", True) else 0.0)
    # BOUND night-soil recycling at the NON-SEWERED pool: f_ns = min(f_recycle, 1-c). Night soil can only
    # draw on non-collected excreta (HUM_nc = P.e_N.(1-c)); the reused fraction of the pool lambda_ns =
    # f_ns/(1-c) <= 1. For documented Tier-1 high-recyclers whose raw trajectory exceeds 1-c, the excess is
    # collected excreta and flows through the sewered pathway instead (Eq 41: HUM_crop,ns = lambda_ns.HUM_nc).
    df["f_recycle"] = df["f_recycle"].clip(upper=(1.0 - df["c"]).clip(lower=0.0))
    if cfg["human"].get("biosolids", True):
        s_bio = float(cfg["human"].get("sludge_biomass_fraction", 0.18))
        is_t1 = df["ISO3"].isin(_recycle_country_anchor(cfg, year).keys())   # Tier-1: f_ns is TOTAL recycling
        # Sludge N as a fraction of excreted N = s_bio x f_treated (biomass assimilation of the treated
        # flow; ~stable across treatment levels, Sagberg et al. 1998 / ORNL), capped at the removed
        # fraction f_rem (for low-treatment cases all removed N is settled sludge). The remainder of the
        # removed N (f_rem - f_sludge) is denitrified to N2 and lost to the atmosphere. Tier-1 documented recyclers set f_sludge=0 (their f_ns is TOTAL recycling).
        df["f_sludge"] = (s_bio * df["f_treated"]).clip(upper=df["f_rem"]).where(~is_t1, 0.0)
    else:
        df["f_sludge"] = 0.0
    df["e_supply"] = df["protein_g_cap_day"].astype(float) * f_N * DAYS_PER_YEAR / 1000.0
    df["e_N"] = df["e_supply"] * (1.0 - df["w_c"])
    return df


def compute(cfg, raw_dir: Path, year: int, nutrient: str = "N") -> pd.DataFrame:
    method = cfg["human"].get("method", "percapita")
    pop = _population(cfg, year)

    # P (and the legacy path for N) use the flat per-capita rate — no sewered/non-sewered split.
    if nutrient != "N" or method != "protein_treated":
        rate = (cfg["human"]["n_excretion_kg_per_capita_yr"] if nutrient == "N"
                else cfg["human"]["p_excretion_kg_per_capita_yr"])
        out = pop[["ISO3"]].copy()
        out["hum_sewage_kg"] = pop["population"].astype(float) * rate
        out["hum_sew_water_kg"] = out["hum_sewage_kg"]; out["hum_sew_land_kg"] = 0.0
        return out.reset_index(drop=True)

    # Protein-based excretion, food-loss-corrected, netted for cropland recycling AND treatment.
    b = _human_budget(cfg, raw_dir, year)
    if b is None:
        print("  [human] FBS/FBSH protein unavailable; falling back to flat per-capita N rate")
        out = pop[["ISO3"]].copy()
        out["hum_sewage_kg"] = pop["population"].astype(float) * cfg["human"]["n_excretion_kg_per_capita_yr"]
        out["hum_sew_water_kg"] = out["hum_sewage_kg"]; out["hum_sew_land_kg"] = 0.0
        return out.reset_index(drop=True)
    # HUM = excreta N DISCHARGED to the environment: the disjoint remainder NOT night-soil-recycled
    # and NOT treatment-removed. It is split by the COLLECTED (sewer-connection) fraction c into the
    # SEWERED flux (-> surface water; total surplus only) and the NON-SEWERED flux (-> deposited on
    # land; LANDSCAPE + total):
    #   hum_sew_land  = e_N . max(0, 1 - c - f_ns)              (non-sewered latrine/open discharge -> land)
    #   hum_sew_water = hum_sewage - hum_sew_land               (sewered effluent -> surface water)
    # (For Tier-1 high-recyclers where f_ns > 1-c, hum_sew_land clips to 0 and the small discharge is
    # treated as sewered residual.)
    excreted = b["population"] * b["e_N"]
    b["hum_sewage_kg"] = excreted * (1.0 - b["f_recycle"] - b["f_rem"]).clip(lower=0.0)
    b["hum_sew_land_kg"] = excreted * (1.0 - b["c"] - b["f_recycle"]).clip(lower=0.0)
    b["hum_sew_land_kg"] = b[["hum_sew_land_kg", "hum_sewage_kg"]].min(axis=1)   # never exceed total
    b["hum_sew_water_kg"] = (b["hum_sewage_kg"] - b["hum_sew_land_kg"]).clip(lower=0.0)
    # Granular split of the sewered water discharge (both -> surface water / TOTAL): UNTREATED collected
    # (c - f_treated) + TREATED EFFLUENT (treated but N not removed, f_treated - f_rem). These re-sum to
    # hum_sew_water = e_N*(c - f_rem) since f_treated <= c and f_rem <= f_treated by construction.
    b["hum_c_utreat_kg"] = excreted * (b["c"] - b["f_treated"]).clip(lower=0.0)
    b["hum_effluent_kg"] = excreted * (b["f_treated"] - b["f_rem"]).clip(lower=0.0)
    print(f"  [human] e_N {b['e_N'].min():.1f}-{b['e_N'].max():.1f} kg N/cap/yr; "
          f"w_c {b['w_c'].min():.2f}-{b['w_c'].max():.2f}; f_rem {b['f_rem'].mean():.2f}; "
          f"f_recycle {b['f_recycle'].mean():.2f}; f_sludge {b['f_sludge'].mean():.3f}; "
          f"sewage total {b['hum_sewage_kg'].sum()/1e9:.1f} Tg (water {b['hum_sew_water_kg'].sum()/1e9:.1f} "
          f"+ land {b['hum_sew_land_kg'].sum()/1e9:.1f})")
    from . import dissolution
    return dissolution.redistribute_table(
        b[["ISO3", "hum_sewage_kg", "hum_sew_water_kg", "hum_sew_land_kg",
           "hum_c_utreat_kg", "hum_effluent_kg"]].reset_index(drop=True),
        cfg, raw_dir, "population")


def food_waste(cfg, raw_dir: Path, year: int, nutrient: str = "N") -> pd.DataFrame:
    """FOOD_waste — N embodied in consumer-wasted food (retail + household + plate waste), returned
    to the landscape via the solid-waste stream (landfill / compost / disposal).

    This is the mass-balance counterpart of the consumer food-loss correction w_c applied to human
    excretion (see `compute`): the food N that reaches the consumer but is NOT eaten does not leave
    the budget — it re-enters the landscape as food waste. Retail food-supply N is split into an
    eaten fraction (1-w_c) that is excreted (the human-waste term) and a wasted fraction w_c that
    becomes food waste:

        FOOD_waste(i,t) = P(i,t) . e_supply(i,t) . w_c(i,t)
        e_supply(i,t)   = protein(i,t) . f_N . 365 / 1000          [kg N cap^-1 yr^-1, retail level]

    A LANDSCAPE/total input (not a cropland input; not treatment-netted, since food waste bypasses
    wastewater treatment). Analogous to the FOOD_p,waste term of Van Meter et al. (2021), but kept
    as a single landscape input here rather than routed to landfill/surface-water/cropland via a
    compartment model. N only, and only when method='protein_treated' with FBS/FBSH protein present;
    otherwise disabled (empty frame)."""
    if (nutrient != "N" or cfg["human"].get("method") != "protein_treated"
            or not cfg["human"].get("food_waste", True)):
        return pd.DataFrame(columns=["ISO3", "food_waste_kg"])
    b = _human_budget(cfg, raw_dir, year)
    if b is None:
        print("  [food_waste] FBS/FBSH protein unavailable; food-waste term = 0")
        return pd.DataFrame(columns=["ISO3", "food_waste_kg"])
    b["food_waste_kg"] = b["population"] * b["e_supply"] * b["w_c"]
    print(f"  [food_waste] {b['food_waste_kg'].sum()/1e9:.2f} Tg N (consumer-wasted food -> landscape)")
    from . import dissolution
    return dissolution.redistribute_table(
        b[["ISO3", "food_waste_kg"]].reset_index(drop=True), cfg, raw_dir, "population")


def human_recycle(cfg, raw_dir: Path, year: int, nutrient: str = "N") -> pd.DataFrame:
    """HUM_recycle (hum_crop) — human excreta N recycled onto CROPLAND, plus the landfilled-sludge term.

        hum_ns(i,t)       = P . e_N . f_ns              (non-sewered NIGHT SOIL     -> rural cropland)
        hum_bio(i,t)      = a_ag . P . e_N . f_sludge   (sewage-sludge BIOSOLIDS    -> cropland)
        hum_crop(i,t)     = hum_ns + hum_bio
        hum_landfill(i,t) = a_lf . P . e_N . f_sludge   (landfilled sludge -> retained in landscape)

    f_ns is the night-soil recycled fraction (`_recycle_fraction`); f_sludge = s_bio.f_treated (capped at
    f_rem) is the sewage-sludge fraction of excreted N (see `_human_budget`). The recovered sludge N is
    apportioned three ways by the national disposal fractions (`_bio_routes`): LAND APPLICATION (a_ag,
    where a_ag now absorbs the former separate 'other-land' route a_other) -> cropland biosolids;
    LANDFILL (a_lf) -> hum_landfill, retained in the landscape (not cropland); and INCINERATION (a_inc)
    -> atmosphere (N volatilizes ~completely in combustion; Werther & Ogada 1999; Donatello & Cheeseman
    2013), which leaves the budget entirely. Cropland biosolids enter the cropland, landscape and total
    surpluses; landfilled sludge enters the landscape and total surpluses only. Returns
    [ISO3, hum_crop_kg, hum_ns_kg, hum_bio_kg, hum_landfill_kg]. N only, method='protein_treated' +
    human.recycle; else empty."""
    cols = ["ISO3", "hum_crop_kg", "hum_ns_kg", "hum_bio_kg", "hum_landfill_kg"]
    if (nutrient != "N" or cfg["human"].get("method") != "protein_treated"
            or not cfg["human"].get("recycle", True)):
        return pd.DataFrame(columns=cols)
    b = _human_budget(cfg, raw_dir, year)
    if b is None:
        return pd.DataFrame(columns=cols)
    # Sludge N (= s_bio x treated, capped at removed; see _human_budget) is apportioned by the national
    # disposal-route fractions. LAND APPLICATION = a_ag + a_other (the crop-vs-other-land split exceeds the
    # data resolution, so all land-applied biosolids -> cropland); LANDFILL a_lf -> landscape (retained);
    # INCINERATION a_inc -> atmosphere (all N volatilized in combustion), leaving the budget.
    routes = _bio_routes(cfg, b["ISO3"].tolist(), year)
    a_land = (b["ISO3"].map(lambda i: routes[i][0]).astype(float)       # a_ag (agricultural)
              + b["ISO3"].map(lambda i: routes[i][1]).astype(float))    # + a_other (other-land) -> land application
    a_lf = b["ISO3"].map(lambda i: routes[i][2]).astype(float)
    sludge = b["population"] * b["e_N"] * b["f_sludge"]                  # sludge N (s_bio x treated, capped)
    b["hum_ns_kg"] = b["population"] * b["e_N"] * b["f_recycle"]          # night soil -> cropland
    b["hum_bio_kg"] = a_land * sludge                                     # land-applied biosolids -> cropland
    b["hum_crop_kg"] = b["hum_ns_kg"] + b["hum_bio_kg"]
    b["hum_landfill_kg"] = a_lf * sludge                                  # landfilled sludge -> landscape (retained)
    print(f"  [hum_crop] {b['hum_crop_kg'].sum()/1e9:.2f} Tg N -> cropland "
          f"(night soil {b['hum_ns_kg'].sum()/1e9:.2f} + biosolids {b['hum_bio_kg'].sum()/1e9:.2f}); "
          f"sludge {sludge.sum()/1e9:.2f} Tg; landfill -> landscape {b['hum_landfill_kg'].sum()/1e9:.2f} Tg")
    from . import dissolution
    return dissolution.redistribute_table(b[cols].reset_index(drop=True), cfg, raw_dir, "population")
