#!/usr/bin/env python3
"""Build coefficients/sewer_connection.csv (ISO3, year, c) and coefficients/wastewater_treatment.csv
(ISO3, year, f_rem, f_treated) — the human-waste COLLECTION and treatment-removal trajectories.

COLLECTED FRACTION c(i,t) — sewer-connection share of the population, built for EVERY model country
by a four-tier cascade (Jones et al. 2021 is NO LONGER used on the c side):

  1. OBSERVED (UNSD):   United Nations Statistics Division, "Population connected to wastewater
     collecting system" (% of resident population; 2000-2021). Observations kept as reported, interior
     gaps interpolated, pre-first-observation back-cast by the GDP development ratio (see _c_from_un).
  2. OBSERVED (anchor): our hand-built country anchors (coefficients/sewer_connection_anchors.csv),
     compiled from country-specific sources (IDB, WHO-EMR, AMCOW, national WATSAN plans, etc.). Held at
     the anchor value from its year forward; earlier years back-cast by the GDP development ratio.
  3. REGION-CALIBRATED GDP LOGISTIC: countries with no observation but with per-capita GDP get
        c(i,t) = clip( L(GDP_i,t) . m_region , 0, 1 ),
     where L is the global GDP-sewerage logistic (Eq. below) and m_region (median c/L over each region's
     observed members) calibrates the global curve to the region's observed level (Sub-Saharan Africa
     m<1 — less sewerage than income predicts; Middle East / W Europe m>1 — more).
  4. REGIONAL MEDIAN: small island states / territories with neither an observation nor GDP take the
     median observed c of their IPCC-Nex region.

  L(gdp) = 1 / (1 + exp(-K (log10 gdp - X0))),  X0=3.95, K=2.62  (FITTED to the UNSD collection panel
  vs Maddison-2023 GDP, n=1215 country-years, r=0.75). All trajectories are RATCHETED non-decreasing on
  their extrapolated segments (durable infrastructure does not reverse when GDP falls); zero before ~1950.

TREATED FRACTION f_treated(i,t) — treated share of the population — mirrors the c cascade (Jones et al.
2021 is NO LONGER used):
  1. OBSERVED (UNSD): "Population connected to wastewater treatment" (% of resident population; 2000-2021,
     100 countries), built on the same observation/interpolation/GDP-back-cast trajectory as c and then
     clamped to f_treated <= c(i,t) (treatment is a strict subset of collection).
  2. REGIONAL treated-of-collected: countries without an observed treatment series take
     f_treated(i,t) = tau_region . c(i,t), where tau_region is the median treatment/collection ratio over
     that region's doubly-observed members.
R_N (effective N-removal, Moree SI: primary 0.10 -> tertiary 0.80) ramps with development to represent the
treatment-level mix: R_N = R_LO + (R_HI-R_LO).L(gdp); f_rem = f_treated . R_N.

Run once:  python scripts/45_build_treatment.py
"""
from __future__ import annotations
import math
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
GDP_DIR = ROOT / "data/raw/gdp"
ANCHORS = ROOT / "coefficients/sewer_connection_anchors.csv"
REGION_XWALK = ROOT / "coefficients/ipcc_nex_region_crosswalk.csv"
POP = ROOT / "coefficients/population_country.csv"


def _newest_maddison():
    import re
    hits = sorted(GDP_DIR.glob("mpd*.xlsx"))

    def yr(p):
        m = re.search(r"mpd(\d{4})", p.name)
        return int(m.group(1)) if m else 0
    return max(hits, key=yr) if hits else GDP_DIR / "mpd2023_web.xlsx"


MADDISON = _newest_maddison()
OUT = ROOT / "coefficients/wastewater_treatment.csv"
OUT_SEWER = ROOT / "coefficients/sewer_connection.csv"
UN_SEWER_XLSX = ROOT / "data/raw/human_waste_sewered_fraction/Population connected to wastewater collecting system.xlsx"
UN_TREAT_XLSX = ROOT / "data/raw/human_waste_sewered_fraction/Population connected to wastewater treatment.xlsx"
# OECD Environment (DF_WATER_TREAT): 'Total public sewerage' (TOTPUBSEW) and 'Public total treatment'
# (PUBTOTTR), % of population, 1970-2024 (51 / 49 countries). Secondary OBSERVED source that extends the
# UNSD series back to 1970-1985 for OECD members; agrees with UNSD to r~0.99 over the overlapping years.
OECD_CSV = ROOT / "data/raw/human_waste_sewered_fraction/OECD.ENV.EPI,DSD_WATER_TREAT@DF_WATER_TREAT,+all.csv"
OECD_MEASURE = {"c": "TOTPUBSEW", "f_treated": "PUBTOTTR"}
YEARS = range(1961, 2024)
X0, K = 3.95, 2.62         # global GDP-sewerage logistic midpoint / slope (UNSD panel vs Maddison-2023)
R_LO, R_HI = 0.10, 0.80    # effective N-removal ramp (primary-like -> tertiary), Van Drecht/Moree
REF_YEAR = 2015


def L(gdp):                # development weight / global sewerage logistic in [0,1]
    return 1.0 / (1.0 + math.exp(-K * (math.log10(gdp) - X0))) if gdp and gdp > 0 else 0.0


def _un_sewered(conv, path=UN_SEWER_XLSX, label="c"):
    """{ISO3: {year: frac}} OBSERVED share from a UNSD workbook (percent -> fraction). Used for both the
    'population connected to wastewater collecting system' (c) and '...to wastewater treatment' (f_treated)
    tables, which share an identical layout."""
    if not path.exists():
        print(f"  [{label}] UNSD workbook not found ({path.name}); model only")
        return {}
    df = pd.read_excel(path, sheet_name="Data", header=0)
    name_col = df.columns[1]
    ycols = [c for c in df.columns if str(c).strip().isdigit() and 1980 <= int(str(c).strip()) <= 2035]
    iso = conv.convert(df[name_col].astype(str).tolist(), src="regex", to="ISO3", not_found=None)
    out = {}
    for i, code in enumerate(iso):
        if not (isinstance(code, str) and len(code) == 3 and code.isalpha()):
            continue
        d = {}
        for yc in ycols:
            v = pd.to_numeric(df.iloc[i][yc], errors="coerce")
            if pd.notna(v) and 0.0 <= float(v) <= 100.0:
                d[int(str(yc).strip())] = float(v) / 100.0
        if d:
            out[code] = d
    print(f"  [{label}] UNSD observed: {len(out)} countries (2000-2021)")
    return out


def _oecd_series(measure, label):
    """{ISO3: {year: frac}} from the OECD DF_WATER_TREAT CSV for a given MEASURE (REF_AREA is ISO3,
    OBS_VALUE is % of population). Secondary observed source extending UNSD back to 1970."""
    if not OECD_CSV.exists():
        print(f"  [{label}] OECD CSV not found ({OECD_CSV.name}); UNSD only")
        return {}
    d = pd.read_csv(OECD_CSV)
    d = d[d["MEASURE"] == measure][["REF_AREA", "TIME_PERIOD", "OBS_VALUE"]].dropna()
    out = {}
    for iso, sub in d.groupby("REF_AREA"):
        code = str(iso)
        if len(code) != 3 or not code.isalpha():
            continue
        dd = {int(y): float(v) / 100.0 for y, v in zip(sub["TIME_PERIOD"], sub["OBS_VALUE"])
              if 0.0 <= float(v) <= 100.0}
        if dd:
            out[code] = dd
    if out:
        y0 = min(min(v) for v in out.values()); y1 = max(max(v) for v in out.values())
        print(f"  [{label}] OECD observed ({measure}): {len(out)} countries ({y0}-{y1})")
    return out


def _merge_obs(primary, secondary):
    """Merge `secondary` observations into `primary` WITHOUT overriding any year present in primary
    (primary source wins in overlapping country-years; secondary only fills gaps and earlier years)."""
    out = {k: dict(v) for k, v in primary.items()}
    for iso, dd in secondary.items():
        tgt = out.setdefault(iso, {})
        for y, v in dd.items():
            tgt.setdefault(y, v)
    return out


def _c_from_un(iso, un_d, gdp_at):
    """1961-2023 collected-fraction trajectory anchored to observed values: observations kept as reported,
    interior gaps linearly interpolated, and both extrapolated tails scaled by the GDP development ratio
    from the nearest observation (pre-first-obs back-cast AND post-last-obs forward-cast), symmetrically.
    Extrapolated years are ratcheted non-decreasing (durable infrastructure does not reverse when GDP
    falls); observed/interpolated years are left exactly as reported."""
    anchors = dict(un_d)
    obs = sorted(anchors)
    y0, y1 = obs[0], obs[-1]
    s0 = L(gdp_at(iso, y0)) or 0.0
    s1 = L(gdp_at(iso, y1)) or 0.0
    c = {}
    for y in YEARS:
        if y in anchors:
            c[y] = anchors[y]
        elif y0 < y < y1:
            lo = max(a for a in obs if a < y); hi = min(a for a in obs if a > y)
            c[y] = anchors[lo] + (anchors[hi] - anchors[lo]) * (y - lo) / (hi - lo)
        elif y < y0:                                    # pre-first-obs: GDP back-cast
            s = L(gdp_at(iso, y)) or 0.0
            c[y] = anchors[y0] * (s / s0) if s0 > 0 else anchors[y0]
        else:                                           # post-last-obs: GDP forward-cast
            s = L(gdp_at(iso, y)) or 0.0
            c[y] = min(anchors[y1] * (s / s1), 1.0) if s1 > 0 else anchors[y1]
    for y in range(y0 - 1, min(YEARS) - 1, -1):         # ratchet the pre-y0 back-cast (non-decreasing fwd)
        c[y] = min(c[y], c[y + 1])
    for y in range(y1 + 1, max(YEARS) + 1):             # ratchet the post-y1 forward-cast (non-decreasing)
        c[y] = max(c[y], c[y - 1])
    return c


def _c_from_anchor(iso, c_a, y_a, gdp_at):
    """Trajectory for a single-year anchor c_a at year y_a, treated as one observation: scaled by the GDP
    development ratio in BOTH directions from y_a (earlier years back-cast, later years forward-cast),
    with extrapolated years ratcheted non-decreasing. Where no GDP record exists (s_a=0) the anchor is
    held constant across all years."""
    s_a = L(gdp_at(iso, y_a)) or 0.0
    c = {}
    for y in YEARS:
        if y == y_a:
            c[y] = c_a
        elif s_a > 0:
            s = L(gdp_at(iso, y)) or 0.0
            c[y] = min(c_a * (s / s_a), 1.0)
        else:
            c[y] = c_a
    for y in range(min(y_a - 1, max(YEARS)), min(YEARS) - 1, -1):    # ratchet pre-y_a back-cast
        if y in c and (y + 1) in c:
            c[y] = min(c[y], c[y + 1])
    for y in range(max(y_a + 1, min(YEARS)), max(YEARS) + 1):        # ratchet post-y_a forward-cast
        if y in c and (y - 1) in c:
            c[y] = max(c[y], c[y - 1])
    return c


def _c_from_gdp(iso, m_r, gdp_at):
    """Region-calibrated GDP logistic: c(i,t) = clip(L(gdp) . m_region, 0, 1), ratcheted non-decreasing."""
    c = {}
    for y in YEARS:
        g = gdp_at(iso, y)
        c[y] = min(max(L(g) * m_r, 0.0), 1.0) if (g and y >= 1950) else 0.0
    ys = sorted(YEARS)
    for i in range(1, len(ys)):                     # non-decreasing forward (durable infrastructure)
        c[ys[i]] = max(c[ys[i]], c[ys[i - 1]])
    return c


def main():
    import country_converter as cc
    import logging; logging.getLogger("country_converter").setLevel(logging.CRITICAL)
    conv = cc.CountryConverter()

    # --- region crosswalk + model universe (population set) ---
    reg = pd.read_csv(REGION_XWALK)
    REGION = dict(zip(reg["iso3"], reg["ipcc_nex_region"]))
    pop = pd.read_csv(POP)
    universe = sorted(set(pop["ISO3"].unique()) & set(REGION))    # model countries with a region
    print(f"Universe: {len(universe)} model countries with an IPCC-Nex region")

    # --- Maddison GDP per capita by ISO3, year ---
    m = pd.read_excel(MADDISON, sheet_name="Full data")
    cl = {c.lower(): c for c in m.columns}
    ccol = cl.get("countrycode", cl.get("code", cl.get("iso3")))
    gcol = cl.get("gdppc", cl.get("cgdppc", cl.get("rgdpnapc")))
    ycol = cl.get("year", "year")
    m = m.rename(columns={ccol: "ISO3", gcol: "gdppc", ycol: "year"})[["ISO3", "year", "gdppc"]].dropna()
    print(f"  Maddison: {MADDISON.name} ({int(m['year'].min())}-{int(m['year'].max())}, "
          f"{m['ISO3'].nunique()} countries)")
    gdp = {k: dict(zip(sub["year"].astype(int), sub["gdppc"].astype(float))) for k, sub in m.groupby("ISO3")}

    def gdp_at(iso, yr):
        d = gdp.get(iso)
        if not d:
            return None
        return d[yr] if yr in d else d[min(sorted(d), key=lambda y: abs(y - yr))]

    # --- observed c: UNSD (primary) extended by OECD 'total public sewerage' (back to 1970) + anchors ---
    un = _un_sewered(conv)
    un = _merge_obs(un, _oecd_series(OECD_MEASURE["c"], "c"))     # UNSD wins in overlap; OECD fills earlier years
    anc = pd.read_csv(ANCHORS)
    anchor = {r["iso3"]: (float(r["c"]), int(r["year"])) for _, r in anc.iterrows()}
    print(f"  [c] hand-built anchors: {len(anchor)} countries")

    # --- regional calibration multiplier m_r = median(c_obs / L(gdp_obs)) over observed members ---
    #     (UNSD latest-year value + anchors), members with L>0.03 to avoid divide-by-tiny; and the
    #     regional-median fallback for no-GDP territories.
    obs_present = {}   # iso -> (c, year)  present-day observed value
    for iso, d in un.items():
        y = max(d); obs_present[iso] = (d[y], y)
    for iso, (c_a, y_a) in anchor.items():
        obs_present.setdefault(iso, (c_a, y_a))       # UNSD takes precedence
    ratios, reg_vals = {}, {}
    for iso, (c, y) in obs_present.items():
        r = REGION.get(iso)
        if not r:
            continue
        reg_vals.setdefault(r, []).append(c)
        l = L(gdp_at(iso, y))
        if l > 0.03:
            ratios.setdefault(r, []).append(c / l)
    m_reg = {r: float(np.median(v)) for r, v in ratios.items()}
    med_reg = {r: float(np.median(v)) for r, v in reg_vals.items()}
    glob_m = float(np.median([x for v in ratios.values() for x in v]))
    glob_med = float(np.median([x for v in reg_vals.values() for x in v]))
    print("  [c] regional calibration m_r:",
          {r: round(m_reg[r], 2) for r in sorted(m_reg)})

    # --- UNSD treatment observed, extended by OECD 'public total treatment'; regional fallback below ---
    un_treat = _un_sewered(conv, UN_TREAT_XLSX, label="f_treated")
    un_treat = _merge_obs(un_treat, _oecd_series(OECD_MEASURE["f_treated"], "f_treated"))

    # --- build c(i,t) for every universe country via the cascade ---
    src_count = {"observed(UNSD/OECD)": 0, "anchor": 0, "gdp": 0, "region_median": 0}
    c_all = {}
    for iso in universe:
        r = REGION.get(iso)
        if iso in un:
            c_all[iso] = _c_from_un(iso, un[iso], gdp_at); src_count["observed(UNSD/OECD)"] += 1
        elif iso in anchor:
            c_a, y_a = anchor[iso]
            c_all[iso] = _c_from_anchor(iso, c_a, y_a, gdp_at); src_count["anchor"] += 1
        elif gdp.get(iso):
            c_all[iso] = _c_from_gdp(iso, m_reg.get(r, glob_m), gdp_at); src_count["gdp"] += 1
        else:
            v = med_reg.get(r, glob_med)
            c_all[iso] = {y: v for y in YEARS}; src_count["region_median"] += 1
    print(f"  [c] sources: {src_count}")

    sewer_rows = [{"ISO3": iso, "year": y, "c": round(c_all[iso][y], 5)}
                  for iso in universe for y in YEARS]
    sw = pd.DataFrame(sewer_rows).sort_values(["ISO3", "year"]).reset_index(drop=True)

    # --- f_treated(i,t): UNSD treatment observed (clamped to c), else regional treated-of-collected . c ---
    #     tau_r = median over region's doubly-observed members of (treatment / collection) at the latest
    #     shared year; treatment is a strict subset of collection, so f_treated <= c(i,t) is enforced.
    HI_GDP = 25000.0                                     # Maddison 2011 int$ "high income" threshold
    tau_ratios, hi_ratios = {}, []
    for iso in un_treat:
        if iso in un:                                        # both collection and treatment observed
            yt = max(un_treat[iso]); c_at = c_all[iso].get(yt); g = gdp_at(iso, yt)
            if c_at and c_at > 0.02:
                t = min(un_treat[iso][yt] / c_at, 1.0)
                tau_ratios.setdefault(REGION.get(iso), []).append(t)
                if g and g >= HI_GDP:
                    hi_ratios.append(t)
    tau_reg = {r: float(np.median(v)) for r, v in tau_ratios.items()}
    glob_tau = float(np.median([x for v in tau_ratios.values() for x in v]))
    # High-income treated-of-collected norm: among high-income observed countries (GDP >= $25k) nearly all
    # collected wastewater is treated (mean ~0.96, median 1.0). Used to FLOOR tau for high-income countries
    # absent from the treatment series (e.g. Taiwan, Macao), so a low regional median does not under-treat
    # them.
    hi_tau = float(np.mean(hi_ratios)) if hi_ratios else 1.0
    print("  [f_treated] regional treated-of-collected tau_r:",
          {r: round(tau_reg[r], 2) for r in sorted(tau_reg)},
          f"| high-income floor tau={hi_tau:.2f} (n={len(hi_ratios)}, GDP>=${HI_GDP:,.0f})")

    # dump the region factors actually used, so Supplementary Table S2 matches the model exactly
    rf = [{"region": r, "n_c_obs": len(reg_vals.get(r, [])), "median_c": round(med_reg.get(r, float("nan")), 3),
           "m_r": round(m_reg.get(r, float("nan")), 2), "n_tau": len(tau_ratios.get(r, [])),
           "tau_r": round(tau_reg.get(r, float("nan")), 2)}
          for r in sorted(set(m_reg) | set(tau_reg) | set(med_reg))]
    pd.DataFrame(rf).to_csv(ROOT / "coefficients/region_factors.csv", index=False)
    print(f"  [f_treated] high-income floor value hi_tau={hi_tau:.2f}")

    tsrc = {"observed(UNSD/OECD)": 0, "tau_x_c": 0}
    treat_rows = []
    for iso in universe:
        if iso in un_treat:
            ft = _c_from_un(iso, un_treat[iso], gdp_at); tsrc["observed(UNSD/OECD)"] += 1
            ft = {y: min(ft[y], c_all[iso][y]) for y in YEARS}      # treatment is a subset of collection
        else:
            tau = tau_reg.get(REGION.get(iso), glob_tau)
            g_now = gdp_at(iso, 2019)
            if g_now and g_now >= HI_GDP:                           # high-income floor (see note above)
                tau = max(tau, hi_tau)
            ft = {y: min(tau * c_all[iso][y], c_all[iso][y]) for y in YEARS}; tsrc["tau_x_c"] += 1
        for y in YEARS:
            g = gdp_at(iso, y)
            r_n = R_LO + (R_HI - R_LO) * (L(g) if g else 0.0)   # treatment-level mix ramps with development
            treat_rows.append({"ISO3": iso, "year": y, "f_rem": round(ft[y] * r_n, 5),
                               "f_treated": round(ft[y], 5)})
    print(f"  [f_treated] sources: {tsrc}")
    out = pd.DataFrame(treat_rows).sort_values(["ISO3", "year"]).reset_index(drop=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    sw.to_csv(OUT_SEWER, index=False)
    print(f"wrote {OUT_SEWER.name} ({sw['ISO3'].nunique()} countries) and "
          f"{OUT.name} ({out['ISO3'].nunique()} countries) x {len(YEARS)} yr")

    # --- sanity ---
    ds = sw.groupby("year")["c"].mean()
    g = out.groupby("year")["f_rem"].mean()
    yy = [y for y in (1961, 1980, 2000, 2015, 2023) if y in ds.index]
    print("global-mean sewer c by year:", {y: round(float(ds.loc[y]), 3) for y in yy})
    print("global-mean f_rem by year: ", {y: round(float(g.loc[y]), 3) for y in yy})
    chk = out.merge(sw, on=["ISO3", "year"])
    over = (chk["f_treated"] - chk["c"]).max()
    print(f"  [check] max(f_treated - c) = {over:+.5f}  (must be <= 0: treated never exceeds collected)")
    assert over <= 1e-6, "f_treated exceeds collected fraction c for some country-year"


if __name__ == "__main__":
    main()
