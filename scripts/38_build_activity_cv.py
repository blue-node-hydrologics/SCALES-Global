#!/usr/bin/env python3
"""Build the per-(activity group, ISO3, year) ACTIVITY (reporting) CV from FAOSTAT provenance flags.

Every FAOSTAT value carries a status flag (A official, X from an international organization, E estimated,
I imputed, M missing) that grades how the value was obtained. We turn that per-value provenance into a
country- and year-specific activity CV so the reporting channel of the uncertainty analysis varies
spatially and temporally instead of using one global CV per component (src/uncertainty.ACT_CV).

Method (magnitude-preserving; see docs/Activity_CV_flag_proposal.md):
  * Each flag maps to a multiplier of a per-group baseline CV:  A 1.0, X 1.2, E 1.6, I 2.4, M 3.0
    (B = time-series break -> treated as official, 1.0; anything else -> estimated, 1.6).
  * For each (group, ISO3, year) we take the VALUE-weighted mean multiplier over that group's FAOSTAT
    items (so a country's reliability reflects its large flows, not its trace crops).
  * The group baseline is set so the VALUE-weighted GLOBAL mean effective CV equals the group's current
    global activity CV (uncertainty.ACT_CV) -- i.e. the flags only REDISTRIBUTE the reporting
    uncertainty across countries/years; the global magnitude (and every headline number) is unchanged.

Provenance source per activity group (the element that actually supplies that term's activity data):
    fert    RFN  'Agricultural Use'  N nutrient           (tonnes N)
    crop    QCL  'Production'        all items            (tonnes)
    bnf     QCL  'Production'        all items            (tonnes)   [same statistical basis as crop]
    man     EMN  'Stocks'            all animals          (head)     [head count, NOT the derived N series]
    forage  RL   'Area'             permanent meadows     (1000 ha)
    pop     FBSH(<=2013)/FBS(>=2014) 'Protein supply quantity (g/capita/day)' Grand Total  [drives excretion]
    dep     -- no activity channel (modelled field) --

Output:
    coefficients/activity_cv_by_flag.csv        long: group, iso3, year, activity_cv
    coefficients/activity_cv_summary.csv        per group: baseline + effective CV for each flag (paper table)

Usage:  python scripts/38_build_activity_cv.py            # 1961-2023, all groups
        python scripts/38_build_activity_cv.py --summary  # also print the paper table
"""
import sys, argparse
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import os; os.chdir(ROOT)
from src import config as cfgmod, faostat_prep as prep, uncertainty as U

PQ = ROOT / "data/raw/faostat/_parquet_cache"
YEAR0, YEAR1 = 1961, 2023

# flag -> reliability multiplier (relative to the official baseline)
MULT = {"A": 1.0, "X": 1.2, "E": 1.6, "I": 2.4, "M": 3.0, "B": 1.0}
DEFAULT_MULT = 1.6   # any unrecognised/blank flag -> treat as estimated

# activity group -> (domain, element substring, item substring or None, FBS-split flag)
GROUP_SRC = {
    "fert":   ("RFN", "Agricultural Use", "Nitrogen", False),
    "crop":   ("QCL", "Production",        None,       False),
    "bnf":    ("QCL", "Production",        None,       False),
    "man":    ("EMN", "Stocks",            None,       False),
    "forage": ("RL",  "Area",             "permanent meadows", False),
    "pop":    ("FBS", "Protein supply quantity (g/capita/day)", "Grand Total", True),
}


def _load(domain: str) -> pd.DataFrame:
    f = PQ / f"{domain}.parquet"
    if not f.exists():
        raise SystemExit(f"missing parquet cache: {f} (run the pipeline once to build it)")
    return pd.read_parquet(f)


def _mult(flag: pd.Series) -> np.ndarray:
    return flag.fillna("").map(lambda x: MULT.get(str(x).strip(), DEFAULT_MULT)).to_numpy(float)


def group_frame(cfg, group: str) -> pd.DataFrame:
    """Return per-(ISO3, Year) value-weighted mean flag-multiplier for one activity group."""
    domain, elem, item, fbs_split = GROUP_SRC[group]
    frames = []
    domains = ["FBSH", "FBS"] if fbs_split else [domain]
    for dom in domains:
        d = _load(dom)
        d = d[d["Element"].astype(str).str.contains(elem, case=False, na=False, regex=False)]
        if item:
            d = d[d["Item"].astype(str).str.contains(item, case=False, na=False, regex=False)]
        d = d[(d["Year"].astype(int) >= YEAR0) & (d["Year"].astype(int) <= YEAR1)]
        if fbs_split:                                  # FBSH is the pre-2014 series; FBS the >=2010 one
            d = d[d["Year"].astype(int) <= 2013] if dom == "FBSH" else d[d["Year"].astype(int) >= 2014]
        if len(d):
            frames.append(prep.to_iso3(d, cfg))
    d = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not len(d):
        return pd.DataFrame(columns=["group", "iso3", "year", "wmult", "w"])
    d["m"] = _mult(d["Flag"])
    d["w"] = pd.to_numeric(d["Value"], errors="coerce").abs().fillna(0.0)
    d["Year"] = d["Year"].astype(int)
    # value-weighted mean multiplier per country-year; fall back to unweighted where all weights 0
    g = d.groupby(["ISO3", "Year"])
    wsum = g["w"].transform("sum")
    d["w_eff"] = np.where(wsum.to_numpy() > 0, d["w"], 1.0)
    agg = (d.assign(wm=d["m"] * d["w_eff"])
             .groupby(["ISO3", "Year"])
             .agg(wm=("wm", "sum"), w=("w_eff", "sum")))
    agg["wmult"] = agg["wm"] / agg["w"]
    out = agg.reset_index().rename(columns={"ISO3": "iso3", "Year": "year"})
    out["group"] = group
    return out[["group", "iso3", "year", "wmult", "w"]]


def build(cfg, summary=False) -> pd.DataFrame:
    rows, summ = [], []
    for group in GROUP_SRC:
        gf = group_frame(cfg, group)
        if not len(gf):
            print(f"  [{group}] no data"); continue
        # global value-weighted mean multiplier -> baseline so weighted-mean eff CV == current ACT_CV
        Mbar = float(np.average(gf["wmult"], weights=gf["w"])) if gf["w"].sum() > 0 else float(gf["wmult"].mean())
        base = U.ACT_CV.get(group, 0.0) / Mbar if Mbar > 0 else U.ACT_CV.get(group, 0.0)
        gf = gf.copy(); gf["activity_cv"] = (base * gf["wmult"]).round(4)
        rows.append(gf[["group", "iso3", "year", "activity_cv"]])
        print(f"  [{group:6s}] n={len(gf):>6}  Mbar={Mbar:.3f}  baseline={base:.4f}  "
              f"target ACT_CV={U.ACT_CV.get(group,0.0):.3f}  "
              f"eff CV range {gf['activity_cv'].min():.3f}-{gf['activity_cv'].max():.3f}")
        for fl in ["A", "X", "E", "I", "M"]:
            summ.append({"group": group, "flag": fl, "baseline_official": round(base, 4),
                         "eff_cv": round(base * MULT[fl], 4)})
    out = pd.concat(rows, ignore_index=True)
    dest = ROOT / "coefficients/activity_cv_by_flag.csv"
    out.to_csv(dest, index=False)
    print(f"wrote {dest}  ({len(out):,} rows; {out['iso3'].nunique()} countries x {out['group'].nunique()} groups)")
    sdf = pd.DataFrame(summ)
    sdf.to_csv(ROOT / "coefficients/activity_cv_summary.csv", index=False)
    if summary:
        piv = sdf.pivot(index="group", columns="flag", values="eff_cv")[["A", "X", "E", "I", "M"]]
        print("\nEffective activity CV by group x flag (paper table):")
        print(piv.to_string())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", action="store_true", help="also print the group x flag CV table")
    args = ap.parse_args()
    cfg = cfgmod.load_config()
    build(cfg, summary=args.summary)


if __name__ == "__main__":
    main()
