"""Figure 6 -- Monte Carlo uncertainty of the SCALES-Global nitrogen budget.

(a) Global total anthropogenic N surplus, 1961-2023: deterministic central estimate with the
    Monte Carlo interquartile range and 95% interval.
(b) Half-width of the 95% interval of the 2023 gridded total anthropogenic surplus (1.96 SD,
    kg N/ha, log colour scale). Absolute rather than relative uncertainty is mapped because the
    CV of a near-zero residual is not informative and any surplus cutoff produces artefacts
    where the coarse deposition field sits near the cutoff. The printed median CV uses the
    MIN_SURPLUS cutoff, as in the text.
(c) Global budget components in 2023: central value, interquartile range (box), 95% interval
    (whiskers) and CV.

Inputs (from scripts/46 and 47):
    data/outputs/uncertainty_mc/uncertainty_N_timeseries.csv
    data/outputs/uncertainty_mc/global_draws_N_<year>.npz
    data/outputs/uncertainty_mc/grids/uncertainty_N_<year>_0p05.nc

Usage:
    python scripts/48_figure6_uncertainty.py --year 2023 --out <folder>
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

ROOT = Path(__file__).resolve().parents[1]
MC_DIR = ROOT / "data" / "outputs" / "uncertainty_mc"
MIN_SURPLUS = 5.0          # kg N/ha; cutoff for the printed median CV (as in the text)

# Panel c components: label -> (terms summed, sign: +1 input, -1 removal/loss)
COMPONENTS = [
    ("Manure",                      ["man_crop", "man_past"],              +1),
    ("Synthetic fertilizer",        ["fert", "fert_grass"],                +1),
    ("Atmospheric deposition",      ["dep_land"],                          +1),
    ("Biological fixation",         ["bnf", "bnf_grass"],                  +1),
    ("Human effluent",              ["hum_sew_land", "hum_sew_water"],     +1),
    ("Consumer food waste",         ["food_waste"],                        +1),
    ("Night soil",                  ["hum_ns"],                            +1),
    ("Biosolids",                   ["hum_bio"],                           +1),
    ("Fertilizer NH$_3$ loss",      ["fert_vol_full"],                     -1),
    ("Manure NH$_3$ loss",          ["man_vol", "man_vol_past"],           -1),
    ("Forage removal",              ["forage"],                            -1),
    ("Crop removal",                ["crop"],                              -1),
]
C_IN, C_OUT = "#2f6fa7", "#d9622b"


def panel_a(ax, ts):
    x = ts["year"].values
    p = "total_"
    ax.fill_between(x, ts[p + "p2.5_Tg"], ts[p + "p97.5_Tg"], color="#c6dbef", lw=0,
                    label="95% interval")
    ax.fill_between(x, ts[p + "p25_Tg"], ts[p + "p75_Tg"], color="#6baed6", lw=0,
                    label="Interquartile range")
    ax.plot(x, ts[p + "central_Tg"], color="white", lw=3.5, solid_capstyle="butt", zorder=3)
    ax.plot(x, ts[p + "central_Tg"], color="black", lw=1.5, ls=(0, (4, 2)), zorder=4)
    ax.plot([], [], color="black", lw=1.5, ls=(0, (4, 2)), label="Central estimate")
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(0, None)
    ax.set_xlabel("Year")
    ax.set_ylabel("Total anthropogenic N surplus (Tg N yr$^{-1}$)")
    ax.legend(loc="upper left", frameon=False)


def panel_c(ax, z):
    rows = []
    for lab, terms, sgn in COMPONENTS:
        d = sum(z[f"term_{k}"].astype("float64") for k in terms)
        c = sum(float(z[f"central_{k}"]) for k in terms)
        q = np.percentile(d, [2.5, 25, 75, 97.5])
        rows.append((lab, sgn, sgn * c, sgn * q, d.std() / d.mean()))
    y = np.arange(len(rows))[::-1]
    for yi, (lab, sgn, c, q, cv) in zip(y, rows):
        col = C_IN if sgn > 0 else C_OUT
        lo, q1, q3, hi = sorted(q)
        ax.plot([lo, hi], [yi, yi], color=col, lw=1.4, zorder=1)
        ax.add_patch(plt.Rectangle((q1, yi - 0.3), q3 - q1, 0.6, color=col, alpha=0.6, lw=0, zorder=2))
        ax.plot([c, c], [yi - 0.35, yi + 0.35], color="black", lw=2, zorder=3)
        ax.text(1.02, yi, f"{100 * cv:.0f}%", transform=ax.get_yaxis_transform(), va="center",
                fontsize=9, color="#444")
    ax.text(1.02, len(rows) - 0.4, "CV", transform=ax.get_yaxis_transform(), fontsize=9,
            fontweight="bold", color="#444")
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.axvline(0, color="#888", lw=0.8)
    lim = 1.08 * max(abs(r[3]).max() for r in rows)
    ax.set_xlim(-lim, lim)
    ax.set_xlabel("Global N flux in 2023 (Tg N yr$^{-1}$; inputs +, removals and losses −)")
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=C_IN, alpha=0.6),
                       plt.Rectangle((0, 0), 1, 1, color=C_OUT, alpha=0.6)],
              labels=["Input", "Removal or loss"], loc="upper left", frameon=False)


def panel_b(fig, ax, ds):
    lat = ds["lat"].values
    h = ds["surplus_full_ci95_kgha"].values.astype("float64")
    m = ds["surplus_full_mean_kgha"].values.astype("float64")
    s = ds["surplus_full_sd_kgha"].values.astype("float64")
    if lat[0] < lat[-1]:
        h, m, s, lat = h[::-1], m[::-1], s[::-1], lat[::-1]
    keep = (lat <= 84) & (lat >= -58)
    ext = [-180, 180, lat[keep].min() - 0.025, lat[keep].max() + 0.025]
    im = ax.imshow(np.where(h[keep] > 0, h[keep], np.nan), extent=ext, cmap="magma",
                   norm=LogNorm(vmin=0.1, vmax=100), interpolation="nearest", aspect="equal")
    ax.set_xticks([]); ax.set_yticks([])
    cb = fig.colorbar(im, ax=ax, extend="both", shrink=0.8, pad=0.01)
    cb.set_ticks([0.1, 1, 10, 100]); cb.set_ticklabels(["0.1", "1", "10", "100"])
    cb.ax.minorticks_off()
    cb.set_label("Uncertainty in N surplus\n(± kg N ha$^{-1}$ yr$^{-1}$, 95% interval)")
    k = np.isfinite(m) & np.isfinite(s) & (np.abs(m) >= MIN_SURPLUS)
    return np.nanmedian(100 * s[k] / np.abs(m[k]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2023)
    ap.add_argument("--mc", default=str(MC_DIR))
    ap.add_argument("--out", default=str(ROOT / "figures"))
    a = ap.parse_args()
    mc, out = Path(a.mc), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.grid": False,
                         "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"]})

    ts = pd.read_csv(mc / "uncertainty_N_timeseries.csv")
    z = np.load(mc / f"global_draws_N_{a.year}.npz")
    ds = xr.open_dataset(mc / "grids" / f"uncertainty_N_{a.year}_0p05.nc")

    fig = plt.figure(figsize=(14, 10.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.15], width_ratios=[1, 1.1], hspace=0.22, wspace=0.42)
    ax_a, ax_c, ax_b = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])
    panel_a(ax_a, ts)
    panel_c(ax_c, z)
    med = panel_b(fig, ax_b, ds)
    for ax, l in ((ax_a, "a"), (ax_c, "c"), (ax_b, "b")):
        ax.text(-0.02, 1.02, l, transform=ax.transAxes, fontsize=14, fontweight="bold", ha="right")
    for f in ("png", "pdf"):
        fig.savefig(out / f"Fig6_uncertainty_{a.year}.{f}", dpi=300, bbox_inches="tight")
    print(f"median gridded CV (|surplus| >= {MIN_SURPLUS:g}): {med:.1f}%")
    print(f"wrote {out / f'Fig6_uncertainty_{a.year}.png'} and .pdf")


if __name__ == "__main__":
    main()
