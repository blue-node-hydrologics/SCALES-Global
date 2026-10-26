"""Manure NH3 volatilization (FracGASM) — field-application / grazing-deposition loss.

For the soil-surface ("as-applied") budget, manure N is netted for the NH3 lost to the
atmosphere AFTER application to soils (and, in the land/NANI view, after deposition on
pasture), parallel to the synthetic-fertilizer FracGASF term in fert_volatilization.py.

IMPORTANT — what FAOSTAT already removes: the FAOSTAT EMN pools used as MAN inputs,
"Manure applied to soils" (72381) and "Manure left on pasture" (72380), are already net of
housing + storage (manure-management) losses. FracGASM here is therefore ONLY the downstream
NH3-N loss that occurs after the manure reaches the field/pasture surface — it is not the
management loss and does not double-count it.

Default FracGASM = 0.21 (IPCC 2019 V4 Ch.11; NH3-N from organic N applied to soils /
deposited by grazing; range 0.0-0.31). Per-scope values (applied vs pasture) live in
coefficients/manure_volatilization_global.csv.

NOTE on the applied-vs-pasture split: IPCC 2019 Table 11.3 defines a SINGLE FracGASM (0.21)
covering BOTH applied organic N (FON) AND grazing urine/dung (FPRP), i.e. IPCC does not provide a
separate grazing factor. We keep 0.21 for applied manure (IPCC) but use a lower value for the
grazing/pasture pool (default 0.10), following the empirical evidence for reduced NH3 loss from
grazing (rapid urine infiltration, less exposed surface) in Bouwman et al. 2002 (Glob. Biogeochem.
Cycles, doi:10.1029/2000GB001389) and the EMEP/EEA 2023 Guidebook (3.B). Set scope='pasture' both
values equal to 0.21 to recover the strict-IPCC (no-split) behaviour.
"""
from __future__ import annotations
import pandas as pd
from .config import coef

DEFAULT_FRACGASM = 0.21   # IPCC 2019 default NH3-N loss fraction for applied/deposited manure


def fracgasm(cfg, scope: str = "applied") -> float:
    """Post-application NH3-N loss fraction for a manure pool ('applied' or 'pasture')."""
    try:
        t = pd.read_csv(coef(cfg, "manure_volatilization_global.csv"))
        row = t.loc[t["scope"] == scope, "fracgasm"]
        return float(row.iloc[0]) if len(row) else DEFAULT_FRACGASM
    except Exception:  # noqa: BLE001
        return DEFAULT_FRACGASM


def volatilization_loss(cfg, manure_nat: pd.DataFrame, man_col: str | None = None,
                        scope: str = "applied") -> pd.DataFrame:
    """Given a national manure table [ISO3, <kg N>], return [ISO3, man_vol_kg] =
    manure(kg N) x FracGASM(scope). The manure value column is auto-detected if not given."""
    if man_col is None or man_col not in manure_nat.columns:
        man_col = [c for c in manure_nat.columns if c != "ISO3"][0]
    fg = fracgasm(cfg, scope)
    m = manure_nat[["ISO3", man_col]].copy()
    m["man_vol_kg"] = m[man_col].astype(float) * fg
    return m[["ISO3", "man_vol_kg"]]
