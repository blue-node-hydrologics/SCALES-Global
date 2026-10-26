"""Shared helpers: unit conversion, coefficient loading, source-file guard."""
from __future__ import annotations
from pathlib import Path
import pandas as pd


class MissingSourceFile(FileNotFoundError):
    """Raised when a coefficient/data file that must be downloaded is absent.

    Several coefficient tables (IPCC excretion, Ludemann crop coeffs, deposition,
    population) cannot be auto-downloaded in a network-restricted environment.
    They ship as schema stubs with status=NEEDS_SOURCE; this error points the user
    to docs/DOWNLOAD_CHECKLIST.md."""


def require_file(path: Path, what: str) -> Path:
    path = Path(path)
    if not path.exists():
        raise MissingSourceFile(
            f"Missing {what}: {path}\n"
            f"  -> see docs/DOWNLOAD_CHECKLIST.md to obtain it, then re-run."
        )
    return path


def load_coef(path: Path, what: str, required_cols: list[str] | None = None) -> pd.DataFrame:
    require_file(path, what)
    df = pd.read_csv(path)
    if "status" in df.columns:
        pending = df["status"].astype(str).str.upper().eq("NEEDS_SOURCE")
        if pending.any():
            raise MissingSourceFile(
                f"{what} ({path.name}) still has {int(pending.sum())} NEEDS_SOURCE rows — "
                f"populate it from the source in docs/DOWNLOAD_CHECKLIST.md."
            )
    if required_cols:
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"{path.name} missing columns {missing}")
    return df


def p2o5_to_p(series, factor: float = 0.4364):
    return series * factor


def tonnes_to_kg(series, factor: float = 1000.0):
    return series * factor
