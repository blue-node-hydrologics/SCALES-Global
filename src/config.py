"""Config + path loading. Mirrors the US model's src/config.py."""
from __future__ import annotations
from pathlib import Path
import yaml

_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path | None = None) -> dict:
    p = Path(path) if path else _ROOT / "config" / "config.yaml"
    cfg = yaml.safe_load(p.read_text())
    cfg["_root"] = str(_ROOT)
    return cfg


def get_path(cfg: dict, key: str) -> Path:
    """Resolve a path from cfg['paths'][key] relative to project root; mkdir for io dirs."""
    rel = cfg["paths"][key]
    p = Path(cfg["_root"]) / rel
    if key in ("raw", "processed", "outputs"):
        p.mkdir(parents=True, exist_ok=True)
    return p


def coef(cfg: dict, filename: str) -> Path:
    """Path to a coefficient file."""
    return get_path(cfg, "coefficients") / filename
