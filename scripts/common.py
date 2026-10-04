"""Shared helpers for the experiment scripts: results file, environment record, paths."""

from __future__ import annotations

import datetime as dt
import json
import platform
import sys
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "docs" / "results.json"
FIGURES = ROOT / "docs" / "figures"
SCREENSHOTS = ROOT / "docs" / "screenshots"
CACHE = ROOT / "data" / "cache"


def _clean(obj: Any) -> Any:
    """Convert numpy scalars and arrays to plain JSON types (non-finite values to null)."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _clean(obj.tolist())
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, float | np.floating):
        return float(obj) if np.isfinite(obj) else None
    return obj


def load_results() -> dict[str, Any]:
    return json.loads(RESULTS.read_text()) if RESULTS.exists() else {}


def update_results(section: str, payload: dict[str, Any]) -> None:
    """Replace one top-level section of ``docs/results.json`` and keep the others."""
    data = load_results()
    data[section] = _clean(payload)
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def environment() -> dict[str, str]:
    """Package versions and date of the run, recorded next to every result."""
    pkgs = ("numpy", "scipy", "pandas", "wincurse")
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        **{p: version(p) for p in pkgs},
        "run_date": dt.date.today().isoformat(),
    }
