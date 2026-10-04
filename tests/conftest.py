from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(20241003)


def equicorrelated(k: int, rho: float, variance: float = 1.0) -> np.ndarray:
    """Covariance matrix with common correlation ``rho``."""
    return variance * (rho * np.ones((k, k)) + (1.0 - rho) * np.eye(k))


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
