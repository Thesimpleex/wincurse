from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from wincurse import (
    equicorrelated_critical_value,
    independent_critical_value,
    projection_critical_value,
)

from .conftest import equicorrelated


def test_single_trial_is_the_normal_quantile():
    z = stats.norm.isf(0.025)
    assert independent_critical_value(1, 0.05) == pytest.approx(z)
    assert equicorrelated_critical_value(1, 0.5, 0.05) == pytest.approx(z)
    assert projection_critical_value(np.array([[2.0]]), 0.05, n_sim=400_000, rng=1) == (
        pytest.approx(z, abs=0.01)
    )


def test_independent_formula_solves_its_defining_equation():
    for k, level in ((13, 0.05), (13, 0.005), (200, 0.05)):
        c = independent_critical_value(k, level)
        assert (2 * stats.norm.cdf(c) - 1) ** k == pytest.approx(1 - level, rel=1e-12)


def test_exact_values_for_the_jobstart_dimension():
    # Used as the oracle for the replication package (K = 13 independent policies).
    assert independent_critical_value(13, 0.05) == pytest.approx(2.883097, abs=1e-6)
    assert independent_critical_value(13, 0.005) == pytest.approx(3.549812, abs=1e-6)


@pytest.mark.parametrize(("k", "rho"), [(10, 0.3), (50, 0.7), (200, 0.9)])
def test_equicorrelated_formula_matches_simulation(k, rho):
    sigma = equicorrelated(k, rho) * 3.0
    exact = equicorrelated_critical_value(k, rho, 0.05)
    sim = projection_critical_value(sigma, 0.05, n_sim=400_000, rng=3)
    assert sim == pytest.approx(exact, abs=0.012)


def test_equicorrelated_interpolates_between_the_extremes():
    low = equicorrelated_critical_value(50, 0.0, 0.05)
    mid = equicorrelated_critical_value(50, 0.5, 0.05)
    high = equicorrelated_critical_value(50, 1 - 1e-8, 0.05)
    assert independent_critical_value(50, 0.05) == pytest.approx(low)
    assert high < mid < low
    assert high == pytest.approx(stats.norm.isf(0.025), abs=0.002)


def test_several_levels_share_draws_and_are_ordered():
    sigma = equicorrelated(20, 0.4)
    c = projection_critical_value(sigma, [0.05, 0.005], n_sim=50_000, rng=0)
    assert c.shape == (2,)
    assert c[1] > c[0]
    single = projection_critical_value(sigma, 0.05, n_sim=50_000, rng=0)
    assert single == c[0]


def test_projection_value_is_reproducible_and_scale_free():
    sigma = equicorrelated(15, 0.5)
    a = projection_critical_value(sigma, 0.05, n_sim=20_000, rng=4)
    b = projection_critical_value(sigma, 0.05, n_sim=20_000, rng=4)
    c = projection_critical_value(7.0 * sigma, 0.05, n_sim=20_000, rng=4)
    assert a == b
    # Rescaling changes rounding in the eigendecomposition of the degenerate correlation
    # matrix and hence the rotation of the draws, so only Monte Carlo accuracy is expected.
    assert a == pytest.approx(c, abs=0.04)


def test_validation():
    with pytest.raises(ValueError, match="level"):
        projection_critical_value(np.eye(2), 1.0)
    with pytest.raises(ValueError, match="level"):
        independent_critical_value(3, 0.0)
    with pytest.raises(ValueError, match="rho"):
        equicorrelated_critical_value(3, 1.0, 0.05)
