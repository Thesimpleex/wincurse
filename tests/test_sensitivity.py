from __future__ import annotations

import numpy as np
import pytest

from wincurse import (
    augmented_covariance,
    equicorrelated_critical_value,
    independent_critical_value,
    nearest_neighbour_correlation,
    projection_critical_value,
    unlogged_projection,
    unlogged_scenario,
    winner_inference,
)

from .conftest import equicorrelated


def test_augmented_covariance_structure():
    sigma = equicorrelated(4, 0.5) * 2.0
    full = augmented_covariance(sigma, 3, 0.8)
    assert full.shape == (7, 7)
    np.testing.assert_allclose(full[:4, :4], sigma)
    np.testing.assert_allclose(full, full.T)
    assert np.linalg.eigvalsh(full)[0] > -1e-12
    np.testing.assert_allclose(np.diag(full)[4:], 2.0)
    # Each hidden trial correlates 0.8 with the logged trial it was built from and
    # 0.8 * 0.5 with the others.
    corr = full[:4, 4:] / 2.0
    for m in range(3):
        column = np.sort(corr[:, m])
        assert column[-1] == pytest.approx(0.8)
        np.testing.assert_allclose(column[:-1], 0.4)
    np.testing.assert_allclose(augmented_covariance(sigma, 0, 0.5), sigma)
    np.testing.assert_allclose(augmented_covariance(sigma, 3, 0.8), full)


def test_augmented_covariance_is_psd_for_any_logged_structure(rng):
    loadings = rng.standard_normal((30, 2))
    sigma = loadings @ loadings.T + 0.01 * np.eye(30)
    full = augmented_covariance(sigma, 45, 0.9)
    assert np.linalg.eigvalsh(full)[0] > -1e-9 * np.diag(full).max()


def test_hidden_clones_of_independent_trials_leave_the_projection_value_unchanged():
    k = 6
    full = augmented_covariance(np.eye(k), k, 1 - 1e-9)
    c_clones = projection_critical_value(full, 0.05, n_sim=300_000, rng=1)
    assert c_clones == pytest.approx(independent_critical_value(k, 0.05), abs=0.015)
    independent = augmented_covariance(np.eye(k), k, 0.0)
    np.testing.assert_allclose(independent, np.eye(2 * k))
    c_new = projection_critical_value(independent, 0.05, n_sim=300_000, rng=1)
    assert c_new == pytest.approx(independent_critical_value(2 * k, 0.05), abs=0.015)


def test_nearest_neighbour_correlation():
    sigma = np.array([[1.0, 0.9, 0.0], [0.9, 1.0, 0.1], [0.0, 0.1, 1.0]])
    assert nearest_neighbour_correlation(sigma) == pytest.approx((0.9 + 0.9 + 0.1) / 3)
    assert nearest_neighbour_correlation(np.eye(3)) == 0.0
    assert nearest_neighbour_correlation(np.array([[2.0]])) == 0.0


def test_hidden_correlation_is_validated():
    block = np.array([[1.0, -0.9], [-0.9, 1.0]])
    with pytest.raises(ValueError, match="correlation"):
        augmented_covariance(block, 1, 1.0)
    with pytest.raises(ValueError, match="correlation"):
        augmented_covariance(block, 1, -0.1)
    with pytest.raises(ValueError, match="n_hidden"):
        augmented_covariance(block, -1, 0.2)


def test_projection_sensitivity_matches_a_direct_computation():
    k, h, rho = 12, 20, 0.4
    sigma = equicorrelated(k, rho) * 0.04
    x = np.linspace(-0.2, 0.9, k)
    out = unlogged_projection(x, sigma, [0, h], 0.7, 0.05, n_sim=300_000, rng=2)
    direct = projection_critical_value(
        augmented_covariance(sigma, h, 0.7), 0.05, n_sim=300_000, rng=2
    )
    assert out[1].critical_value == pytest.approx(direct, abs=0.02)
    assert out[0].critical_value == pytest.approx(
        equicorrelated_critical_value(k, rho, 0.05), abs=0.015
    )
    assert out[0].width_ratio == 1.0
    assert out[1].width_ratio == pytest.approx(out[1].critical_value / out[0].critical_value)
    assert out[1].upper - out[1].lower > out[0].upper - out[0].lower
    se = np.sqrt(sigma[0, 0])
    assert out[0].upper - x.max() == pytest.approx(out[0].critical_value * se)


def test_projection_interval_widens_with_more_hidden_trials(rng):
    sigma = equicorrelated(10, 0.3)
    x = rng.standard_normal(10)
    out = unlogged_projection(x, sigma, [0, 5, 20, 100], 0.3, n_sim=30_000, rng=0)
    widths = [r.upper - r.lower for r in out]
    assert widths == sorted(widths)


def test_scenario_is_reproducible_and_reports_acceptance():
    sigma = equicorrelated(8, 0.3)
    x = np.array([0.2, 1.4, 0.3, 0.1, 0.5, -0.2, 0.9, 0.0])
    a = unlogged_scenario(x, sigma, 5, 0.3, n_draws=150, n_sim=10_000, rng=1)
    b = unlogged_scenario(x, sigma, 5, 0.3, n_draws=150, n_sim=10_000, rng=1)
    assert a == b
    assert 0 < a.acceptance <= 1
    assert a.n_accepted == 150
    base = winner_inference(x, sigma, critical_values=(2.5, 3.1))
    # Hidden competitors only add truncation constraints, so they can only lower the
    # median-unbiased estimate (the hybrid one up to its critical value, which rises).
    assert a.conditional_estimate[1] <= base.conditional_estimate + 1e-9
    assert a.conditional_estimate[2] <= base.conditional_estimate + 1e-9
    assert a.hybrid_estimate[0] <= a.hybrid_estimate[1] <= a.hybrid_estimate[2]


def test_scenario_rejects_impossible_assumptions():
    sigma = equicorrelated(3, 0.2)
    x = np.array([0.1, 0.0, 0.05])
    with pytest.raises(ValueError, match="n_hidden"):
        unlogged_scenario(x, sigma, 0, 0.2)
    far = np.array([-5.0, -6.0, -5.5])
    with pytest.raises(ValueError, match="essentially never"):
        unlogged_scenario(far, sigma, 400, 0.0, true_value=3.0, n_draws=50, n_sim=2000)
