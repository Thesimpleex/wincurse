"""Selecting on one statistic and reporting another (Sharpe ratio selected, alpha reported)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from wincurse import (
    alpha_influence_functions,
    estimate_joint,
    tn_cdf,
    winner_inference,
    winner_inference_cross,
)

from .conftest import equicorrelated


def _joint_cov(k, rho_x=0.5, link=0.6):
    """Joint covariance of K selection and K reported statistics, positive definite."""
    ax = np.linalg.cholesky(equicorrelated(k, rho_x))
    load = np.block([[ax, np.zeros((k, k))], [link * ax, np.sqrt(1 - link**2) * np.eye(k)]])
    cov = load @ load.T
    return cov[:k, :k], cov[:k, k:].T, cov[k:, k:]


def test_same_statistic_reduces_to_winner_inference(rng):
    x = rng.standard_normal(7)
    sigma = equicorrelated(7, 0.4)
    exact = (2.5, 3.2)
    a = winner_inference(x, sigma, critical_values=exact)
    b = winner_inference_cross(x, x, sigma, sigma, sigma, critical_values=exact)
    assert b.index == a.index
    assert b.truncation == pytest.approx(a.truncation)
    assert b.conditional_estimate == pytest.approx(a.conditional_estimate, rel=1e-9)
    assert b.conditional.lower == pytest.approx(a.conditional.lower, rel=1e-9)
    assert b.hybrid.upper == pytest.approx(a.hybrid.upper, rel=1e-9)
    assert b.hybrid_estimate == pytest.approx(a.hybrid_estimate, rel=1e-9)


def test_independent_reported_statistic_needs_no_correction(rng):
    k = 6
    x = rng.standard_normal(k)
    y = rng.standard_normal(k)
    sxx = equicorrelated(k, 0.3)
    syy = np.eye(k) * 0.5
    res = winner_inference_cross(x, y, sxx, np.zeros((k, k)), syy, critical_values=(2.5, 3.1))
    assert res.truncation == (-np.inf, np.inf)
    assert res.conditional.lower == pytest.approx(res.naive.lower)
    assert res.conditional.upper == pytest.approx(res.naive.upper)
    assert res.conditional_estimate == pytest.approx(y[res.index])


def test_cross_pivot_is_uniform_and_coverage_is_exact(rng):
    k, n = 8, 1200
    sxx, sxy, syy = _joint_cov(k, link=0.7)
    full = np.block([[sxx, sxy], [sxy.T, syy]])
    chol = np.linalg.cholesky(full)
    mu_y = np.linspace(-0.2, 0.3, k)
    draws = rng.standard_normal((n, 2 * k)) @ chol.T
    pivots, covered = [], 0
    for i in range(n):
        x, y = draws[i, :k], draws[i, k:] + mu_y
        res = winner_inference_cross(x, y, sxx, sxy, syy, critical_values=(2.6, 3.3))
        lo, hi = res.truncation
        pivots.append(float(tn_cdf(y[res.index], mu_y[res.index], res.se, lo, hi)))
        covered += res.conditional.lower <= mu_y[res.index] <= res.conditional.upper
    assert stats.kstest(pivots, "uniform").pvalue > 0.01
    assert covered / n == pytest.approx(0.95, abs=3 * np.sqrt(0.95 * 0.05 / n))


def test_cross_inference_validation():
    s = np.eye(3)
    with pytest.raises(ValueError, match="K-sized"):
        winner_inference_cross(np.zeros(3), np.zeros(3), s, np.eye(2), s)
    with pytest.raises(ValueError, match="symmetric"):
        winner_inference_cross(np.zeros(3), np.zeros(3), s, s, np.triu(np.ones((3, 3))))
    with pytest.raises(ValueError, match="finite"):
        winner_inference_cross(np.zeros(3), np.zeros(3), s, np.full((3, 3), np.nan), s)


def test_alpha_matches_ols_and_its_influence_function_is_centred(rng):
    t = 600
    f = rng.standard_normal((t, 3)) * 0.01
    r = f @ np.array([[0.5, 1.0], [0.2, -0.3], [0.0, 0.4]]) + rng.standard_normal((t, 2)) * 0.01
    r = r + np.array([0.0003, -0.0002])
    alpha_hat, psi = alpha_influence_functions(r, f)
    w = np.column_stack([np.ones(t), f])
    expected = np.linalg.lstsq(w, r, rcond=None)[0][0]
    np.testing.assert_allclose(alpha_hat, expected, atol=1e-12)
    np.testing.assert_allclose(psi.mean(axis=0), 0.0, atol=1e-12)


def test_joint_covariance_matches_simulation(rng):
    t, k, reps, q = 400, 3, 2500, 12.0
    load = np.array([[0.6, 0.2], [0.3, 0.5], [0.9, -0.2]])
    sharpe_est, alpha_est = [], []
    for _ in range(reps):
        f = rng.standard_normal((t, 2))
        r = f @ load.T + rng.standard_normal((t, k)) + 0.1
        sharpe_est.append(np.sqrt(q) * r.mean(0) / r.std(0, ddof=1))
        w = np.column_stack([np.ones(t), f])
        alpha_est.append(q * np.linalg.lstsq(w, r, rcond=None)[0][0])
    sim = np.cov(np.hstack([sharpe_est, alpha_est]), rowvar=False)
    ests = []
    for method in ("iid", "hac"):
        scaled = []
        for _ in range(150):
            f = rng.standard_normal((t, 2))
            r = f @ load.T + rng.standard_normal((t, k)) + 0.1
            e = estimate_joint(r, f, q, cov_method=method)
            scaled.append(np.block([[e.cov_xx, e.cov_xy], [e.cov_xy.T, e.cov_yy]]))
        ests.append(np.mean(scaled, axis=0))
    for mean_cov in ests:
        np.testing.assert_allclose(np.diag(mean_cov), np.diag(sim), rtol=0.12)
        np.testing.assert_allclose(mean_cov[0, 3], sim[0, 3], rtol=0.25, atol=0.02)


def test_estimate_joint_validation(rng):
    r = rng.standard_normal((50, 3)) * 0.01
    f = rng.standard_normal((50, 2)) * 0.01
    with pytest.raises(ValueError, match="same number of rows"):
        estimate_joint(r, f[:40], 12.0)
    with pytest.raises(ValueError, match="cov_method"):
        estimate_joint(r, f, 12.0, cov_method="bootstrap")
    with pytest.raises(ValueError, match="periods_per_year"):
        estimate_joint(r, f, 0.0)
    est = estimate_joint(r, f[:, 0], 12.0, cov_method="iid")
    assert est.cov_xy.shape == (3, 3)
