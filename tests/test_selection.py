from __future__ import annotations

import numpy as np
import pytest
from scipy import optimize, stats

from wincurse import threshold_inference, top_m_inference, winner_inference
from wincurse.selection import _union_cdf, top_m_windows

from .conftest import equicorrelated


def test_top_one_equals_winner_inference(rng):
    x = rng.standard_normal(9) * 0.8
    sigma = equicorrelated(9, 0.5)
    res = winner_inference(x, sigma, critical_values=(2.5, 3.2))
    (top,) = top_m_inference(x, sigma, 1)
    assert top.index == res.index
    assert top.conditional_estimate == pytest.approx(res.conditional_estimate, abs=1e-7)
    assert top.conditional.lower == pytest.approx(res.conditional.lower, abs=1e-7)
    assert top.conditional.upper == pytest.approx(res.conditional.upper, abs=1e-7)
    assert len(top.windows) == 1
    assert top.windows[0] == pytest.approx(res.truncation)


def test_window_union_for_independent_trials_is_one_half_line():
    x = np.array([2.0, 1.5, 1.0, 0.2, -0.3])
    # Independent trials: trial k is in the top m iff X_k exceeds the m-th largest of the others.
    windows = top_m_windows(x, np.eye(5), 2, 3)
    assert windows == [(0.2, np.inf)]
    windows = top_m_windows(x, np.eye(5), 0, 2)
    assert windows == [(1.0, np.inf)]


def test_correlated_trials_can_give_disconnected_windows():
    sigma = np.array([[1.0, 0.0, 1.6], [0.0, 1.0, -0.2], [1.6, -0.2, 4.0]])
    x = np.array([0.52, 1.23, 0.5])
    (first, second) = top_m_windows(x, sigma, 0, 2)
    assert first[0] == -np.inf
    assert second[1] == np.inf
    assert first[1] < second[0]
    assert first[0] <= x[0] <= first[1]
    # Inside the gap both other trials beat trial 0; just outside it at most one does.
    z = x - sigma[:, 0] * x[0]
    for value, expected_inside in ((0.5 * (first[1] + second[0]), False), (first[1] - 0.01, True)):
        beats = np.sum(z[1:] + sigma[1:, 0] * value > value)
        assert (beats <= 1) == expected_inside


def test_union_cdf_matches_scipy_for_two_windows():
    windows = [(-3.0, -1.0), (0.5, 2.0)]
    mu, sd, obs = 0.2, 1.3, 0.9
    masses = [stats.norm.cdf(b, mu, sd) - stats.norm.cdf(a, mu, sd) for a, b in windows]
    part = masses[0] + stats.norm.cdf(obs, mu, sd) - stats.norm.cdf(0.5, mu, sd)
    assert _union_cdf(obs, mu, sd, windows) == pytest.approx(part / sum(masses), rel=1e-12)
    assert _union_cdf(-5.0, mu, sd, windows) == 0.0
    assert _union_cdf(10.0, mu, sd, windows) == pytest.approx(1.0)


def test_top_m_pivot_is_uniform_for_a_fixed_trial(rng):
    k, m, fixed, n = 8, 3, 5, 6000
    sigma = equicorrelated(k, 0.4)
    mu = np.linspace(0.0, 0.5, k)
    draws = mu + rng.standard_normal((n, k)) @ np.linalg.cholesky(sigma).T
    pivots = []
    for x in draws:
        if np.sum(x > x[fixed]) <= m - 1:
            windows = top_m_windows(x, sigma, fixed, m)
            pivots.append(_union_cdf(x[fixed], mu[fixed], 1.0, windows))
    assert len(pivots) > 1000
    assert stats.kstest(pivots, "uniform").pvalue > 0.01


def test_top_m_intervals_cover_the_reported_trials(rng):
    k, m, n = 8, 3, 500
    sigma = equicorrelated(k, 0.4)
    mu = np.linspace(0.0, 0.5, k)
    draws = mu + rng.standard_normal((n, k)) @ np.linalg.cholesky(sigma).T
    hits = total = below = 0
    for x in draws:
        for res in top_m_inference(x, sigma, m):
            total += 1
            hits += res.conditional.lower <= mu[res.index] <= res.conditional.upper
            below += res.conditional_estimate > mu[res.index]
    assert hits / total == pytest.approx(0.95, abs=3 * np.sqrt(0.95 * 0.05 / total))
    assert below / total == pytest.approx(0.5, abs=3 * 0.5 / np.sqrt(total))


def test_threshold_inference_matches_scipy_truncnorm():
    x = np.array([2.6, 1.1, 3.4, -0.5])
    sigma = np.diag([1.0, 1.0, 1.0, 1.0]) * np.array([0.5, 1.0, 2.0, 1.0]) ** 2
    out = threshold_inference(x, sigma, 2.0)
    assert [r.index for r in out] == [0]  # t = 5.2; trial 2 has t = 1.7; trial 1 has 1.1
    r = out[0]
    sd, cut = 0.5, 2.0 * 0.5

    def quantile(p):
        return optimize.brentq(
            lambda mu: stats.truncnorm.cdf(2.6, (cut - mu) / sd, np.inf, loc=mu, scale=sd) - p,
            -5,
            8,
        )

    assert r.conditional_estimate == pytest.approx(quantile(0.5), abs=1e-7)
    assert r.conditional.lower == pytest.approx(quantile(0.975), abs=1e-7)
    assert r.conditional.upper == pytest.approx(quantile(0.025), abs=1e-7)
    assert r.windows == ((1.0, np.inf),)


def test_threshold_selection_bias_and_conditional_coverage(rng):
    mu, sd, c, n = 0.8, 1.0, 2.0, 40_000
    x = mu + sd * rng.standard_normal(n)
    passed = np.flatnonzero(x / sd > c)
    naive_bias = np.mean(x[passed] - mu)
    assert naive_bias > 0.5
    sample = rng.choice(passed, 1500, replace=False)
    covered, below = 0, 0
    for i in sample:
        (res,) = threshold_inference(np.array([x[i]]), np.array([[sd**2]]), c)
        covered += res.conditional.lower <= mu <= res.conditional.upper
        below += res.conditional_estimate > mu
    assert covered / len(sample) == pytest.approx(0.95, abs=3 * np.sqrt(0.95 * 0.05 / 1500))
    assert below / len(sample) == pytest.approx(0.5, abs=3 * 0.5 / np.sqrt(1500))


def test_threshold_nothing_passes_and_validation():
    assert threshold_inference(np.array([0.5, 1.0]), np.eye(2), 2.0) == []
    with pytest.raises(ValueError, match="m must"):
        top_m_inference(np.array([1.0, 2.0]), np.eye(2), 3)
    with pytest.raises(ValueError, match="alpha"):
        threshold_inference(np.array([3.0]), np.eye(1), 2.0, alpha=0.0)
