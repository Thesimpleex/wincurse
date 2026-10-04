from __future__ import annotations

from pathlib import Path

import mpmath as mp
import numpy as np
import pytest
from scipy import optimize, stats

from wincurse import (
    RootFindingError,
    WincurseWarning,
    forecast_interval,
    independent_critical_value,
    selection_bounds,
    simulate_known_sigma,
    tn_cdf,
    winner_inference,
)
from wincurse.simulation import replication_metrics
from wincurse.winners import _relative_bounds

from .conftest import equicorrelated

mp.mp.dps = 50


def _draw(rng, mu, sigma, n):
    chol = np.linalg.cholesky(sigma)
    return mu + rng.standard_normal((n, len(mu))) @ chol.T


def test_single_trial_reduces_to_the_textbook_interval():
    x, var = np.array([1.3]), np.array([[0.04]])
    z = stats.norm.isf(0.025)
    exact = (z, stats.norm.isf(0.0025))
    res = winner_inference(x, var, 0.05, critical_values=exact)
    for interval in (res.naive, res.conditional, res.hybrid):
        assert interval.lower == pytest.approx(1.3 - z * 0.2, abs=1e-9)
        assert interval.upper == pytest.approx(1.3 + z * 0.2, abs=1e-9)
    assert res.conditional_estimate == pytest.approx(1.3, abs=1e-9)
    assert res.hybrid_estimate == pytest.approx(1.3, abs=1e-9)
    assert res.truncation == (-np.inf, np.inf)
    assert res.projection.upper == pytest.approx(1.3 + z * 0.2)
    assert res.bonferroni.lower == res.naive.lower
    simulated = winner_inference(x, var, 0.05, n_sim=400_000, rng=1)
    assert simulated.projection.upper - 1.3 == pytest.approx(z * 0.2, abs=0.004)


def test_independent_arms_match_scipy_truncnorm():
    # With independent arms of equal variance the truncation set is [X_(2), inf).
    x = np.array([0.4, 1.1, 0.9, 0.2, 1.4])
    res = winner_inference(x, np.eye(5) * 0.25, 0.05, critical_values=(2.5, 3.2))
    assert res.truncation == pytest.approx((1.1, np.inf))
    sd = 0.5

    def quantile(p):
        def f(mu):
            return stats.truncnorm.cdf(1.4, (1.1 - mu) / sd, np.inf, loc=mu, scale=sd) - p

        return optimize.brentq(f, -10, 10, xtol=1e-13)

    assert res.conditional_estimate == pytest.approx(quantile(0.5), abs=1e-8)
    assert res.conditional.lower == pytest.approx(quantile(0.975), abs=1e-8)
    assert res.conditional.upper == pytest.approx(quantile(0.025), abs=1e-8)


def test_truncation_bounds_follow_the_selection_event(rng):
    sigma = equicorrelated(6, 0.4)
    x = _draw(rng, np.zeros(6), sigma, 1)[0]
    k = int(np.argmax(x))
    lo, hi = selection_bounds(x, sigma, k)
    assert lo <= x[k] <= hi
    # Moving X_k along the direction Sigma[:, k] keeps Z fixed; selection must hold
    # exactly inside [L, U] and fail just outside.
    direction = sigma[:, k] / sigma[k, k]

    def selected(value):
        return int(np.argmax(x + direction * (value - x[k]))) == k

    assert selected(lo + 1e-6 * abs(lo))
    assert not selected(lo - 1e-3 * abs(lo) - 1e-6)
    if np.isfinite(hi):
        assert selected(hi - 1e-6)
        assert not selected(hi + 1e-3)


def test_bounds_with_a_larger_covariance_give_upper_limits():
    # The winner (trial 0) has a small variance; trial 1 covaries with it more than the
    # winner varies itself (Sigma_10 = 0.4 > Sigma_00 = 0.25), which bounds X_0 from above.
    sigma = np.array([[0.25, 0.4, 0.0], [0.4, 1.0, 0.0], [0.0, 0.0, 1.0]])
    x = np.array([0.5, 0.2, -0.3])
    _, hi = selection_bounds(x, sigma, 0)
    assert np.isfinite(hi)
    z = x - sigma[:, 0] / sigma[0, 0] * x[0]
    assert hi == pytest.approx(sigma[0, 0] * z[1] / (sigma[0, 0] - sigma[1, 0]))


def test_pivot_is_uniform_given_selection(rng):
    k_trials, rho, n = 12, 0.5, 4000
    sigma = equicorrelated(k_trials, rho)
    x = _draw(rng, np.zeros(k_trials), sigma, n)
    ks = np.argmax(x, axis=1)
    lo, hi = _relative_bounds(x, sigma, ks, sigma[ks, ks])
    xk = x[np.arange(n), ks]
    u = tn_cdf(xk, 0.0, 1.0, xk + lo, xk + hi)
    assert stats.kstest(u, "uniform").pvalue > 0.01


def test_pivot_is_uniform_for_t_statistic_selection_with_unequal_variances(rng):
    sd = np.linspace(0.5, 2.0, 8)
    corr = equicorrelated(8, 0.3)
    sigma = corr * np.outer(sd, sd)
    mu = np.linspace(-0.2, 0.4, 8)
    n = 4000
    x = _draw(rng, mu, sigma, n)
    t = x / sd
    ks = np.argmax(t, axis=1)
    lo, hi = _relative_bounds(t, corr, ks, corr[ks, ks])
    tk = t[np.arange(n), ks]
    u = tn_cdf(tk, mu[ks] / sd[ks], 1.0, tk + lo, tk + hi)
    assert stats.kstest(u, "uniform").pvalue > 0.01


@pytest.mark.parametrize(
    ("k", "rho", "spread"),
    [(10, 0.3, 0.0), (50, 0.7, 0.0), (200, 0.5, 0.0), (30, 0.5, 1.0)],
)
def test_conditional_and_hybrid_coverage_and_bias(k, rho, spread):
    sigma = equicorrelated(k, rho)
    mu = np.linspace(0.0, spread, k)
    n_rep, alpha, beta = 4000, 0.05, 0.005
    res = simulate_known_sigma(mu, sigma, n_rep, alpha, beta, n_sim=40_000, rng=7)
    mc = 3 * np.sqrt(0.95 * 0.05 / n_rep)
    assert res.n_failed == 0
    assert res.conditional_coverage == pytest.approx(0.95, abs=mc)
    assert res.hybrid_coverage >= 0.95 - mc
    assert res.hybrid_coverage <= 0.95 / (1 - beta) + mc
    mc_bias = 3 * 0.5 / np.sqrt(n_rep)
    assert abs(res.conditional_prob_bias) < mc_bias
    assert abs(res.hybrid_prob_bias) < beta / 2 + mc_bias
    assert res.projection_coverage >= 0.95 - mc


def test_naive_inference_fails_where_the_corrections_hold():
    sigma = equicorrelated(100, 0.7)
    res = simulate_known_sigma(np.zeros(100), sigma, 3000, n_sim=40_000, rng=3)
    assert res.naive_coverage < 0.8
    assert res.naive_prob_bias > 0.3
    assert res.naive_mean_bias > 0.5
    assert res.conditional_coverage == pytest.approx(0.95, abs=0.015)


def test_perfect_correlation_removes_the_winners_curse():
    sigma = equicorrelated(20, 0.999999)
    res = simulate_known_sigma(np.zeros(20), sigma, 3000, n_sim=40_000, rng=5)
    assert res.naive_coverage == pytest.approx(0.95, abs=0.015)
    assert abs(res.naive_mean_bias) < 0.03
    assert res.conditional_median_width == pytest.approx(res.naive_median_width, rel=0.05)


def test_dominant_winner_makes_conditional_inference_naive():
    x = np.array([8.0, 0.3, -0.4, 0.1])
    res = winner_inference(x, np.eye(4), 0.05, critical_values=(2.5, 3.1))
    assert res.conditional.lower == pytest.approx(res.naive.lower, abs=1e-6)
    assert res.conditional.upper == pytest.approx(res.naive.upper, abs=1e-3)
    assert res.conditional_estimate == pytest.approx(8.0, abs=1e-6)


def test_conditional_interval_explodes_when_the_winner_barely_wins_and_hybrid_does_not():
    x = np.array([1.0, 1.0 - 1e-4])
    sigma = np.eye(2)
    with pytest.warns(WincurseWarning, match="use the hybrid"):
        res = winner_inference(x, sigma, 0.05, n_sim=100_000, rng=0)
    assert res.conditional.width > 1000
    assert np.isfinite(res.conditional.lower)
    assert res.hybrid.width < res.projection.width + 1e-9
    assert res.hybrid.width < 8.0


def test_extreme_gaps_never_fail_to_bracket():
    # Regression: skipped bracketing failures once inflated coverage from 95.0% to 97.5%.
    sigma = equicorrelated(2, 1e-9)
    res = simulate_known_sigma(np.zeros(2), sigma, 3000, n_sim=20_000, rng=11, chunk=1000)
    assert res.n_failed == 0
    assert res.conditional_coverage == pytest.approx(0.95, abs=0.02)
    sigma = np.eye(2)
    res = simulate_known_sigma(np.zeros(2), sigma, 3000, n_sim=20_000, rng=12)
    assert res.n_failed == 0


def test_failures_are_counted_as_misses_never_skipped():
    rng = np.random.default_rng(0)
    n = 2000
    mu = np.zeros(n)
    x = rng.standard_normal(n)
    se = np.ones(n)
    lo, hi = x - 1.96, x + 1.96
    est = x.copy()
    lo_c, hi_c, est_c = lo.copy(), hi.copy(), est.copy()
    lo_c[:100] = np.nan
    metrics = replication_metrics(
        mu, x, se, 2.5, est_c, lo_c, hi_c, est, lo, hi, alpha=0.05, n_trials=5
    )
    assert metrics.n_failed == 100
    truth = np.mean((lo <= mu) & (mu <= hi))
    assert metrics.hybrid_coverage == pytest.approx(truth)
    assert metrics.conditional_coverage <= truth - 0.9 * 100 / n * truth
    assert metrics.n_rep == n


def test_root_finding_failure_raises_in_the_public_api(monkeypatch):
    from wincurse import winners

    def broken(lp, up, p, c=np.inf, **_):
        return np.full(np.broadcast(lp, up, p).shape, np.nan)

    monkeypatch.setattr(winners, "mean_quantile", broken)
    with pytest.raises(RootFindingError):
        winner_inference(np.array([1.0, 0.2]), np.eye(2), critical_values=(2.0, 3.0))


def test_annualisation_equivariance_of_the_intervals(rng):
    sigma = equicorrelated(8, 0.4) * 0.01
    x = rng.standard_normal(8) * 0.1
    q = 252.0
    base = winner_inference(x, sigma, 0.05, critical_values=(2.5, 3.0))
    scaled = winner_inference(np.sqrt(q) * x, q * sigma, 0.05, critical_values=(2.5, 3.0))
    for name in ("naive", "conditional", "hybrid", "projection"):
        a, b = getattr(base, name), getattr(scaled, name)
        assert b.lower == pytest.approx(np.sqrt(q) * a.lower, rel=1e-9)
        assert b.upper == pytest.approx(np.sqrt(q) * a.upper, rel=1e-9)


def test_t_statistic_selection_rescales_consistently(rng):
    sd = np.array([0.5, 1.0, 2.0, 1.5])
    corr = equicorrelated(4, 0.3)
    sigma = corr * np.outer(sd, sd)
    x = np.array([0.4, 0.9, 1.4, 1.0])
    res = winner_inference(x, sigma, select="t", critical_values=(2.5, 3.0))
    assert res.index == int(np.argmax(x / sd))
    t_res = winner_inference(x / sd, corr, critical_values=(2.5, 3.0))
    k = res.index
    assert res.conditional.lower == pytest.approx(sd[k] * t_res.conditional.lower)
    assert res.hybrid.upper == pytest.approx(sd[k] * t_res.hybrid.upper)
    assert res.naive.lower == pytest.approx(x[k] - stats.norm.isf(0.025) * sd[k])


def test_equal_variance_and_covariance_is_treated_as_equality_not_noise():
    # A duplicate of the winner (Sigma_jk = Sigma_kk) with a smaller estimate imposes no
    # bound; floating-point dust in Sigma_kk - Sigma_jk must not create arbitrary bounds.
    sigma = np.array([[1.0, 1.0 - 1e-15, 0.0], [1.0 - 1e-15, 1.0, 0.0], [0.0, 0.0, 1.0]])
    x = np.array([0.9, 0.5, 0.1])
    lo, hi = selection_bounds(x, sigma, 0)
    assert hi == np.inf
    assert lo == pytest.approx(0.1)
    bad = np.array([0.5, 0.9, 0.1])
    with pytest.raises(ValueError, match="inconsistent"):
        selection_bounds(bad, sigma, 0)


def test_input_validation():
    s = np.eye(3)
    x = np.array([1.0, 0.0, 0.5])
    with pytest.raises(ValueError, match="shape"):
        winner_inference(x, np.eye(2))
    with pytest.raises(ValueError, match="symmetric"):
        winner_inference(x, np.triu(np.ones((3, 3))))
    with pytest.raises(ValueError, match="alpha"):
        winner_inference(x, s, alpha=1.5)
    with pytest.raises(ValueError, match="beta"):
        winner_inference(x, s, alpha=0.05, beta=0.06)
    with pytest.raises(ValueError, match="select"):
        winner_inference(x, s, select="median")
    with pytest.raises(ValueError, match="finite"):
        winner_inference(np.array([np.nan, 0.0, 0.1]), s)
    with pytest.raises(ValueError, match="positive diagonal"):
        winner_inference(x, np.diag([1.0, 0.0, 1.0]))


# ---------------------------------------------------------------- forecast intervals


def test_forecast_interval_covers_a_new_realisation(rng):
    k_trials, n = 6, 1000
    sigma = equicorrelated(k_trials, 0.4)
    mu = np.zeros(k_trials)
    x = _draw(rng, mu, sigma, n)
    oos_var = 0.8
    c_beta = independent_critical_value(k_trials, 0.005)  # conservative stand-in
    hits_c = hits_h = hits_n = 0
    for i in range(n):
        fi = forecast_interval(x[i], sigma, oos_var, 0.05, 0.005, critical_value_beta=c_beta)
        x2 = mu[fi.index] + np.sqrt(oos_var) * rng.standard_normal()
        hits_c += x2 in fi.conditional
        hits_h += x2 in fi.hybrid
        hits_n += x2 in fi.naive
    mc = 3 * np.sqrt(0.95 * 0.05 / n)
    assert hits_c / n == pytest.approx(0.95, abs=mc)
    assert hits_h / n >= 0.95 - mc
    assert hits_n / n < 0.95 - mc


def test_forecast_interval_validation():
    with pytest.raises(ValueError, match="oos_variance"):
        forecast_interval(np.array([1.0, 0.0]), np.eye(2), 0.0)


# ------------------------------------------------- AKM (2024) JobStart replication

JOBSTART = np.load(Path(__file__).parent / "data" / "akm_jobstart.npz")
ALPHA, BETA = 0.05, 0.005


def _exact_reference():
    """Independent 50-digit evaluation of the JobStart intervals for diagonal Sigma."""
    mu = JOBSTART["mu"]
    sigma = JOBSTART["sigma"]
    k = int(np.argmax(mu))
    m, s2 = mu[k], sigma[k, k]
    s = np.sqrt(s2)
    z = mu - sigma[:, k] / s2 * m
    lower, upper = -np.inf, np.inf
    for j in range(len(mu)):
        if j == k:
            continue
        d = s2 - sigma[j, k]
        if d > 0:
            lower = max(lower, s2 * z[j] / d)
        elif d < 0:
            upper = min(upper, s2 * z[j] / d)

    def q(t):
        return mp.erfc(mp.mpf(t) / mp.sqrt(2)) / 2

    def cdf(mean, a, b):
        za, zb, zx = (a - mean) / s, (b - mean) / s, (m - mean) / s
        qb = mp.mpf(0) if zb == np.inf else q(zb)
        return 1 - (q(zx) - qb) / (q(za) - qb)

    def solve(f, p, lo, hi):
        lo, hi = mp.mpf(lo), mp.mpf(hi)
        for _ in range(220):
            mid = (lo + hi) / 2
            if f(mid) > p:
                lo = mid
            else:
                hi = mid
        return float((lo + hi) / 2)

    kk = len(mu)
    c_beta = stats.norm.isf((1 - (1 - BETA) ** (1 / kk)) / 2)
    gamma = (ALPHA - BETA) / (1 - BETA)

    def hybrid_cdf(mean):
        a = max(lower, float(mean) - c_beta * s)
        b = min(upper, float(mean) + c_beta * s)
        return cdf(mean, a, b)

    full = (m - 40 * s, m + 40 * s)
    tight = (m - c_beta * s + 1e-9, m + c_beta * s - 1e-9)
    return {
        "truncation": (lower, upper),
        "conditional_estimate": solve(lambda t: cdf(t, lower, upper), 0.5, *full),
        "conditional": (
            solve(lambda t: cdf(t, lower, upper), 1 - ALPHA / 2, *full),
            solve(lambda t: cdf(t, lower, upper), ALPHA / 2, *full),
        ),
        "hybrid_estimate": solve(hybrid_cdf, 0.5, *tight),
        "hybrid": (
            solve(hybrid_cdf, 1 - gamma / 2, *tight),
            solve(hybrid_cdf, gamma / 2, *tight),
        ),
    }


def test_jobstart_matches_an_independent_exact_computation():
    mu, sigma = JOBSTART["mu"], JOBSTART["sigma"]
    assert np.abs(sigma - np.diag(np.diag(sigma))).max() == 0.0
    ca = independent_critical_value(13, ALPHA)
    cb = independent_critical_value(13, BETA)
    res = winner_inference(mu, sigma, ALPHA, BETA, critical_values=(ca, cb))
    ref = _exact_reference()
    assert res.index == 1
    assert res.truncation == pytest.approx(ref["truncation"])
    assert res.conditional_estimate == pytest.approx(ref["conditional_estimate"], rel=1e-8)
    assert res.conditional.lower == pytest.approx(ref["conditional"][0], rel=1e-8)
    assert res.conditional.upper == pytest.approx(ref["conditional"][1], rel=1e-8)
    assert res.hybrid_estimate == pytest.approx(ref["hybrid_estimate"], rel=1e-8)
    assert res.hybrid.lower == pytest.approx(ref["hybrid"][0], rel=1e-8)
    assert res.hybrid.upper == pytest.approx(ref["hybrid"][1], rel=1e-8)


def test_jobstart_reproduces_the_printed_tables_of_the_paper():
    # Table II of AKM (2024). The authors' endpoints are Monte Carlo output (sorted
    # truncated-normal draws, R_cv up to 5e5, noisy root search), so they agree with the
    # exact values to a few units on a standard error of 1496; lower endpoints of the
    # conditional and hybrid intervals differ by up to about 10 (see docs/methodology.md).
    mu, sigma = JOBSTART["mu"], JOBSTART["sigma"]
    res = winner_inference(mu, sigma, ALPHA, BETA, n_sim=500_000, rng=1)
    assert res.estimate == 6547
    assert res.naive.lower == pytest.approx(3615, abs=1)
    assert res.naive.upper == pytest.approx(9479, abs=1)
    assert res.conditional_estimate == pytest.approx(6544, abs=1)
    assert res.conditional.lower == pytest.approx(3485, abs=12)
    assert res.conditional.upper == pytest.approx(9478, abs=3)
    assert res.projection.lower == pytest.approx(2232, abs=6)
    assert res.projection.upper == pytest.approx(10862, abs=6)
    assert res.hybrid_estimate == pytest.approx(6545, abs=1)
    assert res.hybrid.lower == pytest.approx(3420, abs=12)
    assert res.hybrid.upper == pytest.approx(9538, abs=3)


@pytest.mark.parametrize(
    ("scale", "conditional", "hybrid"),
    [
        (1315.0, (2609, 10449), (2531, 10529)),
        (2607.0, (634, 12436), (520, 12556)),
    ],
)
def test_jobstart_forecast_intervals_match_table_iii(scale, conditional, hybrid):
    mu, sigma = JOBSTART["mu"], JOBSTART["sigma"]
    cb = independent_critical_value(13, BETA)
    fi = forecast_interval(mu, sigma, scale**2, ALPHA, BETA, critical_value_beta=cb)
    assert fi.conditional.lower == pytest.approx(conditional[0], abs=30)
    assert fi.conditional.upper == pytest.approx(conditional[1], abs=5)
    assert fi.hybrid.lower == pytest.approx(hybrid[0], abs=30)
    assert fi.hybrid.upper == pytest.approx(hybrid[1], abs=5)
