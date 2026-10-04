from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.signal import lfilter

from wincurse import (
    WincurseWarning,
    estimate_performance,
    influence_functions,
    ledoit_wolf_constant_correlation,
    long_run_covariance,
    nearest_psd,
    newey_west_bandwidth,
    sharpe_ratios,
    stationary_bootstrap_covariance,
)

from .conftest import equicorrelated


def correlated_normals(rng, reps, t, k, rho, mean_sr):
    """Per-period returns with unit variance, correlation ``rho`` and Sharpe ``mean_sr``."""
    chol = np.linalg.cholesky(equicorrelated(k, rho))
    return rng.standard_normal((reps, t, k)) @ chol.T + mean_sr


def simulated_sharpe_covariance(panels):
    sr = panels.mean(axis=1) / panels.std(axis=1, ddof=1)
    return np.cov(sr, rowvar=False)


def test_sharpe_ratios_follow_the_definition():
    r = np.array([[0.01, 0.02], [0.03, -0.01], [-0.02, 0.01], [0.02, 0.0]] * 4)
    expected = np.sqrt(12) * r.mean(0) / r.std(0, ddof=1)
    np.testing.assert_allclose(sharpe_ratios(r, 12), expected)


def test_influence_function_is_the_delta_method_gradient(rng):
    r = rng.standard_normal((200, 1)) * 0.02 + 0.003
    psi = influence_functions(r)[:, 0]

    def sr(m1, m2):
        return m1 / np.sqrt(m2 - m1**2)

    m1, m2 = r.mean(), (r**2).mean()
    h = 1e-7
    d1 = (sr(m1 + h, m2) - sr(m1 - h, m2)) / (2 * h)
    d2 = (sr(m1, m2 + h) - sr(m1, m2 - h)) / (2 * h)
    expected = d1 * (r[:, 0] - m1) + d2 * (r[:, 0] ** 2 - m2)
    np.testing.assert_allclose(psi, expected, rtol=1e-6)
    assert abs(psi.mean()) < 1e-12


def test_iid_normal_covariance_matches_lo_memmel_and_simulation(rng):
    t, k, rho, sr = 500, 3, 0.5, 0.1
    panels = correlated_normals(rng, 4000, t, k, rho, sr)
    sim = t * simulated_sharpe_covariance(panels)
    formula = rho + sr**2 * rho**2 / 2
    var_formula = 1 + sr**2 / 2
    assert np.diag(sim) == pytest.approx(var_formula, rel=0.08)
    assert sim[0, 1] == pytest.approx(formula, rel=0.08)
    for method in ("normal", "iid", "hac"):
        mean_cov = np.mean(
            [
                estimate_performance(panels[i], cov_method=method).per_period_cov * t
                for i in range(150)
            ],
            axis=0,
        )
        assert np.diag(mean_cov) == pytest.approx(var_formula, rel=0.06)
        assert mean_cov[0, 1] == pytest.approx(formula, rel=0.1)


def test_estimated_covariance_reproduces_lo_formula_exactly_for_normal_method():
    r = np.random.default_rng(3).standard_normal((400, 2))
    r[:, 1] = 0.6 * r[:, 0] + 0.8 * r[:, 1]
    est = estimate_performance(r, cov_method="normal")
    sr = r.mean(0) / r.std(0, ddof=1)
    rho = np.corrcoef(r.T)[0, 1]
    assert est.per_period_cov[0, 0] == pytest.approx((1 + sr[0] ** 2 / 2) / 400)
    assert est.per_period_cov[0, 1] == pytest.approx((rho + sr[0] * sr[1] * rho**2 / 2) / 400)


def _ar1_panels(rng, reps, t, k, phi, rho=0.5, mean=0.1):
    chol = np.linalg.cholesky(equicorrelated(k, rho))
    shocks = rng.standard_normal((reps, t, k)) @ chol.T
    return lfilter([1.0], [1.0, -phi], shocks, axis=1) * np.sqrt(1 - phi**2) + mean


def test_hac_covariance_tracks_simulation_under_ar1(rng):
    t, k = 1500, 3
    panels = _ar1_panels(rng, 1500, t, k, 0.25)
    sim = t * simulated_sharpe_covariance(panels)
    hac = np.mean(
        [estimate_performance(panels[i], cov_method="hac").per_period_cov * t for i in range(150)],
        axis=0,
    )
    iid = np.mean(
        [estimate_performance(panels[i], cov_method="iid").per_period_cov * t for i in range(150)],
        axis=0,
    )
    assert np.diag(sim).mean() > 1.3 * np.diag(iid).mean()
    assert abs(np.diag(hac).mean() / np.diag(sim).mean() - 1) < 0.15
    assert abs(np.diag(iid).mean() / np.diag(sim).mean() - 1) > 0.25
    assert hac[0, 1] / hac[0, 0] == pytest.approx(sim[0, 1] / sim[0, 0], abs=0.06)


def _garch_panels(rng, reps, t, k, alpha=0.05, beta=0.9, rho=0.5, mean=0.1):
    chol = np.linalg.cholesky(equicorrelated(k, rho))
    eps = rng.standard_normal((reps, t, k)) @ chol.T
    h = np.ones((reps, k))
    out = np.empty((reps, t, k))
    omega = 1 - alpha - beta
    for i in range(t):
        out[:, i] = np.sqrt(h) * eps[:, i]
        h = omega + alpha * out[:, i] ** 2 + beta * h
    return out + mean


def test_robust_covariance_tracks_simulation_under_garch_where_normal_formula_fails(rng):
    t, k = 2000, 2
    panels = _garch_panels(rng, 2500, t, k)
    sim = t * simulated_sharpe_covariance(panels)
    hac = np.mean(
        [estimate_performance(panels[i], cov_method="hac").per_period_cov * t for i in range(200)],
        axis=0,
    )
    normal = np.mean(
        [
            estimate_performance(panels[i], cov_method="normal").per_period_cov * t
            for i in range(200)
        ],
        axis=0,
    )
    assert abs(np.diag(hac).mean() / np.diag(sim).mean() - 1) < 0.2
    assert abs(np.diag(hac).mean() / np.diag(sim).mean() - 1) < abs(
        np.diag(normal).mean() / np.diag(sim).mean() - 1
    )


def test_stationary_bootstrap_is_unbiased_for_the_iid_covariance_on_average(rng):
    # A single bootstrap covariance has sampling noise of order sqrt(b / T); the average
    # over independent datasets isolates the bias.
    rho, sr = 0.4, 0.1
    draws = []
    for i in range(60):
        r = correlated_normals(rng, 1, 500, 2, rho, sr)[0]
        draws.append(stationary_bootstrap_covariance(r, n_boot=300, rng=i))
    mean = np.mean(draws, axis=0)
    assert np.diag(mean) == pytest.approx(1 + sr**2 / 2, rel=0.08)
    assert mean[0, 1] == pytest.approx(rho + sr**2 * rho**2 / 2, rel=0.12)


def test_stationary_bootstrap_captures_serial_dependence(rng):
    r = _ar1_panels(rng, 1, 1200, 2, 0.4)[0]
    boot = stationary_bootstrap_covariance(r, n_boot=1500, mean_block_length=12.0, rng=2)
    iid = long_run_covariance(influence_functions(r), "iid")
    assert np.diag(boot).mean() > 1.3 * np.diag(iid).mean()


def test_bootstrap_is_reproducible_and_validated(rng):
    r = rng.standard_normal((100, 2))
    a = stationary_bootstrap_covariance(r, n_boot=50, rng=5)
    b = stationary_bootstrap_covariance(r, n_boot=50, rng=5)
    np.testing.assert_array_equal(a, b)
    with pytest.raises(ValueError, match="mean_block_length"):
        stationary_bootstrap_covariance(r, mean_block_length=0.5)


def test_annualisation_is_a_joint_scaling_of_estimates_and_covariance(rng):
    r = pd.DataFrame(rng.standard_normal((600, 4)) * 0.01 + 0.0004)
    daily = estimate_performance(r, 1.0)
    annual = estimate_performance(r, 252.0)
    np.testing.assert_allclose(annual.values, np.sqrt(252) * daily.values)
    np.testing.assert_allclose(annual.cov, 252 * daily.cov)
    np.testing.assert_allclose(annual.per_period_values, daily.values)
    np.testing.assert_allclose(annual.per_period_cov, daily.cov)


def test_annualisation_uses_the_per_period_sharpe_in_the_variance(rng):
    # Daily Sharpe 0.1 over 10 years: the correct annual standard error is 0.317,
    # not the 0.475 obtained by plugging the annualised Sharpe into the formula.
    t, q = 2520, 252
    panels = rng.standard_normal((1500, t, 1)) + 0.1
    sr_annual = np.sqrt(q) * panels.mean(1) / panels.std(1, ddof=1)
    assert sr_annual.std() == pytest.approx(0.317, rel=0.05)
    est = estimate_performance(panels[0], q, cov_method="normal")
    assert np.sqrt(est.cov[0, 0]) == pytest.approx(0.317, rel=0.1)


def test_mean_statistic_has_the_textbook_covariance(rng):
    r = rng.standard_normal((900, 3)) * 0.02 + 0.001
    est = estimate_performance(r, 12.0, statistic="mean", cov_method="iid")
    np.testing.assert_allclose(est.values, 12 * r.mean(0))
    np.testing.assert_allclose(est.cov, 144 * np.cov(r.T, ddof=0) / 900, rtol=1e-10)
    with pytest.raises(ValueError, match="only defined for the Sharpe"):
        estimate_performance(r, 12.0, statistic="mean", cov_method="normal")


def test_newey_west_bandwidth_grows_with_persistence(rng):
    # The rule is data dependent and noisy for a single panel; compare averages.
    white = [rng.standard_normal((1000, 3)) for _ in range(20)]
    persistent = [lfilter([1.0], [1.0, -0.7], w, axis=0) for w in white]
    mean_white = np.mean([newey_west_bandwidth(w) for w in white])
    mean_persistent = np.mean([newey_west_bandwidth(w) for w in persistent])
    assert mean_white < 4
    assert mean_persistent > 12


def test_hac_is_positive_semi_definite_and_reduces_to_iid_at_zero_lags(rng):
    psi = influence_functions(_ar1_panels(rng, 1, 300, 6, 0.3)[0])
    omega = long_run_covariance(psi, "hac", 8)
    assert np.linalg.eigvalsh(omega).min() > -1e-12
    np.testing.assert_allclose(long_run_covariance(psi, "hac", 0), long_run_covariance(psi, "iid"))
    with pytest.raises(ValueError, match="bandwidth"):
        long_run_covariance(psi, "hac", 300)


def test_hac_matches_the_explicit_bartlett_sum(rng):
    psi = rng.standard_normal((120, 3))
    psi -= psi.mean(0)
    m = 5
    t = psi.shape[0]
    expected = psi.T @ psi / t
    for lag in range(1, m + 1):
        gamma = psi[lag:].T @ psi[: t - lag] / t
        expected = expected + (1 - lag / (m + 1)) * (gamma + gamma.T)
    np.testing.assert_allclose(long_run_covariance(psi, "hac", m), expected, atol=1e-12)


def test_nearest_psd_repairs_and_preserves_the_diagonal():
    bad = np.array([[1.0, 0.9, 0.2], [0.9, 1.0, -0.9], [0.2, -0.9, 1.0]])
    assert np.linalg.eigvalsh(bad)[0] < 0
    fixed, changed = nearest_psd(bad)
    assert changed
    assert np.linalg.eigvalsh(fixed)[0] > 0
    np.testing.assert_allclose(np.diag(fixed), np.diag(bad))
    good = equicorrelated(4, 0.3)
    same, changed = nearest_psd(good)
    assert not changed
    np.testing.assert_allclose(same, good)


def _lw_reference(x):
    """Ledoit and Wolf (2003) constant-correlation intensity, written with explicit loops."""
    t, n = x.shape
    x = x - x.mean(0)
    s = x.T @ x / t
    sd = np.sqrt(np.diag(s))
    r_bar = (s / np.outer(sd, sd)).sum() - n
    r_bar /= n * (n - 1)
    target = r_bar * np.outer(sd, sd)
    np.fill_diagonal(target, np.diag(s))
    pi = sum(np.mean((x[:, i] * x[:, j] - s[i, j]) ** 2) for i in range(n) for j in range(n))
    rho = sum(np.mean((x[:, i] ** 2 - s[i, i]) * (x[:, i] ** 2 - s[i, i])) for i in range(n))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            th_ii_ij = np.mean((x[:, i] ** 2 - s[i, i]) * (x[:, i] * x[:, j] - s[i, j]))
            th_jj_ij = np.mean((x[:, j] ** 2 - s[j, j]) * (x[:, i] * x[:, j] - s[i, j]))
            rho += (
                r_bar
                / 2
                * (np.sqrt(s[j, j] / s[i, i]) * th_ii_ij + np.sqrt(s[i, i] / s[j, j]) * th_jj_ij)
            )
    gamma = np.sum((target - s) ** 2)
    return float(max(0.0, min(1.0, (pi - rho) / gamma / t)))


def test_ledoit_wolf_intensity_matches_the_explicit_formula(rng):
    x = correlated_normals(rng, 1, 60, 8, 0.3, 0.0)[0] * np.linspace(0.5, 2, 8)
    _, delta = ledoit_wolf_constant_correlation(x)
    assert delta == pytest.approx(_lw_reference(x), rel=1e-10)
    assert 0.0 < delta < 1.0


def test_ledoit_wolf_shrinkage_properties(rng):
    k = 30
    block = np.repeat([0, 1, 2], k // 3)
    truth = np.where(block[:, None] == block[None, :], 0.6, 0.1)
    np.fill_diagonal(truth, 1.0)
    chol = np.linalg.cholesky(truth)
    errors_s, errors_lw, deltas = [], [], {}
    for t in (40, 400):
        ds = []
        for _ in range(20):
            x = rng.standard_normal((t, k)) @ chol.T
            sample = np.cov(x.T, ddof=0)
            shrunk, delta = ledoit_wolf_constant_correlation(x)
            ds.append(delta)
            if t == 40:
                errors_s.append(np.sum((sample - truth) ** 2))
                errors_lw.append(np.sum((shrunk - truth) ** 2))
            np.testing.assert_allclose(np.diag(shrunk), np.diag(sample))
            assert np.linalg.eigvalsh(shrunk)[0] > 0
        deltas[t] = np.mean(ds)
    assert np.mean(errors_lw) < 0.95 * np.mean(errors_s)
    assert deltas[40] > deltas[400]


def test_rank_deficient_covariance_triggers_warnings_and_repair(rng):
    r = rng.standard_normal((30, 50)) * 0.01
    with pytest.warns(WincurseWarning) as record:
        est = estimate_performance(r, 12.0, cov_method="iid")
    messages = [str(w.message) for w in record]
    assert any("rank-deficient" in m for m in messages)
    assert any("repaired" in m for m in messages)
    assert est.repaired
    assert np.linalg.eigvalsh(est.cov)[0] > 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", WincurseWarning)
        shrunk = estimate_performance(r, 12.0, cov_method="iid", shrinkage=True)
    assert shrunk.shrinkage is not None
    assert 0 < shrunk.shrinkage <= 1
    assert shrunk.warnings


def test_shrinkage_warns_about_its_effect(rng):
    r = rng.standard_normal((200, 5)) * 0.01
    with pytest.warns(WincurseWarning, match="constant correlation"):
        estimate_performance(r, 12.0, shrinkage=True)


def test_names_and_series_input(rng):
    frame = pd.DataFrame(rng.standard_normal((50, 3)) * 0.01, columns=["a", "b", "c"])
    assert estimate_performance(frame, 12.0).names == ["a", "b", "c"]
    series = frame["a"].rename("solo")
    est = estimate_performance(series, 12.0)
    assert est.names == ["solo"]
    assert est.cov.shape == (1, 1)


@pytest.mark.parametrize(
    ("bad", "message"),
    [
        (np.full((20, 2), np.nan), "NaN"),
        (np.ones((20, 2)), "constant"),
        (np.random.default_rng(0).standard_normal((4, 2)), "at least 8"),
    ],
)
def test_input_validation(bad, message):
    with pytest.raises(ValueError, match=message):
        estimate_performance(bad, 12.0)
    with pytest.raises(ValueError, match="periods_per_year"):
        estimate_performance(np.random.default_rng(0).standard_normal((20, 2)), 0.0)
