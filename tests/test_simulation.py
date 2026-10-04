from __future__ import annotations

import numpy as np
import pytest

from wincurse import simulate_known_sigma

from .conftest import equicorrelated


def test_results_do_not_depend_on_the_chunk_size():
    sigma = equicorrelated(15, 0.5)
    kwargs = {"critical_values": (2.6, 3.3), "rng": 9}
    a = simulate_known_sigma(np.zeros(15), sigma, 600, chunk=600, **kwargs)
    b = simulate_known_sigma(np.zeros(15), sigma, 600, chunk=77, **kwargs)
    assert a == b


def test_every_replication_is_scored():
    res = simulate_known_sigma(
        np.zeros(5), equicorrelated(5, 0.2), 500, critical_values=(2.4, 3.0), rng=0
    )
    assert res.n_rep == 500
    assert res.n_failed == 0
    assert res.n_trials == 5
    assert set(res.to_dict()) >= {"naive_coverage", "hybrid_coverage", "n_failed"}


def test_selection_on_the_t_statistic_with_unequal_variances():
    sd = np.linspace(0.5, 2.0, 10)
    sigma = equicorrelated(10, 0.4) * np.outer(sd, sd)
    res = simulate_known_sigma(
        np.zeros(10), sigma, 3000, select="t", critical_values=(2.5, 3.1), rng=2
    )
    assert res.conditional_coverage == pytest.approx(0.95, abs=0.016)
    assert res.naive_coverage < 0.92
    assert res.n_failed == 0


def test_invalid_levels_are_rejected():
    with pytest.raises(ValueError, match="beta"):
        simulate_known_sigma(np.zeros(3), np.eye(3), 10, alpha=0.05, beta=0.5)


# ------------------------------------------------------------------ return-panel generator


def test_simulated_panels_have_the_requested_moments(rng):
    from wincurse.simulation import simulate_returns

    t, k, rho = 60_000, 4, 0.5
    r = simulate_returns(rng, t, k, rho, annual_sharpe=1.0, periods_per_year=252.0, vol=0.01)
    assert r.std(axis=0) == pytest.approx(0.01, rel=0.02)
    assert np.sqrt(252) * r.mean(0) / r.std(0) == pytest.approx(1.0, abs=0.12)
    corr = np.corrcoef(r, rowvar=False)
    assert corr[0, 1] == pytest.approx(rho, abs=0.02)
    spread = simulate_returns(rng, t, 3, 0.0, np.array([0.0, 0.5, 1.0]))
    assert (np.sqrt(252) * spread.mean(0) / spread.std(0)) == pytest.approx(
        [0.0, 0.5, 1.0], abs=0.12
    )


def test_heavy_tails_and_volatility_clustering(rng):
    from scipy import stats

    from wincurse.simulation import simulate_returns

    t5 = simulate_returns(rng, 200_000, 1, 0.0, dist="t5")[:, 0]
    assert t5.std() == pytest.approx(0.01, rel=0.03)
    assert stats.kurtosis(t5) > 3.0
    garch = simulate_returns(rng, 100_000, 1, 0.9, dist="garch")[:, 0]
    assert garch.std() == pytest.approx(0.01, rel=0.1)
    assert stats.kurtosis(garch) > 1.0
    squared = garch**2 - (garch**2).mean()
    assert np.sum(squared[1:] * squared[:-1]) / np.sum(squared**2) > 0.05
    normal = simulate_returns(rng, 100_000, 1, 0.9, dist="normal")[:, 0]
    assert abs(stats.kurtosis(normal)) < 0.1
    ar1 = simulate_returns(rng, 100_000, 2, 0.5, dist="ar1")
    assert ar1.std(axis=0) == pytest.approx(0.01, rel=0.02)
    assert np.corrcoef(ar1[1:, 0], ar1[:-1, 0])[0, 1] == pytest.approx(0.2, abs=0.02)
    assert np.corrcoef(ar1, rowvar=False)[0, 1] == pytest.approx(0.5, abs=0.03)
    with pytest.raises(ValueError, match="dist"):
        simulate_returns(rng, 10, 2, 0.5, dist="laplace")
    with pytest.raises(ValueError, match="rho"):
        simulate_returns(rng, 10, 2, 1.0)


def test_true_sharpe_covariance_matches_simulation(rng):
    from wincurse.simulation import normal_sharpe_covariance_true, simulate_returns

    t, k, rho, sharpe = 500, 3, 0.6, 1.5
    est = []
    for _ in range(3000):
        r = simulate_returns(rng, t, k, rho, sharpe, 252.0)
        est.append(np.sqrt(252) * r.mean(0) / r.std(0, ddof=1))
    sim = np.cov(np.array(est), rowvar=False)
    true = normal_sharpe_covariance_true(sharpe, rho, t, k, 252.0)
    assert np.diag(sim) == pytest.approx(np.diag(true), rel=0.08)
    assert sim[0, 1] == pytest.approx(true[0, 1], rel=0.1)


def test_feasible_study_is_independent_of_the_number_of_jobs():
    from wincurse.simulation import simulate_feasible

    kwargs = {"n_rep": 12, "n_obs": 400, "n_strategies": 6, "rho": 0.5, "n_sim": 3000}
    one, true_one = simulate_feasible(**kwargs, chunk=4, n_jobs=1, seed=3)
    two, true_two = simulate_feasible(**kwargs, chunk=4, n_jobs=2, seed=3)
    assert one == two
    assert true_one == true_two
    assert one.n_rep == 12


def test_feasible_study_covers_and_separates_estimated_from_true_covariance():
    from wincurse.simulation import simulate_feasible

    est, true = simulate_feasible(
        240, 750, 20, 0.5, 0.5, cov_method="hac", n_sim=6000, chunk=120, seed=1
    )
    assert true is not None
    assert est.n_failed == 0
    assert est.naive_coverage < 0.8
    assert est.conditional_coverage == pytest.approx(0.95, abs=0.06)
    assert true.conditional_coverage == pytest.approx(0.95, abs=0.06)
    heavy, none = simulate_feasible(
        100, 750, 8, 0.5, dist="t5", cov_method="iid", n_sim=4000, chunk=50, seed=2
    )
    assert none is None
    assert heavy.n_rep == 100
