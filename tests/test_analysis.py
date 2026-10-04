from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from wincurse import (
    WincurseWarning,
    analyze,
    check_out_of_sample,
    forecast_interval,
    forecast_pvalue,
    infer_periods_per_year,
)

from .conftest import equicorrelated


def make_panel(rng, t=1000, k=15, rho=0.4, drift=0.0002, freq="B"):
    chol = np.linalg.cholesky(equicorrelated(k, rho))
    values = (rng.standard_normal((t, k)) @ chol.T) * 0.01 + drift
    index = pd.date_range("2016-01-04", periods=t, freq=freq)
    return pd.DataFrame(values, index=index, columns=[f"v{i:02d}" for i in range(k)])


def test_infer_periods_per_year_from_the_index():
    assert infer_periods_per_year(pd.date_range("2020", periods=50, freq="B")) == 252
    assert infer_periods_per_year(pd.date_range("2020", periods=50, freq="W")) == 52
    assert infer_periods_per_year(pd.date_range("2020", periods=50, freq="ME")) == 12
    assert infer_periods_per_year(pd.date_range("2020", periods=50, freq="QE")) == 4
    assert infer_periods_per_year(pd.date_range("2020", periods=50, freq="YE")) == 1
    assert infer_periods_per_year(pd.RangeIndex(50)) is None


def test_analyze_end_to_end(rng):
    panel = make_panel(rng)
    res = analyze(panel, n_sim=20_000, seed=3)
    w = res.winner
    assert res.performance.periods_per_year == 252
    assert res.winner_name == panel.columns[w.index]
    assert w.index == int(np.argmax(res.performance.values))
    assert res.period == ("2016-01-04", str(panel.index[-1].date()))
    # The corrected point estimate is below the naive one and the interval is wider.
    assert res.headline_estimate < w.estimate
    assert 0 < res.haircut < 1
    assert w.hybrid.width >= w.naive.width
    assert res.headline_interval is w.hybrid
    assert res.dsr is not None
    assert res.haircuts is not None
    assert 0 <= res.dsr.dsr <= 1
    assert res.dsr.effective_trials < res.dsr.n_trials
    assert set(res.haircuts) == {"bonferroni", "bhy"}


def test_analyze_headline_follows_the_method(rng):
    panel = make_panel(rng, k=6)
    kwargs = {"n_sim": 10_000, "seed": 1}
    cond = analyze(panel, method="conditional", **kwargs)
    proj = analyze(panel, method="projection", **kwargs)
    naive = analyze(panel, method="naive", **kwargs)
    assert cond.headline_estimate == cond.winner.conditional_estimate
    assert cond.headline_interval is cond.winner.conditional
    assert proj.headline_estimate == proj.winner.estimate
    assert proj.headline_interval is proj.winner.projection
    assert naive.haircut == 0.0
    with pytest.raises(ValueError, match="method"):
        analyze(panel, method="magic")


def test_analyze_requires_a_frequency_for_plain_arrays(rng):
    values = make_panel(rng, t=200, k=4).to_numpy()
    with pytest.raises(ValueError, match="periods_per_year"):
        analyze(values)
    res = analyze(values, 12.0, n_sim=5000)
    assert res.period is None
    assert res.performance.periods_per_year == 12


def test_analyze_single_trial_has_no_correction(rng):
    panel = make_panel(rng, k=1)
    res = analyze(panel, n_sim=5000)
    assert res.winner.conditional.lower == pytest.approx(res.winner.naive.lower)
    assert res.diagnostics.margin == np.inf
    assert res.dsr is not None
    assert res.dsr.n_trials == 1


def test_analyze_mean_statistic_skips_sharpe_comparators(rng):
    res = analyze(make_panel(rng, k=5), statistic="mean", cov_method="iid", n_sim=5000)
    assert res.dsr is None
    assert res.haircuts is None
    assert res.performance.statistic == "mean"


def test_analyze_t_statistic_selection(rng):
    panel = make_panel(rng, k=8)
    panel.iloc[:, 3] *= 3.0
    by_t = analyze(panel, select="t", n_sim=5000)
    by_x = analyze(panel, select="estimate", n_sim=5000)
    t_stats = by_t.performance.values / np.sqrt(np.diag(by_t.performance.cov))
    assert by_t.winner.index == int(np.argmax(t_stats))
    assert by_x.winner.index == int(np.argmax(by_x.performance.values))
    assert by_t.winner.selection == "t"


def test_analyze_warns_when_the_winner_barely_wins(rng):
    panel = make_panel(rng, k=2, rho=0.0)
    panel.iloc[:, 1] = panel.iloc[:, 0] * (1 - 1e-6) + 1e-9 * rng.standard_normal(len(panel))
    res = analyze(panel, n_sim=5000)
    assert any("standard errors" in m for m in res.warnings)


def test_analyze_collects_covariance_warnings_instead_of_emitting_them(rng):
    panel = make_panel(rng, t=40, k=60, rho=0.2)
    with warnings.catch_warnings():
        warnings.simplefilter("error", WincurseWarning)
        res = analyze(panel, n_sim=3000, cov_method="iid")
    assert any("rank-deficient" in m for m in res.warnings)


def test_analyze_hidden_trials_adds_the_sensitivity(rng):
    panel = make_panel(rng, k=10)
    res = analyze(panel, n_sim=10_000, hidden_trials=20, hidden_correlation=0.3)
    assert res.projection_sensitivity is not None
    assert [r.n_hidden for r in res.projection_sensitivity] == [0, 10, 20]
    assert res.scenario is not None
    assert res.scenario.correlation == 0.3
    default_corr = analyze(panel, n_sim=5000, hidden_trials=4)
    assert default_corr.scenario is not None
    assert 0 <= default_corr.scenario.correlation < 1


def test_analyze_is_reproducible(rng):
    panel = make_panel(rng, k=8)
    a = analyze(panel, n_sim=10_000, seed=5)
    b = analyze(panel, n_sim=10_000, seed=5)
    assert a.winner == b.winner


# ---------------------------------------------------------------- out of sample


def test_forecast_pvalue_is_consistent_with_the_interval(rng):
    x = rng.standard_normal(8) * 0.5 + 1.0
    sigma = equicorrelated(8, 0.3) * 0.2
    fi = forecast_interval(x, sigma, 0.3, 0.05, 0.005, critical_value_beta=3.2)
    assert forecast_pvalue(x, sigma, 0.3, fi.conditional.lower) == pytest.approx(0.05, abs=1e-6)
    assert forecast_pvalue(x, sigma, 0.3, fi.conditional.upper) == pytest.approx(0.05, abs=1e-6)
    inside = 0.5 * (fi.conditional.lower + fi.conditional.upper)
    assert forecast_pvalue(x, sigma, 0.3, inside) > 0.05
    assert forecast_pvalue(x, sigma, 0.3, fi.conditional.lower - 1.0) < 0.05
    with pytest.raises(ValueError, match="oos_variance"):
        forecast_pvalue(x, sigma, -1.0, 0.0)


def test_out_of_sample_check_flags_a_decayed_winner(rng):
    panel = make_panel(rng, t=1500, k=12, drift=0.0003)
    res = analyze(panel, n_sim=20_000, seed=1)
    winner = res.winner_name
    same = make_panel(np.random.default_rng(99), t=750, k=12, drift=0.0003)
    ok = check_out_of_sample(res, same)
    assert ok.n_obs == 750
    assert ok.in_hybrid == (ok.realised in ok.forecast.hybrid)
    assert 0 <= ok.conditional_pvalue <= 1
    decayed = pd.DataFrame({winner: np.random.default_rng(5).standard_normal(750) * 0.01 - 0.003})
    bad = check_out_of_sample(res, decayed)
    assert bad.realised < res.winner.hybrid.lower
    assert not bad.in_hybrid
    assert not bad.in_conditional
    assert bad.conditional_pvalue < 0.05
    series_only = check_out_of_sample(res, decayed[winner])
    assert series_only.realised == pytest.approx(bad.realised)
    with pytest.raises(ValueError, match="no column"):
        check_out_of_sample(res, decayed.rename(columns={winner: "other"}))
    with pytest.raises(ValueError, match="winner's column"):
        check_out_of_sample(res, np.zeros((10, 3)))


def test_haircut_is_undefined_for_a_non_positive_naive_estimate(rng):
    from wincurse.cli import _pct

    panel = make_panel(rng, k=5, drift=-0.003)
    res = analyze(panel, n_sim=3000)
    assert res.winner.estimate < 0
    assert np.isnan(res.haircut)
    assert _pct(res.haircut) == "n/a"
