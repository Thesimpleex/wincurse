from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from wincurse import (
    deflated_sharpe_ratio,
    expected_max_sharpe,
    harvey_liu_haircut,
    multiple_testing_pvalues,
)

PERIODS = 250  # observations per year in the Bailey and Lopez de Prado example


def test_deflated_sharpe_reproduces_the_papers_worked_example():
    # Bailey and Lopez de Prado (2014), numerical example: annualised Sharpe ratio 2.5
    # over T = 1250 daily returns, skewness -3, kurtosis 10, cross-trial variance of the
    # annualised Sharpe ratios 0.5. With N = 100 trials the investor sees "only a 90%
    # chance"; with N = 46 the DSR is 0.9505; with normal returns N = 88 suffices.
    sr = 2.5 / np.sqrt(PERIODS)
    variance = 0.5 / PERIODS
    assert expected_max_sharpe(100, variance) == pytest.approx(0.11317, abs=5e-6)
    assert deflated_sharpe_ratio(sr, 1250, 100, variance, -3, 10) == pytest.approx(0.9004, abs=5e-5)
    assert deflated_sharpe_ratio(sr, 1250, 46, variance, -3, 10) == pytest.approx(0.9505, abs=5e-5)
    assert deflated_sharpe_ratio(sr, 1250, 88, variance, 0, 3) == pytest.approx(0.9505, abs=5e-5)
    assert (
        deflated_sharpe_ratio(sr, 1250, 89, variance, 0, 3)
        < 0.95
        < (deflated_sharpe_ratio(sr, 1250, 87, variance, 0, 3))
    )


def test_expected_max_sharpe_tracks_the_simulated_maximum(rng):
    n = 100
    draws = rng.standard_normal((20000, n)).max(axis=1).mean()
    assert expected_max_sharpe(n, 1.0) == pytest.approx(draws, abs=0.03)
    assert expected_max_sharpe(1, 1.0) == 0.0
    assert expected_max_sharpe(50, 4.0) == pytest.approx(2 * expected_max_sharpe(50, 1.0))


def test_deflated_sharpe_is_monotone_in_the_number_of_trials():
    values = [deflated_sharpe_ratio(0.1, 1000, n, 0.001) for n in (1, 5, 20, 100, 1000)]
    assert values == sorted(values, reverse=True)
    assert deflated_sharpe_ratio(0.1, 1000, 1, 0.001) == pytest.approx(
        stats.norm.cdf(0.1 * np.sqrt(999) / np.sqrt(1 + 2 * 0.01 / 4))
    )


def test_deflated_sharpe_validation():
    with pytest.raises(ValueError, match="n_obs"):
        deflated_sharpe_ratio(0.1, 1, 10, 0.01)
    with pytest.raises(ValueError, match="not positive"):
        deflated_sharpe_ratio(10.0, 100, 10, 0.01, skewness=5.0, kurtosis=1.0)
    with pytest.raises(ValueError, match="n_trials"):
        expected_max_sharpe(0.5, 1.0)


def _bhy_reference(p):
    m = len(p)
    c = sum(1 / i for i in range(1, m + 1))
    order = np.argsort(p)
    adjusted = np.empty(m)
    running = 1.0
    for rank in range(m, 0, -1):
        idx = order[rank - 1]
        running = min(running, p[idx] * m * c / rank)
        adjusted[idx] = running
    return adjusted


def test_multiple_testing_adjustments_match_straightforward_references(rng):
    p = rng.uniform(0, 0.2, size=25)
    m = len(p)
    np.testing.assert_allclose(multiple_testing_pvalues(p, "bonferroni"), np.minimum(1, m * p))
    np.testing.assert_allclose(multiple_testing_pvalues(p, "bhy"), _bhy_reference(p))
    order = np.argsort(p)
    holm = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        holm[idx] = min(1.0, running)
    np.testing.assert_allclose(multiple_testing_pvalues(p, "holm"), holm)
    with pytest.raises(ValueError, match="method"):
        multiple_testing_pvalues(p, "sidak")


def test_holm_equals_bonferroni_for_the_smallest_p_value(rng):
    p = rng.uniform(0, 0.05, size=50)
    k = int(np.argmin(p))
    holm = multiple_testing_pvalues(p, "holm")[k]
    bonferroni = multiple_testing_pvalues(p, "bonferroni")[k]
    assert holm == pytest.approx(bonferroni)
    assert bonferroni == pytest.approx(min(1.0, 50 * p[k]))


def test_haircut_for_one_trial_is_zero_and_grows_with_the_number_of_trials():
    out = harvey_liu_haircut(np.array([1.2]), 120, periods_per_year=12)
    assert out["bonferroni"].haircut == pytest.approx(0.0, abs=1e-9)
    assert out["bhy"].adjusted_sharpe == pytest.approx(1.2)
    t_stat = 1.0 * np.sqrt(120 / 12)
    many = np.concatenate([[1.0], np.full(99, 0.2)])
    haircut = harvey_liu_haircut(many, 120, periods_per_year=12)
    p = 2 * stats.norm.sf(t_stat)
    assert haircut["bonferroni"].p_value == pytest.approx(p)
    assert haircut["bonferroni"].adjusted_p_value == pytest.approx(min(1, 100 * p))
    # The adjusted Sharpe ratio is the one whose p-value equals the adjusted p-value.
    t_adj = stats.norm.isf(haircut["bonferroni"].adjusted_p_value / 2)
    assert haircut["bonferroni"].adjusted_sharpe == pytest.approx(t_adj / np.sqrt(10))
    # For the rank-one p-value the BHY factor m * c(m) exceeds the Bonferroni factor m.
    assert 0 < haircut["bonferroni"].haircut <= haircut["bhy"].haircut <= 1


def test_haircut_is_total_when_the_adjusted_p_value_reaches_one():
    out = harvey_liu_haircut(np.full(30, 0.1), 60, periods_per_year=12)
    assert out["bonferroni"].adjusted_p_value == 1.0
    assert out["bonferroni"].adjusted_sharpe == 0.0
    assert out["bonferroni"].haircut == pytest.approx(1.0)


def test_spa_and_stepm_detect_a_genuinely_profitable_strategy(rng):
    pytest.importorskip("arch")
    from wincurse import spa_stepm

    values = rng.standard_normal((1500, 15)) * 0.01
    values[:, 4] += 0.0012
    found = spa_stepm(values, 4, [f"s{i}" for i in range(15)], reps=400, seed=1)
    assert found.spa_pvalue < 0.05
    assert found.winner_is_superior
    assert "s4" in found.superior
    null = spa_stepm(rng.standard_normal((1500, 15)) * 0.01, 0, reps=400, seed=1)
    assert null.spa_pvalue > 0.05
    assert not null.superior
