from __future__ import annotations

import numpy as np
import pytest

from wincurse import effective_number_of_trials, winner_margin
from wincurse.diagnostics import diagnose

from .conftest import equicorrelated


def test_effective_number_of_trials_extremes_and_closed_form():
    assert effective_number_of_trials(np.eye(7)) == pytest.approx(7.0)
    assert effective_number_of_trials(np.ones((7, 7)) + 1e-12 * np.eye(7)) == pytest.approx(
        1.0, abs=1e-6
    )
    k, rho = 20, 0.5
    big = 1 + (k - 1) * rho
    small = 1 - rho
    expected = k**2 / (big**2 + (k - 1) * small**2)
    assert effective_number_of_trials(equicorrelated(k, rho)) == pytest.approx(expected)


def test_effective_number_ignores_the_scale_of_each_trial():
    sigma = equicorrelated(6, 0.3)
    scale = np.diag([1.0, 2.0, 5.0, 0.1, 3.0, 1.5])
    assert effective_number_of_trials(scale @ sigma @ scale) == pytest.approx(
        effective_number_of_trials(sigma)
    )


def test_winner_margin_for_independent_trials():
    x = np.array([1.0, 0.4, -0.2])
    margin, rival = winner_margin(x, np.eye(3))
    assert rival == 1
    assert margin == pytest.approx(0.6 / np.sqrt(2))


def test_winner_margin_uses_the_correlation():
    sigma = equicorrelated(2, 0.9)
    margin, _ = winner_margin(np.array([1.0, 0.5]), sigma)
    assert margin == pytest.approx(0.5 / np.sqrt(2 * 0.1))


def test_winner_margin_single_trial_and_duplicates():
    assert winner_margin(np.array([1.0]), np.array([[1.0]])) == (np.inf, 0)
    sigma = np.ones((2, 2))
    margin, _ = winner_margin(np.array([1.0, 0.9]), sigma)
    assert margin == np.inf


def test_diagnose_collects_the_summary():
    x = np.array([1.0, 0.7, 0.1])
    d = diagnose(x, np.eye(3), 0, conditional_width=3.0, naive_width=2.0)
    assert d.gap_in_se == pytest.approx(0.3)
    assert d.margin == pytest.approx(0.3 / np.sqrt(2))
    assert d.runner_up == 1
    assert d.effective_trials == pytest.approx(3.0)
    assert d.conditional_to_naive_width == pytest.approx(1.5)
    solo = diagnose(np.array([1.0]), np.array([[1.0]]), 0, 2.0, 2.0)
    assert solo.gap_in_se == np.inf
