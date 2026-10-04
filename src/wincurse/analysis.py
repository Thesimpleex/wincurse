"""One-call analysis of a ``T x K`` return panel: estimates, corrections, comparators."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats

from .comparators import (
    ArchTests,
    Haircut,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    harvey_liu_haircut,
    spa_stepm,
)
from .diagnostics import Diagnostics, diagnose
from .exceptions import WincurseWarning
from .performance import CovMethod, PerformanceEstimate, Statistic, estimate_performance
from .sensitivity import (
    ProjectionSensitivity,
    ScenarioSummary,
    nearest_neighbour_correlation,
    unlogged_projection,
    unlogged_scenario,
)
from .winners import Selection, WinnerInference, winner_inference

__all__ = ["Analysis", "DeflatedSharpe", "analyze", "infer_periods_per_year"]

Method = Literal["hybrid", "conditional", "projection", "naive"]


@dataclass(frozen=True)
class DeflatedSharpe:
    """Deflated Sharpe ratio of the winner for two assumptions about the trial count.

    All Sharpe ratios are per period, as in the original formula.
    """

    sharpe_per_period: float
    sharpe_variance: float
    skewness: float
    kurtosis: float
    n_trials: float
    effective_trials: float
    expected_max_sharpe: float
    dsr: float
    expected_max_sharpe_effective: float
    dsr_effective: float


@dataclass(frozen=True)
class Analysis:
    """Everything ``wincurse analyze`` reports for one panel of trials."""

    performance: PerformanceEstimate
    winner: WinnerInference
    winner_name: str
    method: str
    diagnostics: Diagnostics
    haircuts: dict[str, Haircut] | None
    dsr: DeflatedSharpe | None
    projection_sensitivity: list[ProjectionSensitivity] | None
    scenario: ScenarioSummary | None
    warnings: list[str]
    seed: int
    n_sim: int
    period: tuple[str, str] | None = None
    arch: ArchTests | None = None

    @property
    def headline_estimate(self) -> float:
        """Point estimate of the chosen method (naive for ``projection``)."""
        w = self.winner
        return {
            "hybrid": w.hybrid_estimate,
            "conditional": w.conditional_estimate,
        }.get(self.method, w.estimate)

    @property
    def headline_interval(self):
        w = self.winner
        return {
            "hybrid": w.hybrid,
            "conditional": w.conditional,
            "projection": w.projection,
            "naive": w.naive,
        }[self.method]

    @property
    def haircut(self) -> float:
        """Relative reduction of the headline versus the naive estimate; NaN if that is <= 0."""
        if self.winner.estimate <= 0:
            return float("nan")
        return 1.0 - self.headline_estimate / self.winner.estimate


def infer_periods_per_year(index: pd.Index) -> float | None:
    """Observation frequency from a datetime index (daily 252, weekly 52, ...), else ``None``."""
    if not isinstance(index, pd.DatetimeIndex) or len(index) < 3:
        return None
    step = float(np.median(np.diff(index.values).astype("timedelta64[D]").astype(float)))
    if step <= 1.5:
        return 252.0
    if step <= 8:
        return 52.0
    if step <= 35:
        return 12.0
    if step <= 100:
        return 4.0
    return 1.0


def analyze(
    returns,
    periods_per_year: float | None = None,
    alpha: float = 0.05,
    beta: float | None = None,
    *,
    method: Method = "hybrid",
    select: Selection = "estimate",
    statistic: Statistic = "sharpe",
    cov_method: CovMethod = "hac",
    shrinkage: bool = False,
    n_sim: int = 100_000,
    seed: int = 0,
    hidden_trials: int = 0,
    hidden_correlation: float | None = None,
    arch_tests: bool = False,
) -> Analysis:
    """Correct the best of K trials for selection and report comparators and diagnostics.

    Parameters
    ----------
    returns : DataFrame or array_like
        ``T x K`` per-period excess returns (decimals), one column per tried variant.
    periods_per_year : float, optional
        Observation frequency ``q`` for annualisation; inferred from a datetime index
        if omitted.
    alpha, beta : float
        Confidence level ``1 - alpha``; ``beta`` (default ``alpha / 10``) is the
        projection step of the hybrid interval.
    method : {"hybrid", "conditional", "projection", "naive"}
        Which procedure supplies the headline estimate and interval; all are computed.
    select, statistic, cov_method, shrinkage
        See :func:`wincurse.winner_inference` and :func:`wincurse.estimate_performance`.
    n_sim, seed : int
        Draws and seed for the simulated critical values.
    hidden_trials : int
        If positive, add the sensitivity analysis for that many unlogged trials, each a
        noisy variant of a logged one with correlation ``hidden_correlation`` to it
        (default: the mean over logged trials of their largest correlation with
        another logged trial).
    arch_tests : bool
        Also run Hansen's SPA and the Romano-Wolf StepM (needs the ``arch`` extra).

    Returns
    -------
    Analysis
    """
    if method not in ("hybrid", "conditional", "projection", "naive"):
        raise ValueError("method must be 'hybrid', 'conditional', 'projection' or 'naive'")
    index = returns.index if isinstance(returns, pd.DataFrame | pd.Series) else None
    if periods_per_year is None:
        periods_per_year = infer_periods_per_year(index) if index is not None else None
        if periods_per_year is None:
            raise ValueError(
                "periods_per_year is required when it cannot be inferred from a datetime index"
            )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", WincurseWarning)
        perf = estimate_performance(
            returns,
            periods_per_year,
            statistic=statistic,
            cov_method=cov_method,
            shrinkage=shrinkage,
            rng=seed,
        )
        win = winner_inference(
            perf.values, perf.cov, alpha, beta, select=select, n_sim=n_sim, rng=seed
        )
    notes = list(dict.fromkeys(str(w.message) for w in caught))
    k = win.index
    diag = diagnose(perf.values, perf.cov, k, win.conditional.width, win.naive.width)
    if diag.margin < 1.0:
        notes.append(
            f"the winner leads its closest rival by only {diag.margin:.2f} standard errors; "
            "selection matters a lot here"
        )

    haircuts = dsr = None
    if statistic == "sharpe":
        sr_pp = perf.per_period_values
        haircuts = harvey_liu_haircut(
            perf.values, perf.n_obs, winner=k, periods_per_year=periods_per_year
        )
        winner_returns = np.asarray(returns, dtype=float)
        winner_returns = (
            winner_returns[:, [k]] if winner_returns.ndim == 2 else winner_returns[:, None]
        )
        r = winner_returns[:, 0]
        var_sr = float(np.var(sr_pp, ddof=1)) if sr_pp.size > 1 else 0.0
        skew = float(stats.skew(r))
        kurt = float(stats.kurtosis(r, fisher=False))
        n_eff = diag.effective_trials
        dsr = DeflatedSharpe(
            sharpe_per_period=float(sr_pp[k]),
            sharpe_variance=var_sr,
            skewness=skew,
            kurtosis=kurt,
            n_trials=float(perf.n_strategies),
            effective_trials=n_eff,
            expected_max_sharpe=expected_max_sharpe(perf.n_strategies, var_sr),
            dsr=deflated_sharpe_ratio(
                float(sr_pp[k]), perf.n_obs, perf.n_strategies, var_sr, skew, kurt
            ),
            expected_max_sharpe_effective=expected_max_sharpe(max(n_eff, 1.0), var_sr),
            dsr_effective=deflated_sharpe_ratio(
                float(sr_pp[k]), perf.n_obs, max(n_eff, 1.0), var_sr, skew, kurt
            ),
        )

    sens = scenario = None
    if hidden_trials > 0:
        if hidden_correlation is None:
            hidden_correlation = nearest_neighbour_correlation(perf.cov)
        counts = (
            [0, hidden_trials // 2, hidden_trials] if hidden_trials >= 2 else [0, hidden_trials]
        )
        sens = unlogged_projection(
            perf.values,
            perf.cov,
            counts,
            hidden_correlation,
            alpha,
            n_sim=min(n_sim, 50_000),
            rng=seed,
        )
        try:
            scenario = unlogged_scenario(
                perf.values,
                perf.cov,
                hidden_trials,
                hidden_correlation,
                true_value=0.0,
                alpha=alpha,
                beta=beta,
                rng=seed,
            )
        except ValueError as exc:
            notes.append(f"scenario simulation for the unlogged trials skipped: {exc}")

    arch = None
    if arch_tests:
        panel = np.asarray(returns, dtype=float).reshape(perf.n_obs, perf.n_strategies)
        arch = spa_stepm(panel, k, perf.names, size=alpha, reps=1000, seed=seed)

    period = None
    if isinstance(index, pd.DatetimeIndex):
        period = (index.min().strftime("%Y-%m-%d"), index.max().strftime("%Y-%m-%d"))
    return Analysis(
        performance=perf,
        winner=win,
        winner_name=perf.names[k],
        method=method,
        diagnostics=diag,
        haircuts=haircuts,
        dsr=dsr,
        projection_sensitivity=sens,
        scenario=scenario,
        warnings=notes,
        seed=seed,
        n_sim=n_sim,
        period=period,
        arch=arch,
    )
