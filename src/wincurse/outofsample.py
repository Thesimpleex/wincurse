"""Compare the winner's realised out-of-sample performance with a forecast interval.

The out-of-sample Sharpe ratio of the selected strategy is a new, noisy draw. An
interval for the winner's *true* in-sample performance is not designed to cover it
(the realised value has its own sampling error), so the comparison uses the forecast
intervals of Andrews, Kitagawa and McCloskey (2024, Section 4.1 and Appendix E):
conditional on selection (or unconditionally, for the hybrid version) a new
independent estimate of the winner's unchanged true performance falls inside the
interval with probability ``1 - alpha``. A realised value below the interval signals
decay beyond what selection alone explains.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .analysis import Analysis
from .performance import estimate_performance
from .truncnorm import tn_cdf
from .winners import ForecastInterval, _frame, _relative_bounds, _validate, forecast_interval

__all__ = ["OutOfSample", "check_out_of_sample", "forecast_pvalue"]


@dataclass(frozen=True)
class OutOfSample:
    """Realised out-of-sample performance of the winner against its forecast intervals."""

    realised: float
    se: float
    n_obs: int
    forecast: ForecastInterval
    in_naive: bool
    in_conditional: bool
    in_hybrid: bool
    conditional_pvalue: float


def forecast_pvalue(
    x, sigma, oos_variance: float, realised: float, *, select: str = "estimate"
) -> float:
    """Conditional two-sided p-value for "the realised value comes from the winner's true value".

    Equals the smallest ``alpha`` for which the conditional forecast interval excludes
    ``realised``: ``2 min(F, 1 - F)`` with ``F`` the truncated-normal CDF of the
    difference between the winner's in-sample estimate and the realised value.
    """
    x, sigma = _validate(x, sigma)
    if not oos_variance > 0:
        raise ValueError("oos_variance must be positive")
    xs, ss, w = _frame(x, sigma, select)
    k = int(np.argmax(xs))
    var12 = ss[k, k] + oos_variance / w[k] ** 2
    s = float(np.sqrt(var12))
    lo, hi = _relative_bounds(xs[None, :], ss, np.array([k]), var12)
    diff = float(xs[k] - realised / w[k])
    cdf = float(tn_cdf(diff / s, 0.0, 1.0, (diff + lo[0]) / s, (diff + hi[0]) / s))
    return float(min(1.0, 2.0 * min(cdf, 1.0 - cdf)))


def check_out_of_sample(
    result: Analysis,
    oos_returns,
    *,
    cov_method: str | None = None,
    n_sim: int | None = None,
) -> OutOfSample:
    """Evaluate the winner of ``result`` on later data it has not seen.

    Parameters
    ----------
    result : Analysis
        Output of :func:`wincurse.analyze` on the in-sample panel.
    oos_returns : DataFrame, Series or array_like
        Later per-period returns. A frame must contain the winner's column; an array
        or Series is taken to be the winner's own series.
    cov_method : str, optional
        Covariance estimator for the out-of-sample sampling variance (default: the
        one used in sample).
    """
    perf = result.performance
    winner = result.winner
    if isinstance(oos_returns, pd.DataFrame):
        names = [str(c) for c in oos_returns.columns]
        if result.winner_name not in names:
            raise ValueError(f"out-of-sample frame has no column {result.winner_name!r}")
        series = oos_returns.iloc[:, names.index(result.winner_name)].to_numpy(float)
    else:
        series = np.asarray(oos_returns, dtype=float)
        if series.ndim == 2 and series.shape[1] == 1:
            series = series[:, 0]
        if series.ndim != 1:
            raise ValueError("pass the winner's column or a frame containing it")
    oos = estimate_performance(
        series[:, None],
        perf.periods_per_year,
        statistic=perf.statistic,
        cov_method=cov_method or perf.cov_method,
        rng=result.seed,
    )
    realised = float(oos.values[0])
    variance = float(oos.cov[0, 0])
    forecast = forecast_interval(
        perf.values,
        perf.cov,
        variance,
        winner.alpha,
        winner.beta,
        select=winner.selection,
        critical_value_beta=winner.critical_value_beta,
    )
    return OutOfSample(
        realised=realised,
        se=float(np.sqrt(variance)),
        n_obs=int(series.size),
        forecast=forecast,
        in_naive=realised in forecast.naive,
        in_conditional=realised in forecast.conditional,
        in_hybrid=realised in forecast.hybrid,
        conditional_pvalue=forecast_pvalue(
            perf.values, perf.cov, variance, realised, select=winner.selection
        ),
    )
