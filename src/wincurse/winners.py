"""Inference on winners (Andrews, Kitagawa and McCloskey 2024).

Setting: ``X ~ N(mu, Sigma)`` collects the estimated performance of K trials with
known covariance of the estimation error; the trial ``k`` with the largest estimate
is reported. The naive estimate ``X_k`` is biased upwards and the naive interval
undercovers ``mu_k``.

*Conditional inference.* Let ``Z = X - Sigma[:, k] X_k / Sigma_kk``; ``Z`` is
independent of ``X_k``. Selection of ``k`` is equivalent to ``X_k in [L(Z), U(Z)]``
with ``L`` from the trials with ``Sigma_jk < Sigma_kk`` and ``U`` from those with
``Sigma_jk > Sigma_kk``. Hence ``X_k`` given selection and ``Z`` is
``N(mu_k, Sigma_kk)`` truncated to ``[L, U]``. Inverting this CDF in ``mu_k`` gives a
median-unbiased estimator and equal-tailed intervals with exact conditional coverage.

*Hybrid inference.* Conditioning additionally on ``|X_k - mu_k| <= c_beta sqrt(Sigma_kk)``,
with ``c_beta`` the ``1 - beta`` quantile of ``max_j |xi_j| / sqrt(Sigma_jj)``, and
running the conditional test at level ``(alpha - beta) / (1 - beta)`` bounds the
interval by the projection set and keeps unconditional coverage of at least
``1 - alpha``. The hybrid estimator is not median-unbiased; its median bias is at
most ``beta / 2``.

*Forecast intervals.* For a new, independent estimate ``X2`` of the winner with
variance ``Sigma2``, the same construction applied to ``X_k - X2`` gives an interval
for the realised ``X2`` under the hypothesis that the winner's true performance is
unchanged. It is the correct object to compare with a realised out-of-sample value.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import stats

from .critical import projection_critical_value
from .exceptions import RootFindingError, WincurseWarning
from .truncnorm import mean_quantile

__all__ = [
    "ForecastInterval",
    "Interval",
    "WinnerInference",
    "forecast_interval",
    "selection_bounds",
    "winner_inference",
    "winner_inference_cross",
]

Selection = Literal["estimate", "t"]
_EQUAL_RTOL = 1e-12


@dataclass(frozen=True)
class Interval:
    """A confidence or forecast interval with its nominal level."""

    lower: float
    upper: float
    level: float

    @property
    def width(self) -> float:
        return self.upper - self.lower

    def __contains__(self, value: float) -> bool:
        return self.lower <= value <= self.upper


@dataclass(frozen=True)
class WinnerInference:
    """Naive and selection-corrected inference for the selected trial's true value.

    ``conditional_estimate`` is median-unbiased given selection. ``hybrid_estimate``
    is within ``beta / 2`` of median-unbiased. ``truncation`` is ``(L, U)`` in the
    units of the estimates.
    """

    index: int
    selection: str
    alpha: float
    beta: float
    estimate: float
    se: float
    conditional_estimate: float
    hybrid_estimate: float
    naive: Interval
    conditional: Interval
    hybrid: Interval
    projection: Interval
    bonferroni: Interval
    truncation: tuple[float, float]
    critical_value_alpha: float
    critical_value_beta: float
    n_trials: int

    @property
    def gamma(self) -> float:
        """Level of the conditional step of the hybrid interval."""
        return (self.alpha - self.beta) / (1.0 - self.beta)


@dataclass(frozen=True)
class ForecastInterval:
    """Intervals for a realised out-of-sample estimate of the selected trial."""

    index: int
    alpha: float
    beta: float
    naive: Interval
    conditional: Interval
    hybrid: Interval
    oos_variance: float


def _validate(x, sigma) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    if x.ndim != 1 or x.size < 1:
        raise ValueError("x must be a non-empty one-dimensional vector of estimates")
    if sigma.shape != (x.size, x.size):
        raise ValueError(f"sigma must have shape ({x.size}, {x.size}), got {sigma.shape}")
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(sigma))):
        raise ValueError("x and sigma must be finite")
    if not np.allclose(sigma, sigma.T, rtol=1e-8, atol=1e-12 * np.abs(sigma).max()):
        raise ValueError("sigma must be symmetric")
    if np.any(np.diag(sigma) <= 0):
        raise ValueError("sigma must have a positive diagonal")
    return x, sigma


def _check_levels(alpha: float, beta: float | None) -> float:
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between 0 and 1")
    beta = alpha / 10.0 if beta is None else float(beta)
    if not 0.0 < beta < alpha:
        raise ValueError("beta must satisfy 0 < beta < alpha")
    return beta


def _frame(x: np.ndarray, sigma: np.ndarray, select: str):
    """Express the problem in the frame in which selection is ``argmax``.

    Selecting on the t-statistic ``X_j / sd_j`` is selecting on the largest estimate
    after rescaling by the (known) standard deviations.
    """
    if select == "estimate":
        return x, sigma, np.ones_like(x)
    if select == "t":
        w = np.sqrt(np.diag(sigma))
        return x / w, sigma / np.outer(w, w), w
    raise ValueError("select must be 'estimate' or 't'")


def _relative_bounds(
    xs: np.ndarray, sigma: np.ndarray, ks: np.ndarray, var_eff: np.ndarray | float
) -> tuple[np.ndarray, np.ndarray]:
    """Truncation window around the winner's estimate, ``(m_minus, m_plus)``.

    For trial ``j`` put ``s_j = Sigma_kk - Sigma_jk`` and write the statistic being
    inferred as ``D`` with variance ``var_eff``. Selection of ``k`` holds iff
    ``D >= D + m_minus`` and ``D <= D + m_plus`` with
    ``m_minus = var_eff * max_{s_j > 0} (X_j - X_k) / s_j`` and
    ``m_plus = var_eff * min_{s_j < 0} (X_j - X_k) / s_j``; for ``var_eff = Sigma_kk``
    these are ``L - X_k`` and ``U - X_k``. Trials with ``s_j = 0`` only require
    ``X_j <= X_k``, which holds; a violation beyond rounding is an input error.
    """
    r = xs.shape[0]
    rows = np.arange(r)
    skk = sigma[ks, ks]
    s = skk[:, None] - sigma[:, ks].T
    xk = xs[rows, ks]
    diff = xs - xk[:, None]
    tol = _EQUAL_RTOL * skk[:, None]
    equal = np.abs(s) <= tol
    if np.any(equal & (diff > 1e-8 * np.maximum(np.sqrt(skk), np.abs(xk))[:, None])):
        raise ValueError(
            "a trial with the same variance and perfect correlation as the winner has a "
            "larger estimate; sigma is inconsistent with x (duplicate trials with "
            "different returns?)"
        )
    safe = np.where(equal, 1.0, s)
    ratio = diff / safe
    lower = np.where(s > tol, ratio, -np.inf).max(axis=1)
    upper = np.where(s < -tol, ratio, np.inf).min(axis=1)
    var_eff = np.broadcast_to(np.asarray(var_eff, dtype=float), skk.shape)
    return var_eff * np.minimum(lower, 0.0), var_eff * np.maximum(upper, 0.0)


def selection_bounds(x, sigma, k: int | None = None) -> tuple[float, float]:
    """Truncation bounds ``(L, U)`` for ``X_k`` implied by selection of trial ``k``.

    Selection of ``k`` is equivalent to ``L(Z) <= X_k <= U(Z)`` where
    ``Z = X - Sigma[:, k] X_k / Sigma_kk`` is independent of ``X_k``. ``k`` defaults to
    the argmax of ``x``. ``L = -inf`` if no trial has a smaller covariance with the
    winner than the winner's variance, and ``U = +inf`` correspondingly.
    """
    x, sigma = _validate(x, sigma)
    k = int(np.argmax(x)) if k is None else int(k)
    lo, hi = _relative_bounds(x[None, :], sigma, np.array([k]), sigma[k, k])
    return float(x[k] + lo[0]), float(x[k] + hi[0])


def _hybrid_and_conditional(lp, up, alpha, beta, c_beta):
    """Estimators and interval ends in standardised units from window ``[lp, up]``."""
    gamma = (alpha - beta) / (1.0 - beta)
    return {
        "est_c": mean_quantile(lp, up, 0.5),
        "lo_c": mean_quantile(lp, up, 1.0 - alpha / 2.0),
        "hi_c": mean_quantile(lp, up, alpha / 2.0),
        "est_h": mean_quantile(lp, up, 0.5, c_beta),
        "lo_h": mean_quantile(lp, up, 1.0 - gamma / 2.0, c_beta),
        "hi_h": mean_quantile(lp, up, gamma / 2.0, c_beta),
    }


def infer_batch(xs, sigma, ks, *, alpha, beta, c_beta):
    """Conditional and hybrid inference for many selections sharing one ``sigma``.

    ``xs`` is ``R x K`` (already in the selection frame, so ``ks = argmax``), ``ks``
    the winner of every row. Returns arrays in the frame's units; non-finite entries
    mark root-finding failures, which callers must count rather than drop.
    """
    rows = np.arange(xs.shape[0])
    se = np.sqrt(sigma[ks, ks])
    lo, hi = _relative_bounds(xs, sigma, ks, sigma[ks, ks])
    out = _hybrid_and_conditional(lo / se, hi / se, alpha, beta, c_beta)
    xk = xs[rows, ks]
    res = {name: xk + se * val for name, val in out.items()}
    res.update({"se": se, "x": xk, "lower": xk + lo, "upper": xk + hi})
    return res


def winner_inference(
    x,
    sigma,
    alpha: float = 0.05,
    beta: float | None = None,
    *,
    select: Selection = "estimate",
    critical_values: tuple[float, float] | None = None,
    n_sim: int = 100_000,
    rng: np.random.Generator | int | None = 0,
) -> WinnerInference:
    """Selection-corrected estimate and intervals for the best of K trials.

    Parameters
    ----------
    x : (K,) array
        Estimated performance of every trial (e.g. annualised Sharpe ratios).
    sigma : (K, K) array
        Covariance matrix of the estimation error of ``x`` (same units squared).
    alpha : float
        One minus the confidence level.
    beta : float, optional
        Level of the projection step of the hybrid interval, ``0 < beta < alpha``;
        default ``alpha / 10``.
    select : {"estimate", "t"}
        Select the largest estimate (default) or the largest t-statistic ``x / sd``.
    critical_values : (float, float), optional
        Precomputed ``(c_alpha, c_beta)`` for the projection steps; by default they
        are simulated from ``sigma``.
    n_sim : int
        Draws for the simulated critical values.
    rng : Generator or int or None
        Random source for the critical values.

    Returns
    -------
    WinnerInference

    Raises
    ------
    RootFindingError
        If an inversion could not be bracketed. This never happens for valid inputs;
        it is raised rather than skipped.

    Notes
    -----
    Coverage is exact conditional on selection under normality with known ``sigma``
    (conditional), and at least ``1 - alpha`` unconditionally (hybrid); with an
    estimated ``sigma`` and non-normal returns both hold asymptotically.
    """
    x, sigma = _validate(x, sigma)
    beta = _check_levels(alpha, beta)
    xs, ss, w = _frame(x, sigma, select)
    k = int(np.argmax(xs))
    if critical_values is None:
        c_alpha, c_beta = (
            float(v) for v in projection_critical_value(ss, [alpha, beta], n_sim=n_sim, rng=rng)
        )
    else:
        c_alpha, c_beta = (float(v) for v in critical_values)
    res = infer_batch(xs[None, :], ss, np.array([k]), alpha=alpha, beta=beta, c_beta=c_beta)
    vals = {name: float(arr[0]) for name, arr in res.items()}
    if not all(np.isfinite(v) for key, v in vals.items() if key not in ("lower", "upper")):
        raise RootFindingError("truncated-normal inversion failed to bracket a root")
    wk = float(w[k])
    se = float(np.sqrt(sigma[k, k]))
    xk = float(x[k])
    z = float(stats.norm.isf(alpha / 2.0))
    z_bonf = float(stats.norm.isf(alpha / (2.0 * x.size)))
    inv = 1.0 - alpha
    result = WinnerInference(
        index=k,
        selection=select,
        alpha=alpha,
        beta=beta,
        estimate=xk,
        se=se,
        conditional_estimate=wk * vals["est_c"],
        hybrid_estimate=wk * vals["est_h"],
        naive=Interval(xk - z * se, xk + z * se, inv),
        conditional=Interval(wk * vals["lo_c"], wk * vals["hi_c"], inv),
        hybrid=Interval(wk * vals["lo_h"], wk * vals["hi_h"], inv),
        projection=Interval(xk - c_alpha * se, xk + c_alpha * se, inv),
        bonferroni=Interval(xk - z_bonf * se, xk + z_bonf * se, inv),
        truncation=(wk * vals["lower"], wk * vals["upper"]),
        critical_value_alpha=c_alpha,
        critical_value_beta=c_beta,
        n_trials=int(x.size),
    )
    if result.conditional.width > 10.0 * result.naive.width:
        warnings.warn(
            "the conditional interval is more than ten times wider than the naive one "
            "because the winner barely beats the runner-up; use the hybrid interval",
            WincurseWarning,
            stacklevel=2,
        )
    return result


def forecast_interval(
    x,
    sigma,
    oos_variance: float,
    alpha: float = 0.05,
    beta: float | None = None,
    *,
    select: Selection = "estimate",
    critical_value_beta: float | None = None,
    n_sim: int = 100_000,
    rng: np.random.Generator | int | None = 0,
) -> ForecastInterval:
    """Conditional and hybrid forecast intervals for the winner's out-of-sample estimate.

    Under the hypothesis that the selected trial's true performance is unchanged, the
    realised out-of-sample estimate ``X2`` (variance ``oos_variance``) lies in the
    returned interval with probability ``1 - alpha`` given selection (conditional) or
    unconditionally at least ``1 - alpha`` (hybrid). Translated from ``L12_fn``,
    ``U12_fn`` and the forecast-interval block of the authors' ``empirical.m``.

    A realised value outside the interval indicates decay beyond selection bias;
    a value inside is compatible with selection alone.
    """
    x, sigma = _validate(x, sigma)
    beta = _check_levels(alpha, beta)
    if not oos_variance > 0:
        raise ValueError("oos_variance must be positive")
    xs, ss, w = _frame(x, sigma, select)
    k = int(np.argmax(xs))
    wk = float(w[k])
    if critical_value_beta is None:
        c_beta = float(projection_critical_value(ss, beta, n_sim=n_sim, rng=rng))
    else:
        c_beta = float(critical_value_beta)
    var12 = ss[k, k] + oos_variance / wk**2
    s = float(np.sqrt(var12))
    lo, hi = _relative_bounds(xs[None, :], ss, np.array([k]), var12)
    gamma = (alpha - beta) / (1.0 - beta)
    lp, up = lo / s, hi / s
    u = {
        "cl": mean_quantile(lp, up, 1.0 - alpha / 2.0),
        "cu": mean_quantile(lp, up, alpha / 2.0),
        "hl": mean_quantile(lp, up, 1.0 - gamma / 2.0, c_beta),
        "hu": mean_quantile(lp, up, gamma / 2.0, c_beta),
    }
    if not all(np.all(np.isfinite(v)) for v in u.values()):
        raise RootFindingError("forecast-interval inversion failed to bracket a root")
    xk = float(x[k])
    centre = float(xs[k]) * wk
    scale = s * wk
    z = float(stats.norm.isf(alpha / 2.0))
    inv = 1.0 - alpha
    return ForecastInterval(
        index=k,
        alpha=alpha,
        beta=beta,
        naive=Interval(xk - z * scale, xk + z * scale, inv),
        conditional=Interval(
            centre + scale * float(u["cl"][0]), centre + scale * float(u["cu"][0]), inv
        ),
        hybrid=Interval(
            centre + scale * float(u["hl"][0]), centre + scale * float(u["hu"][0]), inv
        ),
        oos_variance=float(oos_variance),
    )


def winner_inference_cross(
    select_x,
    report_y,
    cov_xx,
    cov_xy,
    cov_yy,
    alpha: float = 0.05,
    beta: float | None = None,
    *,
    critical_values: tuple[float, float] | None = None,
    n_sim: int = 100_000,
    rng: np.random.Generator | int | None = 0,
) -> WinnerInference:
    """Select on one statistic, report another (AKM 2024, general ``(X, Y)`` case).

    Trial ``k = argmax(select_x)`` is chosen on the *selection* statistic ``X`` (for
    example the Sharpe ratio) and inference is made for the *reported* statistic
    ``Y_k`` (for example the Fama-French alpha) of the same trial. ``(X, Y)`` is jointly
    normal with covariance blocks ``cov_xx`` (K x K), ``cov_xy`` (K x K, entry
    ``[j, i] = Cov(X_j, Y_i)``) and ``cov_yy`` (K x K).

    Write ``X_j = Z_j + c_j Y_k`` with ``c_j = Cov(X_j, Y_k) / Var(Y_k)`` and ``Z`` independent
    of ``Y_k``. Selection of ``k`` holds iff ``(c_k - c_j) Y_k >= Z_j - Z_k`` for every ``j``,
    an interval for ``Y_k``; conditional and hybrid inference then proceed as in
    :func:`winner_inference` with the truncated normal for ``Y_k``. With ``X = Y`` this is
    exactly :func:`winner_inference`. The projection critical values use the covariance of
    the reported statistics.
    """
    x = np.asarray(select_x, dtype=float)
    y = np.asarray(report_y, dtype=float)
    cxx, cxy, cyy = (np.atleast_2d(np.asarray(m, dtype=float)) for m in (cov_xx, cov_xy, cov_yy))
    n = x.size
    if y.shape != x.shape or any(m.shape != (n, n) for m in (cxx, cxy, cyy)):
        raise ValueError("select_x, report_y and the three covariance blocks must be K-sized")
    _validate(y, cyy)
    _validate(x, cxx)
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(cxy))):
        raise ValueError("inputs must be finite")
    beta = _check_levels(alpha, beta)
    k = int(np.argmax(x))
    var_k = float(cyy[k, k])
    sd = float(np.sqrt(var_k))
    c = cxy[:, k] / var_k
    g = c[k] - c
    d = x - x[k]
    tol = _EQUAL_RTOL * (abs(c[k]) + np.abs(c).max() + 1e-300)
    equal = np.abs(g) <= tol
    equal[k] = True
    safe = np.where(equal, 1.0, g)
    ratio = d / safe
    lo = float(np.where(g > tol, ratio, -np.inf).max()) if n > 1 else -np.inf
    hi = float(np.where(g < -tol, ratio, np.inf).min()) if n > 1 else np.inf
    lo, hi = min(lo, 0.0), max(hi, 0.0)
    if critical_values is None:
        c_alpha, c_beta = (
            float(v) for v in projection_critical_value(cyy, [alpha, beta], n_sim=n_sim, rng=rng)
        )
    else:
        c_alpha, c_beta = (float(v) for v in critical_values)
    u = _hybrid_and_conditional(lo / sd, hi / sd, alpha, beta, c_beta)
    yk = float(y[k])
    vals = {name: yk + sd * float(v) for name, v in u.items()}
    if not all(np.isfinite(v) for v in vals.values()):
        raise RootFindingError("truncated-normal inversion failed to bracket a root")
    z = float(stats.norm.isf(alpha / 2.0))
    z_bonf = float(stats.norm.isf(alpha / (2.0 * n)))
    inv = 1.0 - alpha
    return WinnerInference(
        index=k,
        selection="cross",
        alpha=alpha,
        beta=beta,
        estimate=yk,
        se=sd,
        conditional_estimate=vals["est_c"],
        hybrid_estimate=vals["est_h"],
        naive=Interval(yk - z * sd, yk + z * sd, inv),
        conditional=Interval(vals["lo_c"], vals["hi_c"], inv),
        hybrid=Interval(vals["lo_h"], vals["hi_h"], inv),
        projection=Interval(yk - c_alpha * sd, yk + c_alpha * sd, inv),
        bonferroni=Interval(yk - z_bonf * sd, yk + z_bonf * sd, inv),
        truncation=(yk + lo, yk + hi),
        critical_value_alpha=c_alpha,
        critical_value_beta=c_beta,
        n_trials=n,
    )
