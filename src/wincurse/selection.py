"""Other selection rules: report the top ``m`` trials, or every trial that passes a threshold.

*Threshold selection* ("published only if the t-statistic exceeds 2"): trial ``j`` is
reported iff ``X_j > c sd_j``. Given that event, ``X_j`` is ``N(mu_j, Sigma_jj)`` truncated
to ``[c sd_j, inf)`` and inverting its CDF in ``mu_j`` gives a median-unbiased estimate and an
equal-tailed interval with exact coverage conditional on the trial being reported.

*Top-m selection*: trial ``k`` is reported iff fewer than ``m`` trials have a larger
estimate. With ``Z = X - Sigma[:, k] X_k / Sigma_kk`` independent of ``X_k``, the estimate of
trial ``j`` is the line ``Z_j + (Sigma_jk / Sigma_kk) x`` of ``X_k = x``; the event is a union of
intervals in ``x`` (one interval for ``m = 1``), and ``X_k`` given the event and ``Z`` is normal
truncated to that union. The CDF of a truncated normal is monotone in its mean for any
truncation set, so the same inversion applies. Both rules give conditional inference
only; the hybrid construction of the winner case is not extended here.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
from scipy import optimize, special, stats

from .exceptions import RootFindingError
from .truncnorm import mean_quantile
from .winners import Interval, _check_levels, _validate

__all__ = [
    "SelectedInference",
    "threshold_inference",
    "top_m_inference",
    "top_m_windows",
]

_MAX_EXPAND = 80
_EQUAL_RTOL = 1e-12


@dataclass(frozen=True)
class SelectedInference:
    """Conditional inference for one reported trial.

    ``windows`` lists the intervals (in the units of the estimates) to which the trial's
    estimate is confined by the selection event given the other estimates.
    """

    index: int
    estimate: float
    se: float
    conditional_estimate: float
    naive: Interval
    conditional: Interval
    windows: tuple[tuple[float, float], ...]


def threshold_inference(
    x, sigma, threshold: float = 2.0, alpha: float = 0.05
) -> list[SelectedInference]:
    """Conditional inference for every trial whose t-statistic exceeds ``threshold``.

    Parameters
    ----------
    x : (K,) array
        Estimates of the trials.
    sigma : (K, K) array
        Covariance of the estimation errors (only the diagonal is used: the selection event
        of one trial involves that trial alone).
    threshold : float
        Trial ``j`` is reported iff ``x_j / sd_j > threshold``.
    alpha : float
        One minus the confidence level.

    Returns
    -------
    list of SelectedInference
        One entry per reported trial, ordered by index; empty if none passes.
    """
    x, sigma = _validate(x, sigma)
    _check_levels(alpha, None)
    sd = np.sqrt(np.diag(sigma))
    z = float(stats.norm.isf(alpha / 2.0))
    out = []
    for j in np.flatnonzero(x / sd > threshold):
        lp = float(threshold - x[j] / sd[j])
        lo, med, hi = (
            float(v) for v in mean_quantile(lp, np.inf, [1.0 - alpha / 2.0, 0.5, alpha / 2.0])
        )
        if not np.all(np.isfinite([lo, med, hi])):
            raise RootFindingError("threshold inversion failed to bracket a root")
        s = float(sd[j])
        out.append(
            SelectedInference(
                index=int(j),
                estimate=float(x[j]),
                se=s,
                conditional_estimate=float(x[j] + s * med),
                naive=Interval(float(x[j] - z * s), float(x[j] + z * s), 1.0 - alpha),
                conditional=Interval(float(x[j] + s * lo), float(x[j] + s * hi), 1.0 - alpha),
                windows=((float(threshold * s), np.inf),),
            )
        )
    return out


def top_m_windows(x, sigma, k: int, m: int) -> list[tuple[float, float]]:
    """Intervals of ``X_k`` (given ``Z``) on which trial ``k`` is among the ``m`` largest."""
    x, sigma = _validate(x, sigma)
    k = int(k)
    skk = sigma[k, k]
    z = x - sigma[:, k] / skk * x[k]
    g = 1.0 - sigma[:, k] / skk
    tol = _EQUAL_RTOL
    others = np.array([j for j in range(x.size) if j != k])
    gj, zj = g[others], z[others]
    flat = np.abs(gj) <= tol
    constant_beats = int(np.sum(flat & (zj > 0)))
    moving = ~flat
    breaks = np.sort(np.unique(zj[moving] / gj[moving]))
    edges = np.concatenate([[-np.inf], breaks, [np.inf]])
    segments: list[tuple[float, float]] = []
    for lo, hi in itertools.pairwise(edges):
        if np.isinf(lo) and np.isinf(hi):
            probe = 0.0
        elif np.isinf(lo):
            probe = hi - 1.0
        elif np.isinf(hi):
            probe = lo + 1.0
        else:
            probe = 0.5 * (lo + hi)
        beats = constant_beats + int(np.sum(moving & (zj + (1.0 - gj) * probe > probe)))
        if beats <= m - 1:
            if segments and segments[-1][1] == lo:
                segments[-1] = (segments[-1][0], hi)
            else:
                segments.append((float(lo), float(hi)))
    return segments


def _log_mass(za: np.ndarray, zb: np.ndarray) -> np.ndarray:
    """``log(Phi(zb) - Phi(za))`` for ``za < zb``, accurate in both tails."""
    za, zb = np.broadcast_arrays(np.asarray(za, float), np.asarray(zb, float))
    out = np.empty(za.shape)
    upper = za > 0
    lower = ~upper & (zb < 0)
    mid = ~(upper | lower)
    with np.errstate(divide="ignore", invalid="ignore"):
        qa, qb = special.log_ndtr(-za[upper]), special.log_ndtr(-zb[upper])
        out[upper] = qa + np.log1p(-np.exp(qb - qa))
        pa, pb = special.log_ndtr(za[lower]), special.log_ndtr(zb[lower])
        out[lower] = pb + np.log1p(-np.exp(pa - pb))
        out[mid] = np.log(special.ndtr(zb[mid]) - special.ndtr(za[mid]))
    return out


def _union_cdf(obs: float, mean: float, sd: float, windows: list[tuple[float, float]]) -> float:
    """CDF at ``obs`` of ``N(mean, sd^2)`` truncated to the union of ``windows``."""
    a = (np.array([w[0] for w in windows]) - mean) / sd
    b = (np.array([w[1] for w in windows]) - mean) / sd
    full = _log_mass(a, b)
    clipped = np.minimum(b, (obs - mean) / sd)
    keep = clipped > a
    if not keep.any():
        return 0.0
    part = _log_mass(a[keep], clipped[keep])
    return float(np.exp(special.logsumexp(part) - special.logsumexp(full)))


def _union_quantile(obs: float, sd: float, windows, p: float) -> float:
    """Mean ``mu`` for which the truncated-normal CDF at ``obs`` equals ``p`` (decreasing in mu)."""

    def f(u: float) -> float:
        return _union_cdf(obs, obs + sd * u, sd, windows) - p

    lo, hi = -8.0, 8.0
    for _ in range(_MAX_EXPAND):
        if f(lo) >= 0.0:
            break
        lo *= 2.0
    for _ in range(_MAX_EXPAND):
        if f(hi) <= 0.0:
            break
        hi *= 2.0
    if f(lo) < 0.0 or f(hi) > 0.0:
        raise RootFindingError("top-m inversion failed to bracket a root")
    u = optimize.brentq(f, lo, hi, xtol=1e-12, rtol=4 * np.finfo(float).eps, maxiter=500)
    return obs + sd * u


def top_m_inference(x, sigma, m: int, alpha: float = 0.05) -> list[SelectedInference]:
    """Conditional inference for each of the ``m`` trials with the largest estimates.

    Every reported trial is treated on its own: the interval of trial ``k`` has exact
    coverage conditional on ``k`` being among the top ``m`` (and on ``Z^{(k)}``). For
    ``m = 1`` the result equals the conditional inference of :func:`winner_inference`.

    Returns
    -------
    list of SelectedInference
        Ordered from the largest estimate down.
    """
    x, sigma = _validate(x, sigma)
    _check_levels(alpha, None)
    if not 1 <= m <= x.size:
        raise ValueError("m must lie between 1 and the number of trials")
    z = float(stats.norm.isf(alpha / 2.0))
    out = []
    for k in np.argsort(x)[::-1][:m]:
        k = int(k)
        sd = float(np.sqrt(sigma[k, k]))
        windows = top_m_windows(x, sigma, k, m)
        if not any(lo <= x[k] <= hi for lo, hi in windows):
            raise ValueError("sigma is inconsistent with x: the trial is not in the top m")
        out.append(
            SelectedInference(
                index=k,
                estimate=float(x[k]),
                se=sd,
                conditional_estimate=_union_quantile(float(x[k]), sd, windows, 0.5),
                naive=Interval(float(x[k] - z * sd), float(x[k] + z * sd), 1.0 - alpha),
                conditional=Interval(
                    _union_quantile(float(x[k]), sd, windows, 1.0 - alpha / 2.0),
                    _union_quantile(float(x[k]), sd, windows, alpha / 2.0),
                    1.0 - alpha,
                ),
                windows=tuple(windows),
            )
        )
    return out
