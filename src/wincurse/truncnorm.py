"""Truncated-normal distribution functions that stay accurate in extreme tails.

Conditional inference on winners (Andrews, Kitagawa and McCloskey 2024) needs the
CDF of a normal variable truncated to an interval ``[lower, upper]`` and its
inversion in the mean. When the winner barely wins, the interval sits many
standard deviations away from the mean and a naive ``Phi(b) - Phi(a)`` underflows
to zero. All tail probabilities are therefore evaluated as ratios of survival
functions in log space, using the scaled complementary error function
(``erfcx``), which is exact to machine precision for arbitrarily large arguments.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.special import erfcx, ndtr

from .exceptions import RootFindingError

__all__ = ["mean_quantile", "tn_cdf", "tn_ppf", "tn_sf"]

_SQRT2 = np.sqrt(2.0)
_MAX_EXPAND = 80
_TINY_WIDTH = 1e-9


def _log_sf_ratio(t: np.ndarray, a: np.ndarray) -> np.ndarray:
    """``log Q(t) - log Q(a)`` for ``t >= a >= 0``, ``Q`` the normal survival function.

    Uses ``log Q(t) = -t^2/2 + log(erfcx(t/sqrt(2))/2)`` so that the large terms
    ``t^2/2`` and ``a^2/2`` are combined exactly as ``(t - a)(t + a)/2``.
    """
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        out = -(t - a) * (t + a) / 2 + np.log(erfcx(t / _SQRT2)) - np.log(erfcx(a / _SQRT2))
    return np.where(np.isposinf(t), -np.inf, out)


def _std_tn(z, a, b) -> tuple[np.ndarray, np.ndarray]:
    """CDF and survival function at ``z`` of a standard normal truncated to ``[a, b]``."""
    z, a, b = np.broadcast_arrays(*(np.asarray(v, dtype=float) for v in (z, a, b)))
    if np.any(np.isnan(z)) or np.any(np.isnan(a)) or np.any(np.isnan(b)):
        raise ValueError("arguments must not contain NaN")
    if np.any(a >= b) or np.any(np.isposinf(a)) or np.any(np.isneginf(b)):
        raise ValueError("truncation interval must satisfy lower < upper")

    flip = b <= 0.0
    zz = np.clip(np.where(flip, -z, z), np.where(flip, -b, a), np.where(flip, -a, b))
    aa = np.where(flip, -b, a)
    bb = np.where(flip, -a, b)
    upper_tail = aa >= 0.0

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        # Both branches are evaluated on safe dummy inputs where they do not apply.
        a_u = np.where(upper_tail, aa, 0.0)
        z_u = np.where(upper_tail, zz, 0.0)
        b_u = np.where(upper_tail, bb, 1.0)
        d_z = _log_sf_ratio(z_u, a_u)
        d_b = _log_sf_ratio(b_u, a_u)
        den_u = -np.expm1(d_b)
        cdf_u = -np.expm1(d_z) / den_u
        sf_u = (np.exp(d_z) - np.exp(d_b)) / den_u

        a_s = np.where(upper_tail, -1.0, aa)
        z_s = np.where(upper_tail, 0.0, zz)
        b_s = np.where(upper_tail, 1.0, bb)
        den_s = ndtr(b_s) - ndtr(a_s)
        cdf_s = (ndtr(z_s) - ndtr(a_s)) / den_s
        sf_s = (ndtr(-z_s) - ndtr(-b_s)) / den_s
        thin = (b_s - a_s) < _TINY_WIDTH
        cdf_s = np.where(thin, (z_s - a_s) / (b_s - a_s), cdf_s)
        sf_s = np.where(thin, (b_s - z_s) / (b_s - a_s), sf_s)

    cdf = np.clip(np.where(upper_tail, cdf_u, cdf_s), 0.0, 1.0)
    sf = np.clip(np.where(upper_tail, sf_u, sf_s), 0.0, 1.0)
    return np.where(flip, sf, cdf), np.where(flip, cdf, sf)


def _standardise(x, mu, sigma, lower, upper):
    sigma = np.asarray(sigma, dtype=float)
    if np.any(sigma <= 0) or not np.all(np.isfinite(sigma)):
        raise ValueError("sigma must be positive and finite")
    mu = np.asarray(mu, dtype=float)
    return (
        (np.asarray(x, dtype=float) - mu) / sigma,
        (np.asarray(lower, dtype=float) - mu) / sigma,
        (np.asarray(upper, dtype=float) - mu) / sigma,
    )


def tn_cdf(x, mu, sigma, lower=-np.inf, upper=np.inf):
    """CDF of ``N(mu, sigma^2)`` truncated to ``[lower, upper]``, evaluated at ``x``.

    Arguments broadcast. Accurate when the interval lies hundreds of standard
    deviations from ``mu``, where the textbook formula returns ``nan``.
    """
    z, a, b = _standardise(x, mu, sigma, lower, upper)
    return _std_tn(z, a, b)[0]


def tn_sf(x, mu, sigma, lower=-np.inf, upper=np.inf):
    """Survival function ``1 - CDF`` of the truncated normal, accurate in the upper tail."""
    z, a, b = _standardise(x, mu, sigma, lower, upper)
    return _std_tn(z, a, b)[1]


def tn_ppf(p, mu, sigma, lower=-np.inf, upper=np.inf):
    """Quantile function of ``N(mu, sigma^2)`` truncated to ``[lower, upper]``.

    Inverts the monotone CDF (or survival function, for ``p > 1/2``) with Brent's
    method, so the result inherits the tail accuracy of :func:`tn_cdf`.
    """
    p, mu, sigma, lower, upper = np.broadcast_arrays(
        *(np.asarray(v, dtype=float) for v in (p, mu, sigma, lower, upper))
    )
    if np.any((p < 0) | (p > 1)):
        raise ValueError("p must lie in [0, 1]")
    out = np.empty(p.shape)
    for idx in np.ndindex(p.shape):
        out[idx] = mu[idx] + sigma[idx] * _std_tn_ppf_scalar(
            float(p[idx]),
            (float(lower[idx]) - float(mu[idx])) / float(sigma[idx]),
            (float(upper[idx]) - float(mu[idx])) / float(sigma[idx]),
        )
    return out if out.ndim else float(out)


def _std_tn_ppf_scalar(p: float, a: float, b: float) -> float:
    if p <= 0.0:
        return a
    if p >= 1.0:
        return b
    # Measure the offset d from the edge nearest the bulk of the mass so that the
    # solution keeps full relative precision however far the window is from zero.
    if a >= 0.0:
        origin, sign = a, 1.0
    elif b <= 0.0:
        origin, sign = b, -1.0
    else:
        origin, sign = 0.0, 1.0
    use_cdf = p <= 0.5
    target = p if use_cdf else 1.0 - p
    increasing = use_cdf == (sign > 0)

    def f(d: float) -> float:
        cdf, sf = _std_tn(origin + sign * d, a, b)
        value = float(cdf if use_cdf else sf)
        return value - target if increasing else target - value

    straddle = a < 0.0 < b
    if straddle:
        lo = a if np.isfinite(a) else -8.0
        hi = b if np.isfinite(b) else 8.0
        for _ in range(_MAX_EXPAND):
            if f(lo) <= 0.0:
                break
            lo *= 2.0
        for _ in range(_MAX_EXPAND):
            if f(hi) >= 0.0:
                break
            hi *= 2.0
    else:
        lo = 0.0
        hi = b - a if np.isfinite(a) and np.isfinite(b) else 1.0
        for _ in range(_MAX_EXPAND):
            if f(hi) >= 0.0:
                break
            hi *= 2.0
    if f(lo) > 0.0 or f(hi) < 0.0:
        raise RootFindingError("quantile could not be bracketed")
    d = brentq(f, lo, hi, xtol=1e-300, rtol=4 * np.finfo(float).eps, maxiter=500)
    return origin + sign * d


def _cdf_at_observation(u, lp, up, c):
    """CDF at 0 of ``N(u, 1)`` truncated to ``[max(lp, u - c), min(up, u + c)]``."""
    a = np.maximum(lp - u, -c)
    b = np.minimum(up - u, c)
    return _std_tn(-u, a, b)[0]


def mean_quantile(lp, up, p, c=np.inf, *, max_expand: int = _MAX_EXPAND, tol: float = 1e-13):
    """Mean ``u`` at which a truncated normal places mass ``p`` below the observation.

    Standardised so that the observation sits at 0 and the standard deviation is 1.
    The truncation window around the observation is ``[lp, up]`` (``lp <= 0 <= up``,
    possibly infinite). If ``c`` is finite the window is additionally intersected
    with ``[u - c, u + c]``, the hybrid construction of AKM Section 4. The
    CDF at the observation is strictly decreasing in ``u``, so the root of
    ``F(u) = p`` is unique; it is bracketed adaptively and found by bisection.

    Parameters
    ----------
    lp, up : array_like
        Window ends relative to the observation, in standard-deviation units.
    p : array_like
        Target probability in ``(0, 1)``.
    c : float
        Half-width of the hybrid projection window; ``inf`` for conditional inference.
    max_expand : int
        Maximum number of bracket doublings before an element is reported as failed.

    Returns
    -------
    numpy.ndarray
        The solution, ``nan`` where bracketing failed. Callers count those cases.
    """
    lp, up, p = np.broadcast_arrays(*(np.asarray(v, dtype=float) for v in (lp, up, p)))
    if np.any((p <= 0) | (p >= 1)):
        raise ValueError("p must lie strictly between 0 and 1")
    if np.any(lp > 0) or np.any(up < 0):
        raise ValueError("the window must contain the observation (lp <= 0 <= up)")
    if not c > 0:
        raise ValueError("c must be positive")

    def excess(u):
        return _cdf_at_observation(u, lp, up, c) - p

    ok = np.ones(p.shape, dtype=bool)
    if np.isfinite(c):
        lo = np.full(p.shape, -c)
        hi = np.full(p.shape, c)
    else:
        lo = np.full(p.shape, -8.0)
        hi = np.full(p.shape, 8.0)
        for _ in range(max_expand):
            need = excess(lo) < 0
            if not need.any():
                break
            lo = np.where(need, 2 * lo, lo)
        ok &= excess(lo) >= 0
        for _ in range(max_expand):
            need = excess(hi) > 0
            if not need.any():
                break
            hi = np.where(need, 2 * hi, hi)
        ok &= excess(hi) <= 0
    for _ in range(300):
        mid = 0.5 * (lo + hi)
        above = excess(mid) > 0
        lo = np.where(above, mid, lo)
        hi = np.where(above, hi, mid)
        if np.all((hi - lo) <= tol * np.maximum(1.0, np.abs(mid))):
            break
    out = np.where(ok, 0.5 * (lo + hi), np.nan)
    return out if out.ndim else float(out)
