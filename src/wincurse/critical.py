"""Critical values for projection (simultaneous) confidence sets.

The projection interval for the selected mean uses the ``1 - level`` quantile of
``max_j |xi_j| / sqrt(Sigma_jj)`` with ``xi ~ N(0, Sigma)``. AKM (2024, Section 4)
obtain it by simulation; for independent or equicorrelated trials there are exact
one-dimensional formulas, used here as references and for large simulation studies.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy import integrate, optimize, stats
from scipy.special import ndtr

__all__ = [
    "equicorrelated_critical_value",
    "independent_critical_value",
    "projection_critical_value",
]


def _check_level(level: float) -> None:
    if not 0.0 < level < 1.0:
        raise ValueError("level must lie strictly between 0 and 1")


def _max_abs_t(sigma: np.ndarray, n_sim: int, rng, chunk: int) -> np.ndarray:
    sigma = np.atleast_2d(np.asarray(sigma, dtype=float))
    k = sigma.shape[0]
    sd = np.sqrt(np.diag(sigma))
    corr = sigma / np.outer(sd, sd)
    corr = 0.5 * (corr + corr.T)
    try:
        factor = np.linalg.cholesky(corr)
    except np.linalg.LinAlgError:
        vals, vecs = np.linalg.eigh(corr)
        factor = vecs * np.sqrt(np.maximum(vals, 0.0))
    rng = np.random.default_rng(rng)
    maxima = np.empty(n_sim)
    for start in range(0, n_sim, chunk):
        stop = min(start + chunk, n_sim)
        z = rng.standard_normal((stop - start, k))
        maxima[start:stop] = np.abs(z @ factor.T).max(axis=1)
    return np.sort(maxima)


def projection_critical_value(
    sigma: np.ndarray,
    level: float | Sequence[float],
    *,
    n_sim: int = 100_000,
    rng: np.random.Generator | int | None = 0,
    chunk: int = 4096,
):
    """Simulated ``1 - level`` quantile of ``max_j |xi_j| / sqrt(Sigma_jj)``, ``xi ~ N(0, Sigma)``.

    Follows the authors' replication code: ``n_sim`` multivariate-normal draws and the
    order statistic of rank ``round(n_sim * (1 - level))``. Monte Carlo error of the
    quantile shrinks like ``1 / sqrt(n_sim * level)``; the default gives about
    ``0.01`` for ``level = 0.005``. A sequence of levels is served from one set of
    draws.

    Parameters
    ----------
    sigma : (K, K) array
        Covariance matrix (positive semi-definite).
    level : float or sequence of float
        Tail probabilities, e.g. ``beta`` or ``alpha``.
    n_sim : int
        Number of draws.
    rng : Generator or int or None
        Random source.

    Returns
    -------
    float or numpy.ndarray
        One critical value per requested level.
    """
    levels = np.atleast_1d(np.asarray(level, dtype=float))
    for lev in levels:
        _check_level(float(lev))
    maxima = _max_abs_t(sigma, n_sim, rng, chunk)
    ranks = np.clip(np.rint(n_sim * (1.0 - levels)).astype(int), 1, n_sim)
    values = maxima[ranks - 1]
    return float(values[0]) if np.ndim(level) == 0 else values


def independent_critical_value(k: int, level: float) -> float:
    """Exact critical value for ``K`` independent trials: ``(2 Phi(c) - 1)^K = 1 - level``."""
    _check_level(level)
    return float(stats.norm.isf((1.0 - (1.0 - level) ** (1.0 / k)) / 2.0))


def equicorrelated_critical_value(k: int, rho: float, level: float) -> float:
    """Exact critical value for ``K`` trials with common correlation ``rho >= 0``.

    With ``xi_j = sqrt(rho) z + sqrt(1 - rho) e_j`` the joint probability is a
    one-dimensional integral over the common factor ``z``:
    ``P(max_j |xi_j| <= c) = E_z [ (Phi((c - sqrt(rho) z) / s) - Phi((-c - sqrt(rho) z) / s))^K ]``
    with ``s = sqrt(1 - rho)``.
    """
    _check_level(level)
    if not 0.0 <= rho < 1.0:
        raise ValueError("rho must lie in [0, 1)")
    if rho == 0.0 or k == 1:
        return independent_critical_value(k, level)
    s = np.sqrt(1.0 - rho)
    r = np.sqrt(rho)

    def coverage(c: float) -> float:
        def integrand(z: float) -> float:
            inside = ndtr((c - r * z) / s) - ndtr((-c - r * z) / s)
            return float(stats.norm.pdf(z) * inside**k)

        value, _ = integrate.quad(integrand, -10, 10, limit=400, epsabs=1e-13, epsrel=1e-13)
        return value

    lo = float(stats.norm.isf(level / 2.0))
    hi = float(stats.norm.isf(level / (2.0 * k))) + 1e-9
    return float(optimize.brentq(lambda c: coverage(c) - (1.0 - level), lo, hi, xtol=1e-12))
