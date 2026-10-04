"""Performance statistics of K strategies and their joint sampling covariance.

The winner-inference procedures take an estimate vector ``X`` and a covariance
matrix ``Sigma`` of the estimation error. For a ``T x K`` matrix of returns this
module produces both, for the mean return or the Sharpe ratio, using the delta
method on the moments ``(E r, E r^2)``:

* ``"normal"``: the iid-normal formulas of Lo (2002) and Memmel (2003);
* ``"iid"``: the sample covariance of the influence functions (robust to skewness
  and excess kurtosis);
* ``"hac"``: Newey-West long-run covariance of the influence functions with the
  automatic Bartlett bandwidth of Newey and West (1994);
* ``"bootstrap"``: the stationary bootstrap of Politis and Romano (1994).

All covariances are computed per period and annualised at the end: if ``X`` is
multiplied by ``sqrt(q)`` then ``Sigma`` is multiplied by ``q`` (``q`` periods per
year), which leaves every interval equivariant.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from .exceptions import WincurseWarning

__all__ = [
    "JointEstimate",
    "PerformanceEstimate",
    "alpha_influence_functions",
    "estimate_joint",
    "estimate_performance",
    "influence_functions",
    "ledoit_wolf_constant_correlation",
    "long_run_covariance",
    "nearest_psd",
    "newey_west_bandwidth",
    "sharpe_ratios",
    "stationary_bootstrap_covariance",
]

Statistic = Literal["sharpe", "mean"]
CovMethod = Literal["hac", "iid", "normal", "bootstrap"]


def _as_matrix(returns) -> tuple[np.ndarray, list[str], pd.Index | None]:
    """Validate a ``T x K`` return panel and return values, names and the row index."""
    index = None
    if isinstance(returns, pd.DataFrame):
        names = [str(c) for c in returns.columns]
        index = returns.index
        values = returns.to_numpy(dtype=float)
    elif isinstance(returns, pd.Series):
        names = [str(returns.name) if returns.name is not None else "strategy_0"]
        index = returns.index
        values = returns.to_numpy(dtype=float)[:, None]
    else:
        values = np.asarray(returns, dtype=float)
        if values.ndim == 1:
            values = values[:, None]
        names = [f"strategy_{i}" for i in range(values.shape[1])] if values.ndim == 2 else []
    if values.ndim != 2:
        raise ValueError("returns must be a T x K matrix")
    t, k = values.shape
    if t < 8:
        raise ValueError(f"need at least 8 observations per strategy, got {t}")
    if k < 1:
        raise ValueError("returns must contain at least one strategy")
    if not np.all(np.isfinite(values)):
        raise ValueError("returns contain NaN or infinite values; drop or fill them first")
    if np.any(np.ptp(values, axis=0) == 0):
        bad = [names[i] for i in np.flatnonzero(np.ptp(values, axis=0) == 0)[:5]]
        raise ValueError(f"constant return series have no Sharpe ratio: {bad}")
    return values, names, index


def sharpe_ratios(returns, periods_per_year: float = 1.0) -> np.ndarray:
    """Annualised Sharpe ratio ``sqrt(q) * mean / std`` of each column (``ddof=1``).

    ``returns`` are per-period excess returns; ``periods_per_year`` is ``q``.
    """
    values, _, _ = _as_matrix(returns)
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")
    return np.sqrt(periods_per_year) * values.mean(axis=0) / values.std(axis=0, ddof=1)


def influence_functions(values: np.ndarray, statistic: Statistic = "sharpe") -> np.ndarray:
    """Per-observation influence functions of the chosen statistic, shape ``T x K``.

    For the Sharpe ratio ``SR = m1 / sqrt(m2 - m1^2)`` the delta method on the
    moments ``m1 = E r`` and ``m2 = E r^2`` has gradient
    ``(m2 / s^3, -m1 / (2 s^3))`` with ``s^2 = m2 - m1^2``; the influence function
    is ``gradient . (r - m1, r^2 - m2)``. Columns have zero sample mean.
    """
    m1 = values.mean(axis=0)
    if statistic == "mean":
        return values - m1
    if statistic != "sharpe":
        raise ValueError("statistic must be 'sharpe' or 'mean'")
    m2 = (values**2).mean(axis=0)
    s3 = (m2 - m1**2) ** 1.5
    return (m2 / s3) * (values - m1) - (m1 / (2 * s3)) * (values**2 - m2)


def newey_west_bandwidth(psi: np.ndarray) -> int:
    """Automatic Bartlett lag truncation of Newey and West (1994).

    The pilot autocovariances are summed over the K influence-function columns
    (trace), so strategies of opposite sign cannot cancel each other.
    """
    t = psi.shape[0]
    n = int(np.floor(4 * (t / 100) ** (2 / 9)))
    n = max(1, min(n, t - 2))
    sigma = np.array([np.sum(psi[j:] * psi[: t - j]) / t for j in range(n + 1)])
    j = np.arange(1, n + 1)
    s0 = sigma[0] + 2 * np.sum(sigma[1:])
    s1 = 2 * np.sum(j * sigma[1:])
    if s0 <= 0:
        return 0
    gamma = 1.1447 * (s1 / s0) ** (2 / 3) if s1 > 0 else 0.0
    return int(min(np.floor(gamma * t ** (1 / 3)), t - 2))


def long_run_covariance(
    psi: np.ndarray, method: Literal["hac", "iid"] = "hac", bandwidth: int | None = None
) -> np.ndarray:
    """Long-run covariance of the columns of ``psi`` (per observation, not divided by ``T``).

    ``"iid"`` is the sample covariance. ``"hac"`` adds Bartlett-weighted
    autocovariances up to ``bandwidth`` lags (``None``: Newey-West 1994). The
    kernel sum is evaluated as ``psi' W psi`` with a banded Toeplitz ``W``, which
    costs a single matrix product and is positive semi-definite by construction.
    """
    t = psi.shape[0]
    if method == "iid":
        return psi.T @ psi / t
    if method != "hac":
        raise ValueError("method must be 'hac' or 'iid'")
    m = newey_west_bandwidth(psi) if bandwidth is None else int(bandwidth)
    if m < 0 or m >= t:
        raise ValueError("bandwidth must satisfy 0 <= bandwidth < T")
    smoothed = psi.copy()
    for lag in range(1, m + 1):
        weight = 1.0 - lag / (m + 1)
        smoothed[lag:] += weight * psi[: t - lag]
        smoothed[: t - lag] += weight * psi[lag:]
    omega = psi.T @ smoothed / t
    return 0.5 * (omega + omega.T)


def stationary_bootstrap_covariance(
    values: np.ndarray,
    statistic: Statistic = "sharpe",
    *,
    n_boot: int = 1000,
    mean_block_length: float | None = None,
    rng: np.random.Generator | int | None = 0,
) -> np.ndarray:
    """Covariance of the statistic over stationary-bootstrap resamples (Politis-Romano 1994).

    Blocks start at uniformly random positions and have geometric length with
    mean ``mean_block_length`` (default ``T^(1/3)``). Resampled moments are
    obtained from row-count vectors, so the cost is two matrix products.
    """
    rng = np.random.default_rng(rng)
    t = values.shape[0]
    b = float(t ** (1 / 3)) if mean_block_length is None else float(mean_block_length)
    if b < 1:
        raise ValueError("mean_block_length must be at least 1")
    new = rng.random((n_boot, t)) < 1.0 / b
    new[:, 0] = True
    starts = rng.integers(0, t, size=(n_boot, t))
    pos = np.arange(t)
    last_new = np.maximum.accumulate(np.where(new, pos, 0), axis=1)
    idx = (np.take_along_axis(starts, last_new, axis=1) + (pos - last_new)) % t
    counts = np.zeros((n_boot, t))
    rows = np.repeat(np.arange(n_boot), t)
    np.add.at(counts, (rows, idx.ravel()), 1.0)
    m1 = counts @ values / t
    if statistic == "mean":
        draws = m1
    else:
        m2 = counts @ values**2 / t
        draws = m1 / np.sqrt(np.maximum(m2 - m1**2, np.finfo(float).tiny))
    return np.atleast_2d(np.cov(draws, rowvar=False)) * t


def _normal_sharpe_covariance(values: np.ndarray) -> np.ndarray:
    """Lo (2002) / Memmel (2003): ``Cov(SR_i, SR_j) = (rho_ij + SR_i SR_j rho_ij^2 / 2)``."""
    sr = values.mean(axis=0) / values.std(axis=0, ddof=1)
    rho = np.atleast_2d(np.corrcoef(values, rowvar=False))
    return rho + np.outer(sr, sr) * rho**2 / 2


def nearest_psd(matrix: np.ndarray, floor: float = 1e-10) -> tuple[np.ndarray, bool]:
    """Repair a symmetric matrix to be positive definite, keeping its diagonal.

    Eigenvalues below ``floor * lambda_max`` are raised to that level and the result
    is rescaled so the variances are unchanged. Returns the matrix and whether any
    eigenvalue had to be changed.
    """
    sym = 0.5 * (matrix + matrix.T)
    vals = np.linalg.eigvalsh(sym)
    lo = floor * vals[-1]
    if vals[0] >= lo:
        return sym, False
    vals, vecs = np.linalg.eigh(sym)
    repaired = (vecs * np.maximum(vals, lo)) @ vecs.T
    scale = np.sqrt(np.diag(sym) / np.diag(repaired))
    repaired = repaired * np.outer(scale, scale)
    return 0.5 * (repaired + repaired.T), True


def ledoit_wolf_constant_correlation(
    data: np.ndarray, sample: np.ndarray | None = None
) -> tuple[np.ndarray, float]:
    """Ledoit-Wolf (2003) shrinkage of a covariance matrix towards constant correlation.

    ``data`` is the ``T x K`` panel (here: influence functions) from which the
    optimal intensity is estimated. The target keeps every variance and replaces
    all correlations by their average. ``sample`` is the matrix to shrink
    (default: the sample covariance of ``data``), which lets a HAC matrix use the
    intensity derived for the iid case. Returns the shrunk matrix and the intensity.
    """
    t, n = data.shape
    x = data - data.mean(axis=0)
    s = x.T @ x / t
    target_source = s if sample is None else sample
    if n == 1:
        return target_source.copy(), 0.0
    var = np.diag(s)
    sd = np.sqrt(var)
    r_bar = (np.sum(s / np.outer(sd, sd)) - n) / (n * (n - 1))
    y = x**2
    phi_mat = y.T @ y / t - 2 * (x.T @ x) * s / t + s**2
    phi = phi_mat.sum()
    term1 = (x**3).T @ x / t
    h = x.T @ x / t
    h_diag = np.diag(h)
    theta = term1 - h_diag[:, None] * s - h * var[:, None] + var[:, None] * s
    np.fill_diagonal(theta, 0.0)
    rho = np.trace(phi_mat) + r_bar * np.sum(np.outer(1 / sd, sd) * theta)
    gamma = np.sum((s - _constant_correlation_target(s, r_bar)) ** 2)
    kappa = (phi - rho) / gamma if gamma > 0 else 1.0
    delta = float(max(0.0, min(1.0, kappa / t)))
    ts = np.sqrt(np.diag(target_source))
    r_ts = (np.sum(target_source / np.outer(ts, ts)) - n) / (n * (n - 1))
    target = _constant_correlation_target(target_source, r_ts)
    return delta * target + (1 - delta) * target_source, delta


def _constant_correlation_target(s: np.ndarray, r_bar: float) -> np.ndarray:
    sd = np.sqrt(np.diag(s))
    target = r_bar * np.outer(sd, sd)
    np.fill_diagonal(target, np.diag(s))
    return target


@dataclass(frozen=True)
class PerformanceEstimate:
    """Estimates of K performance statistics with their joint covariance.

    ``values`` and ``cov`` are in annualised units (``cov`` is the covariance of the
    estimation error of ``values``). ``per_period_*`` hold the unannualised versions.
    """

    names: list[str]
    values: np.ndarray
    cov: np.ndarray
    n_obs: int
    periods_per_year: float
    statistic: str
    cov_method: str
    shrinkage: float | None = None
    bandwidth: int | None = None
    repaired: bool = False
    warnings: list[str] = field(default_factory=list)
    scale: float = 1.0

    @property
    def n_strategies(self) -> int:
        return len(self.names)

    @property
    def per_period_values(self) -> np.ndarray:
        return self.values / self.scale

    @property
    def per_period_cov(self) -> np.ndarray:
        return self.cov / self.scale**2


def estimate_performance(
    returns,
    periods_per_year: float = 1.0,
    *,
    statistic: Statistic = "sharpe",
    cov_method: CovMethod = "hac",
    shrinkage: bool = False,
    bandwidth: int | None = None,
    n_boot: int = 1000,
    mean_block_length: float | None = None,
    rng: np.random.Generator | int | None = 0,
) -> PerformanceEstimate:
    """Estimate a performance statistic of every column and its error covariance.

    Parameters
    ----------
    returns : array_like or DataFrame
        ``T x K`` per-period (excess) returns in decimals.
    periods_per_year : float
        ``q``; estimates are scaled by ``sqrt(q)`` and covariances by ``q`` for the
        Sharpe ratio, and by ``q`` and ``q**2`` for the mean.
    statistic : {"sharpe", "mean"}
    cov_method : {"hac", "iid", "normal", "bootstrap"}
        See the module docstring. ``"normal"`` is available for the Sharpe ratio only.
    shrinkage : bool
        Shrink the covariance towards a constant-correlation target (Ledoit-Wolf
        2003 intensity estimated on the influence functions). Changes correlations,
        hence truncation bounds and critical values; use when ``K`` is large
        relative to ``T`` and compare with the unshrunk result.
    bandwidth : int, optional
        HAC lag truncation; default is the Newey-West (1994) rule.
    n_boot, mean_block_length, rng
        Stationary-bootstrap settings.

    Returns
    -------
    PerformanceEstimate
    """
    values, names, _ = _as_matrix(returns)
    t, k = values.shape
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")
    if statistic not in ("sharpe", "mean"):
        raise ValueError("statistic must be 'sharpe' or 'mean'")
    if cov_method not in ("hac", "iid", "normal", "bootstrap"):
        raise ValueError("cov_method must be 'hac', 'iid', 'normal' or 'bootstrap'")
    if cov_method == "normal" and statistic != "sharpe":
        raise ValueError("cov_method='normal' is only defined for the Sharpe ratio")
    notes: list[str] = []
    mean = values.mean(axis=0)
    est = mean / values.std(axis=0, ddof=1) if statistic == "sharpe" else mean
    psi = influence_functions(values, statistic)
    used_bw = None
    delta = None
    if cov_method == "normal":
        omega = _normal_sharpe_covariance(values)
    elif cov_method == "bootstrap":
        omega = stationary_bootstrap_covariance(
            values, statistic, n_boot=n_boot, mean_block_length=mean_block_length, rng=rng
        )
    elif cov_method == "hac":
        used_bw = newey_west_bandwidth(psi) if bandwidth is None else int(bandwidth)
        omega = long_run_covariance(psi, "hac", used_bw)
    else:
        omega = long_run_covariance(psi, "iid")
    omega = np.atleast_2d(omega)

    if shrinkage:
        omega, delta = ledoit_wolf_constant_correlation(psi, omega)
        notes.append(
            f"covariance shrunk towards constant correlation (intensity {delta:.3f}); "
            "correlations drive truncation bounds and critical values, compare with the "
            "unshrunk result"
        )
    if k >= t:
        notes.append(
            f"K={k} >= T={t}: the sample covariance is rank-deficient; "
            "consider shrinkage or fewer trials"
        )
    omega, repaired = nearest_psd(omega)
    if repaired:
        notes.append("covariance was not positive definite and has been repaired")
    for note in notes:
        warnings.warn(note, WincurseWarning, stacklevel=2)

    q = float(periods_per_year)
    scale = np.sqrt(q) if statistic == "sharpe" else q
    return PerformanceEstimate(
        names=names,
        values=est * scale,
        cov=omega / t * scale**2,
        n_obs=t,
        periods_per_year=q,
        statistic=statistic,
        cov_method=cov_method,
        shrinkage=delta,
        bandwidth=used_bw,
        repaired=repaired,
        warnings=notes,
        scale=float(scale),
    )


def alpha_influence_functions(
    values: np.ndarray, factors: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Per-period regression intercepts and their influence functions.

    ``values`` is ``T x K`` (excess returns) and ``factors`` ``T x p`` (factor returns). For
    the OLS intercept ``a_i`` of ``r_it = a_i + b_i' f_t + e_it`` the influence function is
    ``h_t e_it`` with ``h_t = e_1' (E[w w'])^{-1} w_t`` and ``w_t = (1, f_t)``.
    """
    t = values.shape[0]
    w = np.column_stack([np.ones(t), factors])
    gram_inv = np.linalg.inv(w.T @ w / t)
    beta = gram_inv @ (w.T @ values) / t
    resid = values - w @ beta
    h = w @ gram_inv[0]
    return beta[0], h[:, None] * resid


@dataclass(frozen=True)
class JointEstimate:
    """Selection statistic ``X`` and reported statistic ``Y`` of K trials with joint covariance.

    ``select`` (annualised Sharpe ratios) and ``report`` (annualised alphas) come with the
    covariance blocks of their estimation errors: ``cov_xx``, ``cov_xy[j, i] = Cov(X_j, Y_i)``
    and ``cov_yy``.
    """

    names: list[str]
    select: np.ndarray
    report: np.ndarray
    cov_xx: np.ndarray
    cov_xy: np.ndarray
    cov_yy: np.ndarray
    n_obs: int
    periods_per_year: float
    cov_method: str
    repaired: bool = False
    warnings: list[str] = field(default_factory=list)


def estimate_joint(
    returns,
    factors,
    periods_per_year: float = 1.0,
    *,
    cov_method: Literal["hac", "iid"] = "hac",
    bandwidth: int | None = None,
) -> JointEstimate:
    """Sharpe ratios (selection) and factor-model alphas (reporting) with their joint covariance.

    The alpha is the intercept of the regression of each trial's excess return on the
    ``factors`` (``T x p``, same rows as ``returns``), annualised by ``q``; the Sharpe ratio is
    annualised by ``sqrt(q)``. The joint covariance of the ``2K`` estimates comes from the
    Newey-West (``"hac"``) or sample (``"iid"``) covariance of the stacked influence
    functions, rescaled consistently (``sqrt(q)`` for Sharpe ratios, ``q`` for alphas).
    """
    values, names, _ = _as_matrix(returns)
    f = np.asarray(factors, dtype=float)
    if f.ndim == 1:
        f = f[:, None]
    if f.shape[0] != values.shape[0]:
        raise ValueError("factors and returns must have the same number of rows")
    if not np.all(np.isfinite(f)):
        raise ValueError("factors contain NaN or infinite values")
    if cov_method not in ("hac", "iid"):
        raise ValueError("cov_method must be 'hac' or 'iid' for joint estimates")
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")
    t, k = values.shape
    q = float(periods_per_year)
    sharpe = values.mean(axis=0) / values.std(axis=0, ddof=1)
    alpha_hat, psi_alpha = alpha_influence_functions(values, f)
    psi = np.hstack([influence_functions(values, "sharpe"), psi_alpha])
    if cov_method == "hac":
        omega = long_run_covariance(psi, "hac", bandwidth)
    else:
        omega = long_run_covariance(psi, "iid")
    omega, repaired = nearest_psd(omega)
    scale = np.concatenate([np.full(k, np.sqrt(q)), np.full(k, q)])
    cov = omega / t * np.outer(scale, scale)
    notes = ["joint covariance was not positive definite and has been repaired"] if repaired else []
    for note in notes:
        warnings.warn(note, WincurseWarning, stacklevel=2)
    return JointEstimate(
        names=names,
        select=sharpe * np.sqrt(q),
        report=alpha_hat * q,
        cov_xx=cov[:k, :k],
        cov_xy=cov[:k, k:],
        cov_yy=cov[k:, k:],
        n_obs=t,
        periods_per_year=q,
        cov_method=cov_method,
        repaired=repaired,
        warnings=notes,
    )
