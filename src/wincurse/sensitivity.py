"""Sensitivity of a winner's inference to trials that were run but not logged.

Two different questions arise and only one has an exact answer.

*Projection interval (identified).* The projection interval for the winner is valid
whatever rule picked it among a larger set, so the effect of ``H`` unlogged trials
follows from recomputing its critical value for the covariance matrix augmented by
the hidden trials. This needs an assumed covariance structure and nothing else: here
hidden trials are noisy variants of logged ones (see :func:`augmented_covariance`).

*Conditional and hybrid intervals (not identified).* The truncation bounds depend on
the realised values and the true means of the hidden trials, which are unobserved.
:func:`unlogged_scenario` therefore only offers a labelled scenario simulation under
stated assumptions; its output is a range of what the intervals could have been, not
a valid confidence interval.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import linalg

from .critical import projection_critical_value
from .winners import _validate, infer_batch

__all__ = [
    "ProjectionSensitivity",
    "ScenarioSummary",
    "augmented_covariance",
    "nearest_neighbour_correlation",
    "unlogged_projection",
    "unlogged_scenario",
]


def nearest_neighbour_correlation(sigma: np.ndarray) -> float:
    """Mean over trials of their largest correlation with another trial."""
    sigma = np.atleast_2d(np.asarray(sigma, dtype=float))
    if sigma.shape[0] < 2:
        return 0.0
    sd = np.sqrt(np.diag(sigma))
    corr = sigma / np.outer(sd, sd)
    np.fill_diagonal(corr, -np.inf)
    return float(np.clip(corr.max(axis=1).mean(), 0.0, 0.999))


def augmented_covariance(
    sigma: np.ndarray, n_hidden: int, correlation: float, seed: int = 0
) -> np.ndarray:
    """Covariance of ``K`` logged plus ``H`` hidden trials.

    Hidden trials are noisy variants of logged ones: trial ``m`` is a standardised copy
    of a logged trial ``i_m`` (cycling through a random permutation of the logged
    trials, fixed by ``seed``) mixed with independent noise so that its correlation with
    its source is ``correlation``. It inherits the correlations of the source with every
    other logged trial, scaled by ``correlation``. The matrix is positive semi-definite by
    construction for any logged covariance. Hidden trials have the median logged variance.

    ``correlation = 0`` gives hidden trials independent of everything; values close to one
    give near-duplicates that hardly change the effective number of trials.
    """
    sigma = np.atleast_2d(np.asarray(sigma, dtype=float))
    if n_hidden < 0:
        raise ValueError("n_hidden must be non-negative")
    if not 0.0 <= correlation < 1.0:
        raise ValueError("correlation must lie in [0, 1)")
    k = sigma.shape[0]
    full = np.empty((k + n_hidden, k + n_hidden))
    full[:k, :k] = sigma
    if n_hidden == 0:
        return full
    sd = np.sqrt(np.diag(sigma))
    corr = sigma / np.outer(sd, sd)
    sd_h = float(np.median(sd))
    sources = np.resize(np.random.default_rng(seed).permutation(k), n_hidden)
    cross = correlation * corr[sources]
    full[:k, k:] = sd[:, None] * sd_h * cross.T
    full[k:, :k] = full[:k, k:].T
    hh = correlation**2 * corr[np.ix_(sources, sources)]
    np.fill_diagonal(hh, 1.0)
    full[k:, k:] = sd_h**2 * hh
    return full


@dataclass(frozen=True)
class ProjectionSensitivity:
    """Projection interval for the winner when ``n_hidden`` trials were not logged."""

    n_hidden: int
    critical_value: float
    lower: float
    upper: float
    width_ratio: float


def unlogged_projection(
    x,
    sigma,
    hidden_counts,
    correlation: float,
    alpha: float = 0.05,
    *,
    n_sim: int = 100_000,
    rng: np.random.Generator | int | None = 0,
) -> list[ProjectionSensitivity]:
    """Exact projection intervals for the winner for several numbers of hidden trials.

    ``width_ratio`` compares each width with the one for the logged trials alone
    (``hidden_counts`` should therefore start with 0 or the ratio is relative to the
    first entry).
    """
    x, sigma = _validate(x, sigma)
    k = int(np.argmax(x))
    se = float(np.sqrt(sigma[k, k]))
    out: list[ProjectionSensitivity] = []
    base = None
    for h in hidden_counts:
        aug = augmented_covariance(sigma, int(h), correlation)
        c = float(projection_critical_value(aug, alpha, n_sim=n_sim, rng=rng))
        base = c if base is None else base
        out.append(ProjectionSensitivity(int(h), c, x[k] - c * se, x[k] + c * se, c / base))
    return out


@dataclass(frozen=True)
class ScenarioSummary:
    """Range of conditional and hybrid results over simulated hidden-trial outcomes.

    A scenario simulation under stated assumptions, not a confidence interval.
    ``acceptance`` is the share of draws in which the logged winner is still the best
    of all trials; the summary is conditional on that event.
    """

    n_hidden: int
    correlation: float
    true_value: float
    acceptance: float
    n_accepted: int
    hybrid_estimate: tuple[float, float, float]
    hybrid_lower: tuple[float, float, float]
    hybrid_upper: tuple[float, float, float]
    conditional_estimate: tuple[float, float, float]
    conditional_lower: tuple[float, float, float]
    conditional_upper: tuple[float, float, float]


def _quantiles(values: np.ndarray) -> tuple[float, float, float]:
    q = np.nanquantile(values, [0.05, 0.5, 0.95])
    return float(q[0]), float(q[1]), float(q[2])


def unlogged_scenario(
    x,
    sigma,
    n_hidden: int,
    correlation: float,
    true_value: float = 0.0,
    alpha: float = 0.05,
    beta: float | None = None,
    *,
    n_draws: int = 400,
    n_sim: int = 20_000,
    rng: np.random.Generator | int | None = 0,
) -> ScenarioSummary:
    """Simulate how ``H`` unlogged trials could change the conditional and hybrid results.

    Scenario assumption: all logged and hidden trials share the true value
    ``true_value`` (the least-separated configuration, where selection bias is largest).
    Given the logged estimates this fixes the logged estimation errors; the hidden
    trials' estimates are drawn from their conditional Gaussian distribution, draws in
    which a hidden trial beats the logged winner are discarded (the winner would have
    been a different trial) and the acceptance rate is reported. For each accepted draw
    the conditional and hybrid inference is run on the augmented problem.
    """
    x, sigma = _validate(x, sigma)
    if n_hidden < 1:
        raise ValueError("n_hidden must be at least 1")
    beta = alpha / 10.0 if beta is None else beta
    rng = np.random.default_rng(rng)
    k_obs = x.size
    winner = int(np.argmax(x))
    full = augmented_covariance(sigma, n_hidden, correlation)
    s_oo = sigma + 1e-10 * np.trace(sigma) / k_obs * np.eye(k_obs)
    s_ho = full[k_obs:, :k_obs]
    gain = linalg.solve(s_oo, s_ho.T, assume_a="pos").T
    cond_mean = true_value + gain @ (x - true_value)
    cond_cov = full[k_obs:, k_obs:] - gain @ s_ho.T
    vals, vecs = np.linalg.eigh(0.5 * (cond_cov + cond_cov.T))
    factor = vecs * np.sqrt(np.maximum(vals, 0.0))
    accepted: list[np.ndarray] = []
    drawn = 0
    while sum(len(a) for a in accepted) < n_draws and drawn < 200 * n_draws:
        z = rng.standard_normal((n_draws, n_hidden))
        hidden = cond_mean + z @ factor.T
        drawn += n_draws
        keep = hidden.max(axis=1) <= x[winner]
        if keep.any():
            accepted.append(hidden[keep])
    n_acc = sum(len(a) for a in accepted)
    if n_acc == 0:
        raise ValueError(
            "the logged winner is essentially never the best of all trials under this "
            "scenario; lower the number of hidden trials or their assumed mean"
        )
    hidden = np.vstack(accepted)[:n_draws]
    xs_full = np.hstack([np.broadcast_to(x, (len(hidden), k_obs)), hidden])
    c_beta = float(projection_critical_value(full, beta, n_sim=n_sim, rng=rng))
    ks = np.full(len(hidden), winner)
    res = infer_batch(xs_full, full, ks, alpha=alpha, beta=beta, c_beta=c_beta)
    return ScenarioSummary(
        n_hidden=int(n_hidden),
        correlation=float(correlation),
        true_value=float(true_value),
        acceptance=float(n_acc / drawn),
        n_accepted=len(hidden),
        hybrid_estimate=_quantiles(res["est_h"]),
        hybrid_lower=_quantiles(res["lo_h"]),
        hybrid_upper=_quantiles(res["hi_h"]),
        conditional_estimate=_quantiles(res["est_c"]),
        conditional_lower=_quantiles(res["lo_c"]),
        conditional_upper=_quantiles(res["hi_c"]),
    )
