"""Diagnostics that qualify a winner-inference result."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["Diagnostics", "effective_number_of_trials", "winner_margin"]


def effective_number_of_trials(sigma: np.ndarray) -> float:
    """Eigenvalue-based effective number of independent trials.

    The participation ratio ``(sum lambda)^2 / sum lambda^2`` of the eigenvalues of the
    correlation matrix: ``K`` for uncorrelated trials, ``1`` if all are perfectly
    correlated.
    """
    sigma = np.atleast_2d(np.asarray(sigma, dtype=float))
    sd = np.sqrt(np.diag(sigma))
    corr = sigma / np.outer(sd, sd)
    vals = np.clip(np.linalg.eigvalsh(0.5 * (corr + corr.T)), 0.0, None)
    return float(vals.sum() ** 2 / np.sum(vals**2))


def winner_margin(x: np.ndarray, sigma: np.ndarray, k: int | None = None) -> tuple[float, int]:
    """Smallest studentised gap between the winner and any other trial.

    Returns ``min_j (X_k - X_j) / sqrt(Var(X_k - X_j))`` and the index of the closest
    rival. Small values mean the winner barely wins, selection matters most and the
    conditional interval becomes very wide.
    """
    x = np.asarray(x, dtype=float)
    sigma = np.atleast_2d(np.asarray(sigma, dtype=float))
    k = int(np.argmax(x)) if k is None else int(k)
    if x.size == 1:
        return float("inf"), k
    var = np.diag(sigma) + sigma[k, k] - 2.0 * sigma[:, k]
    gap = x[k] - x
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(var > 1e-12 * sigma[k, k], gap / np.sqrt(np.maximum(var, 0.0)), np.inf)
    t[k] = np.inf
    j = int(np.argmin(t))
    return float(t[j]), j


@dataclass(frozen=True)
class Diagnostics:
    """Summary diagnostics of a selection problem."""

    margin: float
    runner_up: int
    gap_in_se: float
    effective_trials: float
    n_trials: int
    conditional_to_naive_width: float


def diagnose(x, sigma, k: int, conditional_width: float, naive_width: float) -> Diagnostics:
    """Collect margin, effective trials and the conditional-to-naive width ratio."""
    x = np.asarray(x, dtype=float)
    sigma = np.atleast_2d(np.asarray(sigma, dtype=float))
    margin, runner = winner_margin(x, sigma, k)
    order = np.argsort(x)[::-1]
    second = order[1] if x.size > 1 else k
    gap = float((x[k] - x[second]) / np.sqrt(sigma[k, k])) if x.size > 1 else float("inf")
    return Diagnostics(
        margin=margin,
        runner_up=runner,
        gap_in_se=gap,
        effective_trials=effective_number_of_trials(sigma),
        n_trials=int(x.size),
        conditional_to_naive_width=float(conditional_width / naive_width),
    )
