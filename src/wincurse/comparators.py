"""Established selection-bias adjustments reported next to the corrected interval.

* Deflated Sharpe Ratio (Bailey and Lopez de Prado 2014): the probability that the
  true Sharpe ratio exceeds the expected maximum Sharpe ratio of ``N`` unskilled
  trials, with a correction for skewness, kurtosis and sample length.
* Harvey and Liu (2015) haircut: the Sharpe ratio is turned into a p-value, the
  p-value is adjusted for multiple testing (Bonferroni, Holm or
  Benjamini-Hochberg-Yekutieli) and mapped back to a Sharpe ratio.

For the best of ``K`` trials the Holm-adjusted p-value equals the Bonferroni one,
because the winner has the smallest p-value; the two are not independent evidence.
These tools test whether the winner differs from zero; none of them gives a
corrected estimate or an interval for the winner's true Sharpe ratio.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

__all__ = [
    "ArchTests",
    "Haircut",
    "deflated_sharpe_ratio",
    "expected_max_sharpe",
    "harvey_liu_haircut",
    "multiple_testing_pvalues",
    "spa_stepm",
]

_EULER_GAMMA = 0.5772156649015329


def expected_max_sharpe(n_trials: float, sharpe_variance: float) -> float:
    """Expected maximum Sharpe ratio of ``n_trials`` independent trials with true Sharpe zero.

    ``SR0 = sqrt(V) * ((1 - g) Phi^-1(1 - 1/N) + g Phi^-1(1 - 1/(N e)))`` with ``g`` the
    Euler-Mascheroni constant (Bailey and Lopez de Prado 2014, eq. 2).
    ``sharpe_variance`` is the cross-trial variance of the estimated Sharpe ratios in
    the same units as the Sharpe ratio being tested (per period if the latter is).
    """
    if n_trials < 1:
        raise ValueError("n_trials must be at least 1")
    if sharpe_variance < 0:
        raise ValueError("sharpe_variance must be non-negative")
    if n_trials == 1:
        return 0.0
    ppf = stats.norm.ppf
    g = _EULER_GAMMA
    return float(
        np.sqrt(sharpe_variance)
        * ((1.0 - g) * ppf(1.0 - 1.0 / n_trials) + g * ppf(1.0 - 1.0 / (n_trials * np.e)))
    )


def deflated_sharpe_ratio(
    sharpe: float,
    n_obs: int,
    n_trials: float,
    sharpe_variance: float,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Deflated Sharpe Ratio: probability that the true Sharpe ratio exceeds ``SR0``.

    ``DSR = Phi( (SR - SR0) sqrt(T - 1) / sqrt(1 - g3 SR + (g4 - 1) SR^2 / 4) )``
    with ``SR`` the per-period Sharpe ratio of the selected trial, ``T`` the number of
    observations, ``g3`` the skewness and ``g4`` the (non-excess) kurtosis of its
    returns, and ``SR0`` from :func:`expected_max_sharpe`.
    """
    if n_obs < 2:
        raise ValueError("n_obs must be at least 2")
    sr0 = expected_max_sharpe(n_trials, sharpe_variance)
    var_term = 1.0 - skewness * sharpe + (kurtosis - 1.0) / 4.0 * sharpe**2
    if var_term <= 0:
        raise ValueError("the Sharpe-ratio variance term is not positive; check skew and kurtosis")
    return float(stats.norm.cdf((sharpe - sr0) * np.sqrt(n_obs - 1.0) / np.sqrt(var_term)))


def multiple_testing_pvalues(p: np.ndarray, method: str) -> np.ndarray:
    """Adjusted p-values for ``bonferroni``, ``holm`` or ``bhy`` (step-up with harmonic factor)."""
    p = np.asarray(p, dtype=float)
    m = p.size
    order = np.argsort(p)
    ranked = p[order]
    if method == "bonferroni":
        adj = np.minimum(1.0, m * ranked)
    elif method == "holm":
        adj = np.minimum(1.0, np.maximum.accumulate((m - np.arange(m)) * ranked))
    elif method == "bhy":
        c_m = np.sum(1.0 / np.arange(1, m + 1))
        raw = np.minimum(1.0, m * c_m * ranked / np.arange(1, m + 1))
        adj = np.minimum.accumulate(raw[::-1])[::-1]
    else:
        raise ValueError("method must be 'bonferroni', 'holm' or 'bhy'")
    out = np.empty(m)
    out[order] = adj
    return out


@dataclass(frozen=True)
class Haircut:
    """Harvey-Liu haircut of the winner's Sharpe ratio for one adjustment method."""

    method: str
    p_value: float
    adjusted_p_value: float
    adjusted_sharpe: float
    haircut: float


def harvey_liu_haircut(
    sharpes: np.ndarray, n_obs: int, winner: int | None = None, periods_per_year: float = 1.0
) -> dict[str, Haircut]:
    """Harvey-Liu (2015) haircuts of the winner's Sharpe ratio.

    Each trial's t-statistic is ``SR_per_period * sqrt(T)``, its two-sided p-value comes
    from the standard normal, and the winner's p-value is adjusted with all ``K``
    p-values. The adjusted Sharpe ratio is the one whose p-value equals the adjusted
    p-value; the haircut is ``1 - SR_adjusted / SR``. Returns Bonferroni and BHY
    (Holm coincides with Bonferroni for the winner and is omitted).

    Parameters
    ----------
    sharpes : (K,) array
        Annualised Sharpe ratios.
    n_obs : int
        Number of return observations ``T``.
    winner : int, optional
        Index of the winner (default: the largest Sharpe ratio).
    periods_per_year : float
        Used to convert annualised Sharpe ratios to per-period ones.
    """
    sr = np.asarray(sharpes, dtype=float)
    k = int(np.argmax(sr)) if winner is None else int(winner)
    q = float(periods_per_year)
    tstat = sr / np.sqrt(q) * np.sqrt(n_obs)
    p = 2.0 * stats.norm.sf(np.abs(tstat))
    out = {}
    for method in ("bonferroni", "bhy"):
        adj = float(multiple_testing_pvalues(p, method)[k])
        t_adj = float(stats.norm.isf(adj / 2.0)) if adj < 1.0 else 0.0
        sr_adj = np.sign(sr[k]) * t_adj / np.sqrt(n_obs) * np.sqrt(q)
        out[method] = Haircut(
            method=method,
            p_value=float(p[k]),
            adjusted_p_value=adj,
            adjusted_sharpe=float(sr_adj),
            haircut=float(1.0 - sr_adj / sr[k]) if sr[k] != 0 else float("nan"),
        )
    return out


@dataclass(frozen=True)
class ArchTests:
    """Hansen SPA and Romano-Wolf StepM of the strategies against a zero benchmark.

    ``spa_pvalue`` tests "no strategy has a positive mean"; ``superior`` lists the
    strategies that StepM identifies as having a positive mean at the chosen size.
    Both answer whether something beats zero, not what the winner is worth.
    """

    spa_pvalue: float
    superior: list[str]
    winner_is_superior: bool
    size: float
    reps: int


def spa_stepm(
    returns,
    winner: int,
    names: list[str] | None = None,
    *,
    size: float = 0.05,
    reps: int = 1000,
    block_size: int | None = None,
    seed: int = 0,
) -> ArchTests:
    """Run Hansen's SPA and the Romano-Wolf StepM with the ``arch`` package.

    Losses are negative returns and the benchmark is a zero-return position, so a
    strategy "beats the benchmark" when its mean return is positive. Requires the
    optional dependency ``arch`` (``pip install wincurse[arch]``).
    """
    try:
        from arch.bootstrap import SPA, StepM
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "spa_stepm needs the optional dependency: pip install wincurse[arch]"
        ) from exc
    values = np.asarray(returns, dtype=float)
    t = values.shape[0]
    block = int(np.sqrt(t)) if block_size is None else int(block_size)
    losses = -values
    bench = np.zeros(t)
    spa = SPA(bench, losses, block_size=block, reps=reps, seed=seed)
    spa.compute()
    step = StepM(bench, losses, size=size, block_size=block, reps=reps, seed=seed)
    step.compute()
    labels = names if names is not None else [str(i) for i in range(values.shape[1])]
    idx = [int(i) for i in step.superior_models]
    superior = [labels[i] for i in idx]
    return ArchTests(
        spa_pvalue=float(spa.pvalues["consistent"]),
        superior=superior,
        winner_is_superior=int(winner) in idx,
        size=size,
        reps=reps,
    )
