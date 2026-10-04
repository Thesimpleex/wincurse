"""Monte Carlo evaluation of winner-inference procedures.

Every replication draws ``X ~ N(mu, Sigma)``, selects the largest estimate and runs the
naive, conditional, hybrid, projection and Bonferroni procedures on it. Failures of the
truncated-normal inversion are never dropped: a replication whose interval could not
be computed is counted in ``n_failed`` and scored as *not covering*, so coverage is
always computed over all ``n_rep`` replications.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
from scipy import stats
from scipy.signal import lfilter

from .critical import projection_critical_value
from .performance import estimate_performance
from .winners import _check_levels, _frame, infer_batch

__all__ = [
    "CoverageResult",
    "feasible_replications",
    "normal_sharpe_covariance_true",
    "replication_metrics",
    "simulate_feasible",
    "simulate_known_sigma",
    "simulate_returns",
    "summarise_replications",
]

_AR1_PHI = 0.2
Distribution = Literal["normal", "t5", "garch", "ar1"]


@dataclass(frozen=True)
class CoverageResult:
    """Coverage, bias and width of each procedure over ``n_rep`` replications.

    ``median_bias`` is the median of ``estimate - mu_k``; ``prob_bias`` is
    ``P(estimate > mu_k) - 1/2`` (zero for a median-unbiased estimator). Widths are
    medians over replications with a finite interval.
    """

    n_rep: int
    n_failed: int
    n_trials: int
    naive_coverage: float
    conditional_coverage: float
    hybrid_coverage: float
    projection_coverage: float
    bonferroni_coverage: float
    naive_mean_bias: float
    naive_median_bias: float
    conditional_median_bias: float
    hybrid_median_bias: float
    naive_prob_bias: float
    conditional_prob_bias: float
    hybrid_prob_bias: float
    naive_median_width: float
    conditional_median_width: float
    hybrid_median_width: float
    projection_median_width: float
    mean_selected_mu: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def _cover(lo: np.ndarray, hi: np.ndarray, mu: np.ndarray) -> np.ndarray:
    finite = np.isfinite(lo) & np.isfinite(hi)
    return finite & (lo <= mu) & (mu <= hi)


def replication_metrics(
    mu_k, x, se, c_alpha, est_c, lo_c, hi_c, est_h, lo_h, hi_h, *, alpha: float, n_trials: int
) -> CoverageResult:
    """Summarise per-replication results; non-finite endpoints count as misses.

    All arguments are arrays over replications (``c_alpha`` may be a scalar): the true
    mean of the selected trial, its estimate and standard error, the projection critical
    value, and the conditional and hybrid estimates and interval ends.
    """
    mu_k, x, se, est_c, lo_c, hi_c, est_h, lo_h, hi_h = (
        np.asarray(v, dtype=float) for v in (mu_k, x, se, est_c, lo_c, hi_c, est_h, lo_h, hi_h)
    )
    c_alpha = np.broadcast_to(np.asarray(c_alpha, dtype=float), x.shape)
    n = x.size
    z = stats.norm.isf(alpha / 2.0)
    z_bonf = stats.norm.isf(alpha / (2.0 * n_trials))
    failed_c = ~(np.isfinite(lo_c) & np.isfinite(hi_c) & np.isfinite(est_c))
    failed_h = ~(np.isfinite(lo_h) & np.isfinite(hi_h) & np.isfinite(est_h))
    failed = failed_c | failed_h

    def med(values: np.ndarray, mask: np.ndarray) -> float:
        return float(np.median(values[mask])) if mask.any() else float("nan")

    return CoverageResult(
        n_rep=n,
        n_failed=int(failed.sum()),
        n_trials=n_trials,
        naive_coverage=float(np.mean(np.abs(x - mu_k) <= z * se)),
        conditional_coverage=float(np.mean(_cover(lo_c, hi_c, mu_k))),
        hybrid_coverage=float(np.mean(_cover(lo_h, hi_h, mu_k))),
        projection_coverage=float(np.mean(np.abs(x - mu_k) <= c_alpha * se)),
        bonferroni_coverage=float(np.mean(np.abs(x - mu_k) <= z_bonf * se)),
        naive_mean_bias=float(np.mean(x - mu_k)),
        naive_median_bias=float(np.median(x - mu_k)),
        conditional_median_bias=med(est_c - mu_k, ~failed_c),
        hybrid_median_bias=med(est_h - mu_k, ~failed_h),
        naive_prob_bias=float(np.mean(x > mu_k) - 0.5),
        conditional_prob_bias=float(np.mean(est_c[~failed_c] > mu_k[~failed_c]) - 0.5)
        if (~failed_c).any()
        else float("nan"),
        hybrid_prob_bias=float(np.mean(est_h[~failed_h] > mu_k[~failed_h]) - 0.5)
        if (~failed_h).any()
        else float("nan"),
        naive_median_width=float(np.median(2 * z * se)),
        conditional_median_width=med(hi_c - lo_c, ~failed_c),
        hybrid_median_width=med(hi_h - lo_h, ~failed_h),
        projection_median_width=float(np.median(2 * c_alpha * se)),
        mean_selected_mu=float(np.mean(mu_k)),
    )


def simulate_known_sigma(
    mu,
    sigma,
    n_rep: int = 2000,
    alpha: float = 0.05,
    beta: float | None = None,
    *,
    select: str = "estimate",
    critical_values: tuple[float, float] | None = None,
    n_sim: int = 100_000,
    rng: np.random.Generator | int | None = 0,
    chunk: int = 500,
) -> CoverageResult:
    """Coverage study with known covariance: ``X ~ N(mu, sigma)``, ``n_rep`` replications.

    Parameters
    ----------
    mu : (K,) array
        True means of the trials.
    sigma : (K, K) array
        Covariance of the estimates (known to the procedures).
    critical_values : (float, float), optional
        ``(c_alpha, c_beta)``; default: simulated from ``sigma`` with ``n_sim`` draws.
    chunk : int
        Replications processed at once (memory control only).
    """
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    beta = _check_levels(alpha, beta)
    rng = np.random.default_rng(rng)
    vals, vecs = np.linalg.eigh(0.5 * (sigma + sigma.T))
    factor = vecs * np.sqrt(np.maximum(vals, 0.0))
    _, ss, w = _frame(mu, sigma, select)
    if critical_values is None:
        c_alpha, c_beta = (
            float(v) for v in projection_critical_value(ss, [alpha, beta], n_sim=n_sim, rng=rng)
        )
    else:
        c_alpha, c_beta = (float(v) for v in critical_values)
    keys = ("mu_k", "x", "se", "est_c", "lo_c", "hi_c", "est_h", "lo_h", "hi_h")
    store: dict[str, list[np.ndarray]] = {key: [] for key in keys}
    for start in range(0, n_rep, chunk):
        r = min(chunk, n_rep - start)
        x = mu + rng.standard_normal((r, mu.size)) @ factor.T
        xs = x / w if select == "t" else x
        ks = np.argmax(xs, axis=1)
        res = infer_batch(xs, ss, ks, alpha=alpha, beta=beta, c_beta=c_beta)
        wk = w[ks]
        rows = np.arange(r)
        se = np.sqrt(sigma[ks, ks])
        store["mu_k"].append(mu[ks])
        store["x"].append(x[rows, ks])
        store["se"].append(se)
        for name in ("est_c", "lo_c", "hi_c", "est_h", "lo_h", "hi_h"):
            store[name].append(wk * res[name])
    cat = {key: np.concatenate(parts) for key, parts in store.items()}
    return replication_metrics(
        cat["mu_k"],
        cat["x"],
        cat["se"],
        c_alpha,
        cat["est_c"],
        cat["lo_c"],
        cat["hi_c"],
        cat["est_h"],
        cat["lo_h"],
        cat["hi_h"],
        alpha=alpha,
        n_trials=mu.size,
    )


def simulate_returns(
    rng: np.random.Generator,
    n_obs: int,
    n_strategies: int,
    rho: float,
    annual_sharpe: float | np.ndarray = 0.0,
    periods_per_year: float = 252.0,
    *,
    dist: Distribution = "normal",
    vol: float = 0.01,
) -> np.ndarray:
    """One ``T x K`` panel from a one-factor equicorrelation model.

    ``r_it = mu_i + vol * s_t * (sqrt(rho) f_t + sqrt(1 - rho) e_it)`` with unit-variance
    innovations. ``dist`` selects the innovations and ``s_t``: ``"normal"``; ``"t5"``
    (Student-t with 5 degrees of freedom, rescaled to unit variance); ``"garch"``
    (normal innovations, a market-wide GARCH(1,1) volatility ``s_t^2 = h_t`` with
    ``h_t = 1 - a - b + a * (s_{t-1} f_{t-1})^2 + b h_{t-1}``, ``a = 0.1``, ``b = 0.88``,
    so that every strategy's variance is ``vol^2`` on average); ``"ar1"`` (normal, every
    factor and idiosyncratic series follows an AR(1) with coefficient 0.2 and unit variance,
    so the long-run variance of the mean exceeds the iid one by a factor 1.5). The true
    annualised Sharpe
    ratio of strategy ``i`` is ``annual_sharpe[i]``.
    """
    if not 0.0 <= rho < 1.0:
        raise ValueError("rho must lie in [0, 1)")
    sharpe = np.broadcast_to(np.asarray(annual_sharpe, dtype=float), (n_strategies,))
    mean = vol * sharpe / np.sqrt(periods_per_year)
    if dist == "t5":
        scale = np.sqrt(3.0 / 5.0)
        common = scale * rng.standard_t(5, size=(n_obs, 1))
        idio = scale * rng.standard_t(5, size=(n_obs, n_strategies))
        vol_path = np.ones((n_obs, 1))
    elif dist in ("normal", "garch", "ar1"):
        burn = 200 if dist == "ar1" else 0
        common = rng.standard_normal((n_obs + burn, 1))
        idio = rng.standard_normal((n_obs + burn, n_strategies))
        if dist == "ar1":
            filt = ([np.sqrt(1.0 - _AR1_PHI**2)], [1.0, -_AR1_PHI])
            common = lfilter(*filt, common, axis=0)[burn:]
            idio = lfilter(*filt, idio, axis=0)[burn:]
        vol_path = np.ones((n_obs, 1))
        if dist == "garch":
            a, b = 0.1, 0.88
            h = 1.0
            for t in range(n_obs):
                vol_path[t, 0] = np.sqrt(h)
                h = (1.0 - a - b) + a * (vol_path[t, 0] * common[t, 0]) ** 2 + b * h
    else:
        raise ValueError("dist must be 'normal', 't5' or 'garch'")
    return mean + vol * vol_path * (np.sqrt(rho) * common + np.sqrt(1.0 - rho) * idio)


def normal_sharpe_covariance_true(
    annual_sharpe: float | np.ndarray,
    rho: float,
    n_obs: int,
    n_strategies: int,
    periods_per_year: float = 252.0,
) -> np.ndarray:
    """True covariance of the annualised Sharpe estimates under iid normal returns.

    Lo (2002) / Memmel (2003) with per-period Sharpe ratios, scaled by the number of
    periods per year; equicorrelated returns with correlation ``rho``.
    """
    sharpe = np.broadcast_to(np.asarray(annual_sharpe, dtype=float), (n_strategies,))
    per = sharpe / np.sqrt(periods_per_year)
    corr = rho * np.ones((n_strategies, n_strategies)) + (1.0 - rho) * np.eye(n_strategies)
    cov = (corr + np.outer(per, per) * corr**2 / 2.0) / n_obs
    return cov * periods_per_year


_RAW_KEYS = ("mu_k", "x", "se", "est_c", "lo_c", "hi_c", "est_h", "lo_h", "hi_h")


def _stack(rows: list[dict[str, float]]) -> dict[str, np.ndarray]:
    return {key: np.array([r[key] for r in rows]) for key in rows[0]}


def feasible_replications(
    n_rep: int,
    n_obs: int,
    n_strategies: int,
    rho: float,
    annual_sharpe: float | np.ndarray = 0.0,
    *,
    dist: Distribution = "normal",
    cov_method: str = "hac",
    periods_per_year: float = 252.0,
    alpha: float = 0.05,
    beta: float | None = None,
    n_sim: int = 20_000,
    true_sigma_critical: tuple[float, float] | None = None,
    rng: np.random.Generator | int | None = 0,
) -> dict[str, np.ndarray]:
    """Raw per-replication results of the *feasible* procedure.

    Every replication simulates a return panel, estimates the Sharpe ratios and their
    covariance from it (``cov_method``), picks the best trial and computes the naive,
    conditional, hybrid, projection and Bonferroni results from the estimated
    covariance, with critical values simulated from it (``n_sim`` draws). Failures of the
    inversion appear as ``nan`` and are counted, never dropped. If ``dist == "normal"``
    the same replication is also analysed with the *true* covariance (keys prefixed
    ``true_``), which isolates the cost of estimating it.
    """
    rng = np.random.default_rng(rng)
    beta = _check_levels(alpha, beta)
    sharpe = np.broadcast_to(np.asarray(annual_sharpe, dtype=float), (n_strategies,))
    sigma_true = None
    if dist == "normal":
        sigma_true = normal_sharpe_covariance_true(
            sharpe, rho, n_obs, n_strategies, periods_per_year
        )
        if true_sigma_critical is None:
            true_sigma_critical = tuple(
                float(v)
                for v in projection_critical_value(sigma_true, [alpha, beta], n_sim=n_sim, rng=rng)
            )
    rows: list[dict[str, float]] = []
    true_rows: list[dict[str, float]] = []
    for _ in range(n_rep):
        panel = simulate_returns(rng, n_obs, n_strategies, rho, sharpe, periods_per_year, dist=dist)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            perf = estimate_performance(panel, periods_per_year, cov_method=cov_method, rng=rng)
        for cov, crit, out in (
            (perf.cov, None, rows),
            (sigma_true, true_sigma_critical, true_rows),
        ):
            if cov is None:
                continue
            x = perf.values
            if crit is None:
                c_alpha, c_beta = (
                    float(v)
                    for v in projection_critical_value(cov, [alpha, beta], n_sim=n_sim, rng=rng)
                )
            else:
                c_alpha, c_beta = crit
            k = int(np.argmax(x))
            res = infer_batch(x[None, :], cov, np.array([k]), alpha=alpha, beta=beta, c_beta=c_beta)
            row = {
                key: float(res[key][0])
                for key in ("est_c", "lo_c", "hi_c", "est_h", "lo_h", "hi_h")
            }
            row.update(mu_k=float(sharpe[k]), x=float(x[k]), se=float(np.sqrt(cov[k, k])))
            row["c_alpha"] = c_alpha
            out.append(row)
    raw = _stack(rows)
    if true_rows:
        raw.update({f"true_{key}": val for key, val in _stack(true_rows).items()})
    return raw


def summarise_replications(
    raw: dict[str, np.ndarray], n_trials: int, alpha: float = 0.05, prefix: str = ""
) -> CoverageResult:
    """:func:`replication_metrics` of the raw arrays from :func:`feasible_replications`."""
    g = {key: raw[prefix + key] for key in (*_RAW_KEYS, "c_alpha")}
    return replication_metrics(
        g["mu_k"],
        g["x"],
        g["se"],
        g["c_alpha"],
        g["est_c"],
        g["lo_c"],
        g["hi_c"],
        g["est_h"],
        g["lo_h"],
        g["hi_h"],
        alpha=alpha,
        n_trials=n_trials,
    )


def _worker(args):
    kwargs, seed_seq = args
    return feasible_replications(**kwargs, rng=np.random.default_rng(seed_seq))


def simulate_feasible(
    n_rep: int,
    n_obs: int,
    n_strategies: int,
    rho: float,
    annual_sharpe: float | np.ndarray = 0.0,
    *,
    dist: Distribution = "normal",
    cov_method: str = "hac",
    periods_per_year: float = 252.0,
    alpha: float = 0.05,
    beta: float | None = None,
    n_sim: int = 20_000,
    seed: int = 0,
    n_jobs: int = 1,
    chunk: int = 100,
) -> tuple[CoverageResult, CoverageResult | None]:
    """Coverage of the feasible procedure (and, for normal returns, with the true covariance).

    Replications are split into chunks of ``chunk`` with independent random streams
    (``SeedSequence(seed).spawn``), so the result does not depend on ``n_jobs``.

    Returns
    -------
    (estimated, true)
        ``estimated`` uses the covariance estimated from every panel; ``true`` uses the
        known covariance (only for ``dist == "normal"``, else ``None``).
    """
    sharpe = np.broadcast_to(np.asarray(annual_sharpe, dtype=float), (n_strategies,)).copy()
    sizes: Sequence[int] = [min(chunk, n_rep - s) for s in range(0, n_rep, chunk)]
    seeds = np.random.SeedSequence(seed).spawn(len(sizes))
    base = {
        "n_obs": n_obs,
        "n_strategies": n_strategies,
        "rho": rho,
        "annual_sharpe": sharpe,
        "dist": dist,
        "cov_method": cov_method,
        "periods_per_year": periods_per_year,
        "alpha": alpha,
        "beta": beta,
        "n_sim": n_sim,
    }
    if dist == "normal":
        sigma_true = normal_sharpe_covariance_true(
            sharpe, rho, n_obs, n_strategies, periods_per_year
        )
        crit = tuple(
            float(v)
            for v in projection_critical_value(
                sigma_true, [alpha, _check_levels(alpha, beta)], n_sim=n_sim, rng=seed
            )
        )
        base["true_sigma_critical"] = crit
    jobs = [({**base, "n_rep": size}, ss) for size, ss in zip(sizes, seeds, strict=True)]
    if n_jobs > 1:
        with ProcessPoolExecutor(max_workers=n_jobs) as pool:
            parts = list(pool.map(_worker, jobs))
    else:
        parts = [_worker(job) for job in jobs]
    raw = {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}
    est = summarise_replications(raw, n_strategies, alpha)
    true = summarise_replications(raw, n_strategies, alpha, "true_") if dist == "normal" else None
    return est, true
