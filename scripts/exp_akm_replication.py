"""Replicate the empirical application of Andrews, Kitagawa and McCloskey (2024).

Inputs are the JobStart estimates and covariance matrix of the authors' replication
package (``tests/data/akm_jobstart.npz``, extracted by ``extract_akm_fixture.py``).
Compares (i) the exact values of this package, (ii) the numbers printed in Tables II
and III of the paper, and (iii) the spread of the authors' own Monte Carlo algorithm
(sorted truncated-normal draws, ``R_cv`` draws, noisy root search) re-implemented here,
which shows that the printed endpoints are within the Monte Carlo error of their method.
"""

from __future__ import annotations

import sys
import time

import numpy as np
from common import ROOT, environment, update_results
from scipy import optimize
from scipy.special import ndtr, ndtri

from wincurse import forecast_interval, independent_critical_value, winner_inference

ALPHA, BETA = 0.05, 0.005
PRINTED_TABLE_II = {
    "naive": [3615, 9479],
    "conditional": [3485, 9478],
    "projection": [2232, 10862],
    "hybrid": [3420, 9538],
    "estimates": {"naive": 6547, "conditional": 6544, "hybrid": 6545},
}
PRINTED_TABLE_III = {
    "all_12_sites": {"oos_se": 1315.0, "conditional": [2609, 10449], "hybrid": [2531, 10529]},
    "high_fidelity_4_sites": {
        "oos_se": 2607.0,
        "conditional": [634, 12436],
        "hybrid": [520, 12556],
    },
}


def _trandn(rng, a: float, b: float, n: int) -> np.ndarray:
    u = rng.random(n)
    if a > 0:
        pa, pb = ndtr(-a), ndtr(-b)
        return -ndtri(pa - u * (pa - pb))
    pa, pb = ndtr(a), ndtr(b)
    return ndtri(pa + u * (pb - pa))


def authors_lower_endpoints(mu, sigma, r_cv: int, seeds: int) -> dict[str, list[float]]:
    """Lower endpoints of the conditional and hybrid intervals from the authors' algorithm."""
    k = int(np.argmax(mu))
    m, var = mu[k], sigma[k, k]
    sd = np.sqrt(var)
    z = mu - sigma[:, k] / var * m
    v_minus, v_plus = -np.inf, np.inf
    for j in range(len(mu)):
        gap = var - sigma[j, k]
        if gap > 0:
            v_minus = max(v_minus, var * z[j] / gap)
        elif gap < 0:
            v_plus = min(v_plus, var * z[j] / gap)
    cv_beta = independent_critical_value(len(mu), BETA) * sd
    gamma = 1 - (1 - ALPHA) / (1 - BETA)

    def upper_quantile(rng, mean, lo, hi, level):
        draws = mean + sd * _trandn(rng, (lo - mean) / sd, (hi - mean) / sd, r_cv)
        return np.sort(draws)[round(r_cv * (1 - level / 2)) - 1]

    out: dict[str, list[float]] = {"conditional": [], "hybrid": []}
    for seed in range(seeds):
        rng = np.random.default_rng(seed)

        def cond(mean, rng=rng):
            return m - upper_quantile(rng, mean, v_minus, v_plus, ALPHA)

        def hyb(mean, rng=rng):
            lo, hi = max(v_minus, mean - cv_beta), min(v_plus, mean + cv_beta)
            return m - upper_quantile(rng, mean, lo, hi, gamma)

        out["conditional"].append(optimize.brentq(cond, m - 3 * sd, m, xtol=1e-3))
        out["hybrid"].append(optimize.brentq(hyb, m - cv_beta + 1e-6, m, xtol=1e-3))
    return out


def main(seeds: int = 40) -> None:
    started = time.time()
    data = np.load(ROOT / "tests" / "data" / "akm_jobstart.npz")
    mu, sigma = data["mu"], data["sigma"]
    k = len(mu)
    c_alpha = independent_critical_value(k, ALPHA)
    c_beta = independent_critical_value(k, BETA)
    exact = winner_inference(mu, sigma, ALPHA, BETA, critical_values=(c_alpha, c_beta))
    simulated = winner_inference(mu, sigma, ALPHA, BETA, n_sim=500_000, rng=1)

    def pair(iv):
        return [iv.lower, iv.upper]

    emulation = {}
    for r_cv in (500_000, 100_000):
        ends = authors_lower_endpoints(mu, sigma, r_cv, seeds)
        emulation[str(r_cv)] = {
            name: {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1))}
            for name, v in ends.items()
        }

    forecasts = {}
    for label, spec in PRINTED_TABLE_III.items():
        fi = forecast_interval(
            mu, sigma, spec["oos_se"] ** 2, ALPHA, BETA, critical_value_beta=c_beta
        )
        forecasts[label] = {
            "oos_se": spec["oos_se"],
            "conditional": pair(fi.conditional),
            "hybrid": pair(fi.hybrid),
            "printed_conditional": spec["conditional"],
            "printed_hybrid": spec["hybrid"],
        }

    update_results(
        "akm_replication",
        {
            "setting": {
                "alpha": ALPHA,
                "beta": BETA,
                "n_policies": k,
                "winner_index_one_based": exact.index + 1,
                "winner_estimate": exact.estimate,
                "winner_se": exact.se,
                "truncation": list(exact.truncation),
                "critical_value_alpha_exact": c_alpha,
                "critical_value_beta_exact": c_beta,
                "critical_value_alpha_simulated_500k": simulated.critical_value_alpha,
                "critical_value_beta_simulated_500k": simulated.critical_value_beta,
            },
            "exact": {
                "naive": pair(exact.naive),
                "conditional_estimate": exact.conditional_estimate,
                "conditional": pair(exact.conditional),
                "projection": pair(exact.projection),
                "hybrid_estimate": exact.hybrid_estimate,
                "hybrid": pair(exact.hybrid),
            },
            "with_simulated_critical_values_500k": {
                "projection": pair(simulated.projection),
                "hybrid": pair(simulated.hybrid),
                "hybrid_estimate": simulated.hybrid_estimate,
            },
            "printed_table_ii": PRINTED_TABLE_II,
            "authors_algorithm_emulation_lower_endpoints": {
                "n_seeds": seeds,
                "description": "mean and sd over seeds of the authors' Monte Carlo algorithm",
                **emulation,
            },
            "forecast_intervals": forecasts,
            "environment": environment(),
            "runtime_seconds": time.time() - started,
        },
    )
    print("akm replication written to docs/results.json")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 40)
