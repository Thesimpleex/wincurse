"""Monte Carlo coverage study of the naive, conditional, hybrid and projection intervals.

Design (see docs/methodology.md):

* Returns are daily, ten years (T = 2520), from a one-factor equicorrelation model with
  K strategies; the unit of every statistic is the annualised Sharpe ratio.
* ``known_sigma``: the estimates are drawn from N(mu, Sigma) with the true sampling
  covariance Sigma of the Sharpe estimates (iid normal returns) and the procedures use
  Sigma. This is the setting of the theory and it is cheap enough for 10,000 replications
  over the full grid K x rho x {equal, spread true Sharpe ratios}.
* ``feasible``: every replication simulates a return panel, estimates the Sharpe ratios and
  their covariance (HAC unless stated), and applies the procedures to the estimates, with
  critical values simulated from the estimated covariance. For normal returns the same
  replication is also analysed with the true covariance, which isolates the cost of
  estimating it. Student-t(5) and GARCH(1,1) returns and three covariance estimators
  show robustness and limits.

Every replication is scored; failed inversions would be counted as misses (none occur).
Cells are cached in ``data/cache/coverage/`` so an interrupted run resumes.
"""

from __future__ import annotations

import os

for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from dataclasses import dataclass  # noqa: E402

import numpy as np  # noqa: E402
from common import CACHE, environment, update_results  # noqa: E402

from wincurse import simulate_known_sigma  # noqa: E402
from wincurse.simulation import normal_sharpe_covariance_true, simulate_feasible  # noqa: E402

T_OBS, Q = 2520, 252.0
EQUAL_SHARPE = 0.5
ALPHA = 0.05
KS = (10, 100, 1000)
RHOS = (0.3, 0.5, 0.7, 0.9)
CELL_DIR = CACHE / "coverage"


@dataclass(frozen=True)
class Cell:
    """One cell of the feasible study."""

    k: int
    rho: float
    config: str
    dist: str
    cov: str
    n_obs: int = T_OBS
    q: float = Q

    def name(self, reps: int) -> str:
        base = f"feas_K{self.k}_rho{self.rho}_{self.config}_{self.dist}_{self.cov}"
        suffix = "" if self.n_obs == T_OBS else f"_T{self.n_obs}"
        return f"{base}{suffix}_n{reps}"


def sharpe_vector(k: int, config: str) -> np.ndarray:
    if config == "equal":
        return np.full(k, EQUAL_SHARPE)
    return np.linspace(0.0, 1.0, k)


def cached(name: str, compute):
    path = CELL_DIR / f"{name}.json"
    if path.exists():
        return json.loads(path.read_text())
    result = compute()
    CELL_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result))
    return result


def known_cell(k: int, rho: float, config: str, reps: int, seed: int) -> dict:
    sharpe = sharpe_vector(k, config)
    sigma = normal_sharpe_covariance_true(sharpe, rho, T_OBS, k, Q)
    res = simulate_known_sigma(sharpe, sigma, reps, ALPHA, n_sim=200_000, rng=seed)
    return {"k": k, "rho": rho, "config": config, "reps": reps, **res.to_dict()}


def feasible_cell(
    k: int,
    rho: float,
    config: str,
    dist: str,
    cov: str,
    reps: int,
    jobs: int,
    seed: int,
    n_obs: int = T_OBS,
    q: float = Q,
) -> dict:
    sharpe = sharpe_vector(k, config)
    est, true = simulate_feasible(
        reps,
        n_obs,
        k,
        rho,
        sharpe,
        dist=dist,
        cov_method=cov,
        periods_per_year=q,
        alpha=ALPHA,
        n_sim=20_000,
        seed=seed,
        n_jobs=jobs,
        chunk=50,
    )
    out = {
        "k": k,
        "rho": rho,
        "config": config,
        "dist": dist,
        "cov": cov,
        "reps": reps,
        "n_obs": n_obs,
        "periods_per_year": q,
        "estimated_sigma": est.to_dict(),
    }
    if true is not None:
        out["true_sigma"] = true.to_dict()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--quick", action="store_true", help="small run for a smoke test")
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--only", choices=["known", "feasible"], default=None)
    parser.add_argument("--no-write", action="store_true", help="do not touch docs/results.json")
    args = parser.parse_args()
    reps_known = 400 if args.quick else 10_000
    reps_feas = 60 if args.quick else 2_000
    started = time.time()
    seed = 20261004

    known: dict[str, dict] = {}
    if args.only in (None, "known"):
        for k in KS:
            for rho in RHOS:
                for config in ("equal", "spread"):
                    name = f"known_K{k}_rho{rho}_{config}_n{reps_known}"
                    t0 = time.time()
                    known[name] = cached(
                        name,
                        lambda k=k, rho=rho, config=config: known_cell(
                            k, rho, config, reps_known, seed
                        ),
                    )
                    c = known[name]
                    print(
                        f"{name}: naive {c['naive_coverage']:.3f} "
                        f"cond {c['conditional_coverage']:.3f} "
                        f"hyb {c['hybrid_coverage']:.3f} failed {c['n_failed']} "
                        f"({time.time() - t0:.0f}s)",
                        flush=True,
                    )

    feasible: dict[str, dict] = {}
    if args.only in (None, "feasible"):
        cells: list[Cell] = []
        for k in (10, 100):
            cells += [Cell(k, rho, "equal", "normal", "hac") for rho in RHOS]
        for dist in ("normal", "t5", "garch", "ar1"):
            cells += [Cell(100, 0.5, "equal", dist, cov) for cov in ("normal", "iid", "hac")]
        cells += [
            Cell(100, 0.5, "spread", "normal", "hac"),
            Cell(10, 0.5, "spread", "normal", "hac"),
        ]
        cells += [
            Cell(100, 0.5, "equal", "normal", "hac", 504, Q),
            Cell(100, 0.5, "equal", "normal", "hac", 120, 12.0),
        ]
        if not args.quick:
            cells += [
                Cell(1000, 0.5, "equal", "normal", "hac"),
                Cell(1000, 0.9, "equal", "normal", "hac"),
            ]
        for cell in dict.fromkeys(cells):
            name = cell.name(reps_feas)
            t0 = time.time()
            feasible[name] = cached(
                name,
                lambda cell=cell: feasible_cell(
                    cell.k,
                    cell.rho,
                    cell.config,
                    cell.dist,
                    cell.cov,
                    reps_feas,
                    args.jobs,
                    seed + 1,
                    cell.n_obs,
                    cell.q,
                ),
            )
            c = feasible[name]["estimated_sigma"]
            print(
                f"{name}: naive {c['naive_coverage']:.3f} cond {c['conditional_coverage']:.3f} "
                f"hyb {c['hybrid_coverage']:.3f} proj {c['projection_coverage']:.3f} "
                f"failed {c['n_failed']} ({time.time() - t0:.0f}s)",
                flush=True,
            )

    if args.no_write or args.quick:
        return
    payload = {
        "design": {
            "n_obs": T_OBS,
            "periods_per_year": Q,
            "alpha": ALPHA,
            "beta": ALPHA / 10,
            "equal_true_sharpe": EQUAL_SHARPE,
            "spread_true_sharpe": "linspace(0, 1, K)",
            "reps_known_sigma": reps_known,
            "reps_feasible": reps_feas,
            "critical_value_draws_known": 200_000,
            "critical_value_draws_feasible": 20_000,
            "seed": seed,
        },
        "known_sigma": known,
        "feasible": feasible,
        "environment": environment(),
        "runtime_seconds": time.time() - started,
    }
    update_results("coverage", payload)
    print("coverage results written to docs/results.json")


if __name__ == "__main__":
    main()
