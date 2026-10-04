"""Shared pipeline of the real-data case studies: select in sample, test out of sample."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from common import CACHE
from scipy import stats

from wincurse import WincurseWarning, analyze, check_out_of_sample, nearest_neighbour_correlation

ALPHA = 0.05
N_SIM = 200_000
SEED = 20261004


def drop_duplicates(panel: pd.DataFrame, tol: float = 1e-10) -> tuple[pd.DataFrame, list[str]]:
    """Remove variants whose in-sample series are (almost) perfectly correlated with an earlier
    one; AKM's uniqueness condition fails for such pairs."""
    corr = np.corrcoef(panel.to_numpy(), rowvar=False)
    keep: list[int] = []
    dropped: list[str] = []
    for j in range(panel.shape[1]):
        if any(corr[j, i] > 1 - tol for i in keep):
            dropped.append(str(panel.columns[j]))
        else:
            keep.append(j)
    return panel.iloc[:, keep], dropped


def interval(iv) -> list[float]:
    return [iv.lower, iv.upper]


def run_case(
    label: str,
    panel: pd.DataFrame,
    periods_per_year: float,
    in_sample_end: str,
    *,
    cov_method: str = "hac",
) -> dict[str, Any]:
    """Select the best variant before ``in_sample_end`` and evaluate it afterwards.

    Returns the dictionary written to ``docs/results.json`` and stores the in-sample and
    out-of-sample panels in ``data/cache`` for the command-line demonstration.
    """
    full = panel.dropna(how="any")
    in_sample = full.loc[:in_sample_end]
    out_sample = full.loc[pd.Timestamp(in_sample_end) + pd.Timedelta(days=1) :]
    in_sample, dropped = drop_duplicates(in_sample)
    out_sample = out_sample[in_sample.columns]
    CACHE.mkdir(parents=True, exist_ok=True)
    in_sample.to_csv(CACHE / f"{label}_insample.csv")
    out_sample.to_csv(CACHE / f"{label}_oos.csv")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", WincurseWarning)
        res = analyze(
            in_sample,
            periods_per_year,
            ALPHA,
            cov_method=cov_method,
            n_sim=N_SIM,
            seed=SEED,
            hidden_trials=in_sample.shape[1],
            arch_tests=True,
        )
        shrunk = analyze(
            in_sample,
            periods_per_year,
            ALPHA,
            cov_method=cov_method,
            shrinkage=True,
            n_sim=N_SIM,
            seed=SEED,
        )
        oos = check_out_of_sample(res, out_sample)
        oos_t = check_out_of_sample(res, out_sample, cov_method="normal")

    w = res.winner
    k = w.index
    sr_in = res.performance.values
    sr_out = np.sqrt(periods_per_year) * out_sample.mean().to_numpy() / out_sample.std().to_numpy()
    top = np.argsort(sr_in)[::-1][:10]
    rank_corr = float(stats.spearmanr(sr_in, sr_out).statistic)
    perf = res.performance
    sd = np.sqrt(np.diag(perf.cov))
    corr = perf.cov / np.outer(sd, sd)
    k_all = perf.n_strategies
    dsr = res.dsr
    return {
        "label": label,
        "n_variants": k_all,
        "n_duplicates_dropped": len(dropped),
        "periods_per_year": periods_per_year,
        "in_sample": {
            "start": str(in_sample.index[0].date()),
            "end": str(in_sample.index[-1].date()),
            "n_obs": int(in_sample.shape[0]),
        },
        "out_of_sample": {
            "start": str(out_sample.index[0].date()),
            "end": str(out_sample.index[-1].date()),
            "n_obs": int(out_sample.shape[0]),
        },
        "winner": {
            "name": res.winner_name,
            "naive": w.estimate,
            "se": w.se,
            "conditional_estimate": w.conditional_estimate,
            "hybrid_estimate": w.hybrid_estimate,
            "naive_interval": interval(w.naive),
            "conditional_interval": interval(w.conditional),
            "hybrid_interval": interval(w.hybrid),
            "projection_interval": interval(w.projection),
            "bonferroni_interval": interval(w.bonferroni),
            "truncation": list(w.truncation),
            "haircut_hybrid": res.haircut,
            "critical_value_beta": w.critical_value_beta,
            "critical_value_alpha": w.critical_value_alpha,
        },
        "shrinkage_robustness": {
            "intensity": shrunk.performance.shrinkage,
            "hybrid_estimate": shrunk.winner.hybrid_estimate,
            "hybrid_interval": interval(shrunk.winner.hybrid),
            "conditional_estimate": shrunk.winner.conditional_estimate,
        },
        "diagnostics": {
            "margin_se": res.diagnostics.margin,
            "gap_to_second_se": res.diagnostics.gap_in_se,
            "effective_trials": res.diagnostics.effective_trials,
            "conditional_to_naive_width": res.diagnostics.conditional_to_naive_width,
            "mean_pairwise_correlation": float((corr.sum() - k_all) / (k_all * (k_all - 1))),
            "second_best_naive": float(np.sort(sr_in)[-2]),
        },
        "comparators": {
            "dsr_n_trials": dsr.dsr if dsr else None,
            "dsr_effective_trials": dsr.dsr_effective if dsr else None,
            "effective_trials_used": dsr.effective_trials if dsr else None,
            "haircut_bonferroni": res.haircuts["bonferroni"].haircut if res.haircuts else None,
            "haircut_bhy": res.haircuts["bhy"].haircut if res.haircuts else None,
            "bonferroni_adjusted_sharpe": (
                res.haircuts["bonferroni"].adjusted_sharpe if res.haircuts else None
            ),
            "spa_pvalue": res.arch.spa_pvalue if res.arch else None,
            "stepm_winner_superior": res.arch.winner_is_superior if res.arch else None,
            "stepm_n_superior": len(res.arch.superior) if res.arch else None,
        },
        "out_of_sample_check": {
            "realised": oos.realised,
            "se": oos.se,
            "forecast_naive": interval(oos.forecast.naive),
            "forecast_conditional": interval(oos.forecast.conditional),
            "forecast_hybrid": interval(oos.forecast.hybrid),
            "inside_naive": oos.in_naive,
            "inside_conditional": oos.in_conditional,
            "inside_hybrid": oos.in_hybrid,
            "conditional_pvalue": oos.conditional_pvalue,
            "realised_inside_in_sample_hybrid_interval": bool(oos.realised in w.hybrid),
            "realised_inside_in_sample_naive_interval": bool(oos.realised in w.naive),
            "se_with_normal_formula": oos_t.se,
            "forecast_hybrid_with_normal_formula": interval(oos_t.forecast.hybrid),
        },
        "all_variants": {
            "spearman_in_vs_out_of_sample_sharpe": rank_corr,
            "mean_in_sample_sharpe": float(sr_in.mean()),
            "mean_out_of_sample_sharpe": float(sr_out.mean()),
            "top10_mean_in_sample_sharpe": float(sr_in[top].mean()),
            "top10_mean_out_of_sample_sharpe": float(sr_out[top].mean()),
            "share_positive_out_of_sample": float(np.mean(sr_out > 0)),
            "winner_out_of_sample_rank_percentile": float(np.mean(sr_out < sr_out[k])),
        },
        "unlogged_trials": {
            "hidden_correlation": nearest_neighbour_correlation(perf.cov),
            "scenario_acceptance": res.scenario.acceptance if res.scenario else None,
            "projection": [
                {
                    "n_hidden": r.n_hidden,
                    "interval": [r.lower, r.upper],
                    "width_ratio": r.width_ratio,
                }
                for r in (res.projection_sensitivity or [])
            ],
        },
        "warnings": res.warnings,
        "covariance_method": cov_method,
        "bandwidth": res.performance.bandwidth,
        "n_sim": N_SIM,
        "seed": SEED,
        "alpha": ALPHA,
    }
