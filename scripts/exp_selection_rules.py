"""Other selection rules on the momentum grid (should-have extensions).

(a) Select on the Sharpe ratio, report the Fama-French three-factor alpha of the selected
    variant: conditional and hybrid inference for the reported statistic.
(b) Report the five variants with the highest Sharpe ratio: conditional inference for each.
(c) "Published only if t > 2": every variant whose in-sample t-statistic exceeds 2, with the
    conditional median-unbiased estimate compared with the realised out-of-sample value.

Uses the in-sample panel written by ``exp_momentum.py`` (data/cache/momentum_insample.csv).
"""

from __future__ import annotations

import warnings

import frenchdata
import numpy as np
import pandas as pd
from casestudy import ALPHA, N_SIM, SEED
from common import CACHE, environment, update_results

from wincurse import (
    WincurseWarning,
    estimate_joint,
    estimate_performance,
    threshold_inference,
    top_m_inference,
    winner_inference_cross,
)


def interval(iv) -> list[float]:
    return [iv.lower, iv.upper]


def main() -> None:
    insample = pd.read_csv(CACHE / "momentum_insample.csv", index_col=0, parse_dates=True)
    oos = pd.read_csv(CACHE / "momentum_oos.csv", index_col=0, parse_dates=True)
    factors = frenchdata.factors_monthly()
    ff = factors.frame[["Mkt-RF", "SMB", "HML"]]
    ff_in, ff_out = ff.reindex(insample.index), ff.reindex(oos.index)
    if ff_in.isna().any().any() or ff_out.isna().any().any():
        raise RuntimeError("factor dates do not match the return dates")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", WincurseWarning)
        joint = estimate_joint(insample, ff_in, 12.0, cov_method="hac")
        res = winner_inference_cross(
            joint.select,
            joint.report,
            joint.cov_xx,
            joint.cov_xy,
            joint.cov_yy,
            ALPHA,
            n_sim=N_SIM,
            rng=SEED,
        )
        oos_joint = estimate_joint(oos[[insample.columns[res.index]]], ff_out, 12.0)
        perf = estimate_performance(insample, 12.0, cov_method="hac")

    k = res.index
    sd = np.sqrt(np.diag(perf.cov))
    alpha_block = {
        "winner": insample.columns[k],
        "sharpe_in_sample": float(joint.select[k]),
        "alpha_in_sample_naive": res.estimate,
        "alpha_se": res.se,
        "alpha_conditional_estimate": res.conditional_estimate,
        "alpha_hybrid_estimate": res.hybrid_estimate,
        "naive_interval": interval(res.naive),
        "conditional_interval": interval(res.conditional),
        "hybrid_interval": interval(res.hybrid),
        "projection_interval": interval(res.projection),
        "truncation": list(res.truncation),
        "alpha_out_of_sample_realised": float(oos_joint.report[0]),
        "alpha_out_of_sample_se": float(np.sqrt(oos_joint.cov_yy[0, 0])),
        "units": "annualised alpha, decimal return per year",
        "factors": ["Mkt-RF", "SMB", "HML"],
    }

    top5 = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", WincurseWarning)
        top = top_m_inference(perf.values, perf.cov, 5, ALPHA)
        oos_perf = estimate_performance(oos, 12.0, cov_method="hac")
    for r in top:
        top5.append(
            {
                "name": insample.columns[r.index],
                "naive": r.estimate,
                "conditional_estimate": r.conditional_estimate,
                "conditional_interval": interval(r.conditional),
                "n_windows": len(r.windows),
                "out_of_sample": float(oos_perf.values[r.index]),
            }
        )

    t_stat = perf.values / sd
    passed = threshold_inference(perf.values, perf.cov, 2.0, ALPHA)
    oos_values = oos_perf.values
    cond_est = np.array([r.conditional_estimate for r in passed])
    thr = {
        "threshold_t": 2.0,
        "n_variants": int(perf.n_strategies),
        "n_pass": len(passed),
        "share_pass": len(passed) / perf.n_strategies,
        "mean_naive": float(np.mean([r.estimate for r in passed])),
        "median_conditional": float(np.median(cond_est)),
        "conditional_p10": float(np.quantile(cond_est, 0.1)),
        "conditional_p90": float(np.quantile(cond_est, 0.9)),
        "mean_out_of_sample": float(np.mean([oos_values[r.index] for r in passed])),
        "median_conditional_interval_width": float(
            np.median([r.conditional.width for r in passed])
        ),
        "median_naive_interval_width": float(np.median([r.naive.width for r in passed])),
        "share_oos_inside_conditional_interval": float(
            np.mean([oos_values[r.index] in r.conditional for r in passed])
        ),
        "max_t": float(t_stat.max()),
    }
    update_results(
        "selection_rules",
        {
            "select_sharpe_report_alpha": alpha_block,
            "top5_by_sharpe": top5,
            "threshold": thr,
            "environment": environment(),
            "factors_source": factors.source,
            "factors_last_observation": factors.last_observation,
        },
    )
    print(
        f"alpha {alpha_block['alpha_in_sample_naive']:.4f} -> hybrid "
        f"{alpha_block['alpha_hybrid_estimate']:.4f}; threshold passes {thr['n_pass']}"
    )


if __name__ == "__main__":
    main()
