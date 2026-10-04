"""Write ``examples/parameter_sweep.csv``: a synthetic sweep of 40 correlated strategy variants.

Four years of daily returns; every variant has the same true annualised Sharpe ratio
(0.3) and a pairwise correlation of 0.5, so whichever variant looks best is the best
of noise. The seed was picked from the first 40 so that the winner has a clearly positive
naive Sharpe ratio (1.3); other seeds illustrate the same point. The file is the input of
the README quickstart.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from common import ROOT, update_results

from wincurse import analyze
from wincurse.simulation import simulate_returns

OUT = ROOT / "examples" / "parameter_sweep.csv"


def main() -> None:
    rng = np.random.default_rng(9)
    returns = simulate_returns(rng, 1008, 40, 0.5, annual_sharpe=0.3, vol=0.01)
    index = pd.bdate_range("2021-01-04", periods=1008)
    frame = pd.DataFrame(returns, index=index, columns=[f"variant_{i + 1:02d}" for i in range(40)])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    frame.round(6).to_csv(OUT, index_label="date")
    result = analyze(frame, alpha=0.05, method="hybrid", n_sim=100_000, seed=0)
    w = result.winner
    update_results(
        "quickstart",
        {
            "file": "examples/parameter_sweep.csv",
            "n_obs": int(frame.shape[0]),
            "n_trials": int(frame.shape[1]),
            "winner": result.winner_name,
            "naive": w.estimate,
            "naive_interval": [w.naive.lower, w.naive.upper],
            "hybrid_estimate": w.hybrid_estimate,
            "hybrid_interval": [w.hybrid.lower, w.hybrid.upper],
            "conditional_estimate": w.conditional_estimate,
            "haircut": result.haircut,
            "effective_trials": result.diagnostics.effective_trials,
            "deflated_sharpe": result.dsr.dsr if result.dsr else None,
            "true_sharpe_of_every_variant": 0.3,
        },
    )
    print(f"wrote {OUT} {frame.shape}")


if __name__ == "__main__":
    main()
