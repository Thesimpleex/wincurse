"""Case study (a): industry momentum grid, selected in sample and tested out of sample.

Data: Ken French Data Library, 49 industry portfolios (value-weighted, monthly). A grid of
256 long-short momentum variants (lookback, skip, holding period, number of industries per
leg) is built with information up to the previous month. The best variant by Sharpe ratio
before 1991 is corrected for selection and then compared with its realised Sharpe ratio
from 1991 onwards through the forecast interval for a new realisation.
"""

from __future__ import annotations

import frenchdata
import strategies
from casestudy import run_case
from common import environment, update_results

IN_SAMPLE_END = "1990-12-31"


def main() -> None:
    data = frenchdata.industries_monthly()
    panel, params = strategies.momentum_variants(data.frame)
    result = run_case("momentum", panel, 12.0, IN_SAMPLE_END)
    winner = params.loc[result["winner"]["name"]].to_dict()
    result["winner"]["parameters"] = {k: int(v) for k, v in winner.items() if k != "missing_held"}
    result["data"] = {
        "source": data.source,
        "crsp_vintage": data.created,
        "last_observation": data.last_observation,
        "grid": {
            "lookbacks": [1, 2, 3, 6, 9, 12, 18, 24],
            "skips": [0, 1],
            "holdings": [1, 3, 6, 12],
            "legs": [3, 5, 7, 10],
            "n_grid_points": int(params.shape[0]),
            "held_industry_months_with_missing_return": int(params["missing_held"].sum()),
        },
    }
    result["environment"] = environment()
    update_results("momentum", result)
    w, o = result["winner"], result["out_of_sample_check"]
    print(
        f"momentum: winner {w['name']}: naive {w['naive']:.2f}, hybrid {w['hybrid_estimate']:.2f} "
        f"{w['hybrid_interval']}, realised OOS {o['realised']:.2f}, "
        f"forecast hybrid {o['forecast_hybrid']}"
    )


if __name__ == "__main__":
    main()
