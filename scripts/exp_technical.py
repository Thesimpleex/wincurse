"""Case study (b): technical trading rules on the US market, in the spirit of
Sullivan, Timmermann and White (1999).

Data: Ken French daily Mkt-RF and RF. 340 rules (moving-average crossovers with bands,
filter rules, channel breakouts, time-series momentum; each long/short and long/flat) trade
the market; positions use closes up to the previous day and no transaction costs are
charged. The best rule by Sharpe ratio before 1991 is corrected for selection and compared
with its realised Sharpe ratio from 1991 onwards.
"""

from __future__ import annotations

import frenchdata
import strategies
from casestudy import run_case
from common import environment, update_results

IN_SAMPLE_START = "1927-07-01"
IN_SAMPLE_END = "1990-12-31"


def main() -> None:
    data = frenchdata.factors_daily()
    factors = data.frame
    panel, params = strategies.technical_variants(factors["Mkt-RF"], factors["RF"])
    panel = panel.loc[IN_SAMPLE_START:]
    result = run_case("technical", panel, 252.0, IN_SAMPLE_END)
    winner = params.loc[result["winner"]["name"]].to_dict()
    result["winner"]["parameters"] = {
        k: (v if isinstance(v, str) else float(v)) for k, v in winner.items()
    }
    market = factors["Mkt-RF"].loc[IN_SAMPLE_START:]
    in_sample = market.loc[:IN_SAMPLE_END]
    out_sample = market.loc["1991-01-01":]
    result["benchmark_always_long_market"] = {
        "in_sample_sharpe": float(252**0.5 * in_sample.mean() / in_sample.std()),
        "out_of_sample_sharpe": float(252**0.5 * out_sample.mean() / out_sample.std()),
    }
    result["data"] = {
        "source": data.source,
        "crsp_vintage": data.created,
        "last_observation": data.last_observation,
        "rule_families": params["family"].value_counts().to_dict(),
        "in_sample_start": IN_SAMPLE_START,
    }
    result["environment"] = environment()
    update_results("technical", result)
    w, o = result["winner"], result["out_of_sample_check"]
    print(
        f"technical: winner {w['name']}: naive {w['naive']:.2f}, hybrid {w['hybrid_estimate']:.2f} "
        f"{w['hybrid_interval']}, realised OOS {o['realised']:.2f}, "
        f"forecast hybrid {o['forecast_hybrid']}"
    )


if __name__ == "__main__":
    main()
