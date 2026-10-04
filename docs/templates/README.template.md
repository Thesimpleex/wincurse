# wincurse

Selection-corrected Sharpe ratios: median-unbiased estimates and valid confidence intervals for the best of many backtests.

[![CI](https://github.com/Thesimpleex/wincurse/actions/workflows/ci.yml/badge.svg)](https://github.com/Thesimpleex/wincurse/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/screenshots/hero-dark.png">
  <img alt="wincurse analyze on the best of 256 industry-momentum variants: naive and corrected Sharpe ratio, comparators, diagnostics, simulated coverage and an out-of-sample check" src="docs/screenshots/hero-light.png" width="900">
</picture>

## Why this matters

A researcher tries many correlated variants of a strategy and reports the best one. Its backtest Sharpe ratio is the maximum of many noisy estimates, so it is biased upwards, and its textbook confidence interval covers the truth far less often than claimed. Existing open tools test hypotheses: the Deflated Sharpe Ratio asks whether the winner differs from zero, the probability of backtest overfitting asks how often the in-sample winner underperforms, and SPA / StepM ask whether anything beats a benchmark. None returns what a portfolio manager needs to size capital: a corrected point estimate and a valid interval for the selected strategy's true Sharpe ratio that uses the correlation of all variants that were tried. `wincurse` implements the conditional and hybrid inference of Andrews, Kitagawa and McCloskey ("Inference on Winners", *Quarterly Journal of Economics*, 2024) for Sharpe ratios, with covariance estimators for backtests, the comparators above, diagnostics, a trial ledger and a command-line tear sheet. It is meant for quant researchers after a parameter sweep, allocators shown "the best variant", model-validation teams that need a reproducible sign-off statistic, and academics reporting a best specification.

## Key results

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/coverage_vs_k-dark.png">
  <img alt="Coverage of the naive, conditional and hybrid 95% intervals for the best of K trials, for four correlations" src="docs/figures/coverage_vs_k-light.png">
</picture>

**Figure 1.** Coverage of nominal 95% intervals for the true Sharpe ratio of the best of $K$ trials. Ten years of daily returns, every trial equally good (annual Sharpe ratio {{ coverage/design/equal_true_sharpe | 1 }}, the least-separated configuration), {{ coverage/design/reps_known_sigma | int }} replications with known covariance (lines) and {{ coverage/design/reps_feasible | int }} replications in which the covariance is estimated from the simulated returns (diamonds; hybrid only).

* With $K=100$ trials and correlation 0.7 the naive interval covers the true Sharpe ratio of the winner in {{ coverage/known_sigma/known_K100_rho0.7_equal_n10000/naive_coverage | pct1 }} of replications; the conditional interval in {{ coverage/known_sigma/known_K100_rho0.7_equal_n10000/conditional_coverage | pct1 }}, the hybrid interval in {{ coverage/known_sigma/known_K100_rho0.7_equal_n10000/hybrid_coverage | pct1 }}. At $K=1000$ and correlation 0.5 the naive interval covers {{ coverage/known_sigma/known_K1000_rho0.5_equal_n10000/naive_coverage | pct1 }} of the time and the naive estimate overstates the true Sharpe ratio by {{ coverage/known_sigma/known_K1000_rho0.5_equal_n10000/naive_mean_bias | 2 }} on average; conditional {{ coverage/known_sigma/known_K1000_rho0.5_equal_n10000/conditional_coverage | pct1 }}, hybrid {{ coverage/known_sigma/known_K1000_rho0.5_equal_n10000/hybrid_coverage | pct1 }}.
* Estimating the covariance costs almost nothing with ten years of daily data: at $K=1000$, correlation 0.5 (HAC covariance) naive {{ coverage/feasible/feas_K1000_rho0.5_equal_normal_hac_n2000/estimated_sigma/naive_coverage | pct1 }}, conditional {{ coverage/feasible/feas_K1000_rho0.5_equal_normal_hac_n2000/estimated_sigma/conditional_coverage | pct1 }}, hybrid {{ coverage/feasible/feas_K1000_rho0.5_equal_normal_hac_n2000/estimated_sigma/hybrid_coverage | pct1 }}. No inversion failed in any cell of the study; failures would be counted as misses, never skipped.
* The price is width: at $K=100$, correlation 0.7 the median hybrid interval is {{ coverage/known_sigma/known_K100_rho0.7_equal_n10000/hybrid_median_width | 2 }} Sharpe units wide against {{ coverage/known_sigma/known_K100_rho0.7_equal_n10000/naive_median_width | 2 }} for the naive one; the median conditional interval is {{ coverage/known_sigma/known_K100_rho0.7_equal_n10000/conditional_median_width | 2 }} wide, which is why the hybrid interval is the default.
* Limits: with autocorrelated returns (AR(1), coefficient 0.2) a covariance estimator that ignores the autocorrelation undercovers (hybrid {{ coverage/feasible/feas_K100_rho0.5_equal_ar1_normal_n2000/estimated_sigma/hybrid_coverage | pct1 }} with the iid-normal formula) and HAC recovers most of it ({{ coverage/feasible/feas_K100_rho0.5_equal_ar1_hac_n2000/estimated_sigma/hybrid_coverage | pct1 }}). Student-t(5) and GARCH(1,1) returns at daily frequency change little, as the delta-method variance of a daily Sharpe ratio near 0.03 is dominated by the term for the mean. Details in `docs/methodology.md`.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/real_data-dark.png">
  <img alt="Naive and corrected intervals for the winner of two backtest grids, with forecast intervals and the realised out-of-sample Sharpe ratio" src="docs/figures/real_data-light.png">
</picture>

**Figure 2.** Two real case studies from Ken French's data library (illustrative research, not investment advice). The winner is selected before 1991 on the Sharpe ratio; intervals above the gap are for its true in-sample Sharpe ratio; the two forecast intervals below are for a *new* out-of-sample estimate if the true Sharpe ratio were unchanged; the dashed line is the realised Sharpe ratio from 1991 to {{ momentum/out_of_sample/end | str }}.

* **Industry momentum**, {{ momentum/n_variants | int }} variants (lookback, skip, holding period, industries per leg), {{ momentum/in_sample/start | str }} to {{ momentum/in_sample/end | str }}. The winner `{{ momentum/winner/name | str }}` has a naive Sharpe ratio of {{ momentum/winner/naive | 2 }} ({{ momentum/winner/naive_interval/0 | 2 }} to {{ momentum/winner/naive_interval/1 | 2 }}); the hybrid estimate is {{ momentum/winner/hybrid_estimate | 2 }} ({{ momentum/winner/hybrid_interval/0 | 2 }} to {{ momentum/winner/hybrid_interval/1 | 2 }}), a haircut of {{ momentum/winner/haircut_hybrid | pct0 }}. The realised Sharpe ratio after 1991 is {{ momentum/out_of_sample_check/realised | 2 }}, inside the hybrid forecast interval ({{ momentum/out_of_sample_check/forecast_hybrid/0 | 2 }} to {{ momentum/out_of_sample_check/forecast_hybrid/1 | 2 }}; conditional p-value {{ momentum/out_of_sample_check/conditional_pvalue | 2 }}): no decay beyond selection is detectable. The winner leads its closest rival by only {{ momentum/diagnostics/margin_se | 2 }} standard errors and the grid has {{ momentum/diagnostics/effective_trials | 1 }} effective independent trials. The result is sensitive to the covariance of these near-twin variants: with Ledoit-Wolf shrinkage (intensity {{ momentum/shrinkage_robustness/intensity | 2 }}) the hybrid estimate drops to {{ momentum/shrinkage_robustness/hybrid_estimate | 2 }}. Grid choices such as 12-month lookbacks come from literature written after 1990; that prior snooping is not corrected.
* **Technical trading rules** on the US market, {{ technical/n_variants | int }} rules, {{ technical/in_sample/start | str }} to {{ technical/in_sample/end | str }} (daily). The winner `{{ technical/winner/name | str }}` (a 0.5% filter rule, long or short) has a naive Sharpe ratio of {{ technical/winner/naive | 2 }} ({{ technical/winner/naive_interval/0 | 2 }} to {{ technical/winner/naive_interval/1 | 2 }}) against {{ technical/benchmark_always_long_market/in_sample_sharpe | 2 }} for holding the market; the hybrid estimate is {{ technical/winner/hybrid_estimate | 2 }} ({{ technical/winner/hybrid_interval/0 | 2 }} to {{ technical/winner/hybrid_interval/1 | 2 }}), a haircut of only {{ technical/winner/haircut_hybrid | pct0 }}. After 1991 its Sharpe ratio is {{ technical/out_of_sample_check/realised | 2 }}, outside the hybrid forecast interval ({{ technical/out_of_sample_check/forecast_hybrid/0 | 2 }} to {{ technical/out_of_sample_check/forecast_hybrid/1 | 2 }}; conditional p-value {{ technical/out_of_sample_check/conditional_pvalue | 3 }}). Selection explains little of this decay: the haircut is small because the {{ technical/n_variants | int }} rules are highly correlated ({{ technical/diagnostics/effective_trials | 1 }} effective independent trials). The in-sample ten best rules average {{ technical/all_variants/top10_mean_in_sample_sharpe | 2 }} and {{ technical/all_variants/top10_mean_out_of_sample_sharpe | 2 }} afterwards (rank correlation between in-sample and out-of-sample Sharpe ratios of all rules: {{ technical/all_variants/spearman_in_vs_out_of_sample_sharpe | 2 }}). The rest of the decay is not explained by selection; candidate explanations that I did not test are the disappearance of the daily return autocorrelation of the early index data, a change of regime, and the absence of transaction costs.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/screenshots/decay-dark.png">
  <img alt="wincurse analyze on 340 technical trading rules: the realised out-of-sample Sharpe ratio lies outside every forecast interval" src="docs/screenshots/decay-light.png" width="900">
</picture>

The same analysis as a self-contained HTML tear sheet (`--report`), here the top of the momentum report (full example: [`docs/tearsheet_momentum.html`](docs/tearsheet_momentum.html)):

<img alt="Top of the selection-adjusted tear sheet: winner, naive and corrected Sharpe ratio, estimates and intervals" src="docs/screenshots/tearsheet.png" width="760">

Why the hybrid interval is the default: when the winner barely beats the runner-up, the conditional interval extends over thousands of standard errors, while the hybrid interval stays inside the projection set (two independent trials, lead over the runner-up on the horizontal axis).

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/endpoints_vs_gap-dark.png">
  <img alt="Interval ends relative to the winner's estimate as a function of its lead over the runner-up, for the naive, conditional, hybrid and projection intervals" src="docs/figures/endpoints_vs_gap-light.png">
</picture>

## Method

Let $X\sim N(\mu,\Sigma)$ be the estimated performance of $K$ trials and $k=\arg\max_j X_j$ the reported one. Put $Z = X - \Sigma_{\cdot k}X_k/\Sigma_{kk}$, which is independent of $X_k$. Selection of $k$ is equivalent to $X_k\in[L(Z),U(Z)]$ with

$$
L=\max_{\Sigma_{jk}<\Sigma_{kk}}\frac{\Sigma_{kk}Z_j}{\Sigma_{kk}-\Sigma_{jk}},\qquad
U=\min_{\Sigma_{jk}>\Sigma_{kk}}\frac{\Sigma_{kk}Z_j}{\Sigma_{kk}-\Sigma_{jk}},
$$

so that, given selection and $Z$, $X_k$ is $N(\mu_k,\Sigma_{kk})$ truncated to $[L,U]$. Inverting its CDF in $\mu_k$ (monotone) gives the **conditional** median-unbiased estimator and equal-tailed intervals with exact conditional coverage. The **hybrid** procedure additionally conditions on $|X_k-\mu_k|\le c_\beta\sqrt{\Sigma_{kk}}$, where $c_\beta$ is the $1-\beta$ quantile of $\max_j|\xi_j|/\sqrt{\Sigma_{jj}}$, $\xi\sim N(0,\Sigma)$, and runs the conditional step at level $(\alpha-\beta)/(1-\beta)$ with $\beta=\alpha/10$. It stays inside the projection set, has unconditional coverage between $1-\alpha$ and $(1-\alpha)/(1-\beta)$ and a median bias of at most $\beta/2$. For Sharpe ratios, $\Sigma$ comes from the delta method on $(r_t,r_t^2)$ with the influence function

$$
\psi_t=\frac{m_2}{s^3}(r_t-m_1)-\frac{m_1}{2s^3}(r_t^2-m_2),\qquad s^2=m_2-m_1^2,
$$

estimated with a Newey-West long-run covariance (default), a stationary bootstrap, or the iid-normal formulas of Lo (2002) and Memmel (2003). Everything is computed per period and annualised at the end. Truncated-normal tails are evaluated in log space with the scaled complementary error function, because the window can lie thousands of standard errors away when the winner barely wins. Derivations, the forecast interval used for the out-of-sample check, the unlogged-trials sensitivity and the validation record are in [`docs/methodology.md`](docs/methodology.md).

## What is standard and what is new here

Standard: the conditional and hybrid inference itself is the method of Andrews, Kitagawa and McCloskey (2024); the Deflated Sharpe Ratio (Bailey and Lopez de Prado 2014), the Harvey-Liu haircut (2015) with Bonferroni and Benjamini-Hochberg-Yekutieli adjustments, Hansen's SPA and the Romano-Wolf StepM (through the optional `arch` package) are reimplemented or wrapped as published. The Lo/Memmel Sharpe covariance, Newey-West and the stationary bootstrap are textbook.

What `wincurse` adds, to my knowledge as of October 2026: a Python implementation of AKM (the authors' code is MATLAB; there is a small R package) and its application to backtested Sharpe ratios with a full covariance of Sharpe estimates. My own contributions are engineering and extensions around the published method: the influence-function covariance of Sharpe ratios and alphas with HAC and bootstrap variants; log-space numerics with every failed inversion counted; the forecast-interval check for out-of-sample performance (a port of the authors' forecast intervals); the trial ledger and a sensitivity analysis for unlogged trials (exact for the projection interval, a labelled scenario for the others); the tear sheet; and the extensions to selecting on the Sharpe ratio and reporting another statistic, top-$m$ selection and threshold selection. I derived those extensions with the same truncation argument and tested them by simulation; they have not been peer reviewed and the hybrid construction is not extended to top-$m$ and threshold rules.

The replication package of the paper (Harvard Dataverse, doi:10.7910/DVN/9GAYB5, CC0) is the oracle: for the authors' JobStart application this package reproduces the printed intervals of Table II up to the Monte Carlo error of the authors' own method (details in `docs/methodology.md`), for example the conditional interval ({{ akm_replication/exact/conditional/0 | 0 }}, {{ akm_replication/exact/conditional/1 | 0 }}) here against ({{ akm_replication/printed_table_ii/conditional/0 | int }}, {{ akm_replication/printed_table_ii/conditional/1 | int }}) in the paper.

### Other selection rules

<details>
<summary>Select on the Sharpe ratio, report the Fama-French alpha; top-m; threshold ("published only if t &gt; 2")</summary>

All three use the momentum grid (`scripts/exp_selection_rules.py`).

* Selecting on the Sharpe ratio and reporting the three-factor alpha of the same variant: naive alpha {{ selection_rules/select_sharpe_report_alpha/alpha_in_sample_naive | pct1 }} per year ({{ selection_rules/select_sharpe_report_alpha/naive_interval/0 | pct1 }} to {{ selection_rules/select_sharpe_report_alpha/naive_interval/1 | pct1 }}), hybrid estimate {{ selection_rules/select_sharpe_report_alpha/alpha_hybrid_estimate | pct1 }} ({{ selection_rules/select_sharpe_report_alpha/hybrid_interval/0 | pct1 }} to {{ selection_rules/select_sharpe_report_alpha/hybrid_interval/1 | pct1 }}); realised alpha after 1991: {{ selection_rules/select_sharpe_report_alpha/alpha_out_of_sample_realised | pct1 }}.
* Top five variants by Sharpe ratio, conditional inference for each: the conditional estimates are {{ selection_rules/top5_by_sharpe/0/conditional_estimate | 2 }}, {{ selection_rules/top5_by_sharpe/1/conditional_estimate | 2 }}, {{ selection_rules/top5_by_sharpe/2/conditional_estimate | 2 }}, {{ selection_rules/top5_by_sharpe/3/conditional_estimate | 2 }} and {{ selection_rules/top5_by_sharpe/4/conditional_estimate | 2 }} against naive values near {{ selection_rules/top5_by_sharpe/0/naive | 2 }}: when several near-twin variants compete for the top places the conditional estimates of the lower-ranked ones are very unstable (the 95% interval of the fifth reaches {{ selection_rules/top5_by_sharpe/4/conditional_interval/0 | 1 }}).
* Threshold t &gt; {{ selection_rules/threshold/threshold_t | 1 }}: {{ selection_rules/threshold/n_pass | int }} of {{ selection_rules/threshold/n_variants | int }} variants pass. Their mean naive Sharpe ratio is {{ selection_rules/threshold/mean_naive | 2 }} and their mean realised out-of-sample Sharpe ratio {{ selection_rules/threshold/mean_out_of_sample | 2 }}: with a genuine effect much larger than the noise there is little selection bias to remove. The conditional median-unbiased estimates are highly dispersed (median {{ selection_rules/threshold/median_conditional | 2 }}, 10th to 90th percentile {{ selection_rules/threshold/conditional_p10 | 2 }} to {{ selection_rules/threshold/conditional_p90 | 2 }}), as expected for variants just above the threshold, and the conditional intervals are {{ selection_rules/threshold/median_conditional_interval_width | 1 }} wide against {{ selection_rules/threshold/median_naive_interval_width | 1 }} for the naive ones. In this data the conditional estimator for threshold selection is not a useful point forecast.

</details>

## Installation

```bash
pip install git+https://github.com/Thesimpleex/wincurse   # numpy, scipy, pandas, rich
```

Extras: `arch` (Hansen SPA and Romano-Wolf StepM comparators), `plot` (matplotlib), `parquet` (reading Parquet files). From a clone: `pip install -e ".[dev]"`. Python 3.11 or newer.

## Quickstart

`examples/parameter_sweep.csv` holds {{ quickstart/n_trials | int }} synthetic strategy variants over {{ quickstart/n_obs | int }} days. By construction every variant has the same true annual Sharpe ratio, {{ quickstart/true_sharpe_of_every_variant | 1 }}, so the best one is the best of noise.

```python
import pandas as pd
from wincurse import analyze

returns = pd.read_csv("examples/parameter_sweep.csv", index_col="date", parse_dates=True)
result = analyze(returns, alpha=0.05, method="hybrid")

print(result.winner_name)
print(f"naive Sharpe      {result.winner.estimate:.2f}  {result.winner.naive}")
print(f"corrected Sharpe  {result.headline_estimate:.2f}  {result.headline_interval}")
print(f"effective number of trials {result.diagnostics.effective_trials:.1f} of {returns.shape[1]}")
```

prints the winner `{{ quickstart/winner | str }}`, a naive Sharpe ratio of {{ quickstart/naive | 2 }} with interval ({{ quickstart/naive_interval/0 | 2 }}, {{ quickstart/naive_interval/1 | 2 }}), a corrected (hybrid) estimate of {{ quickstart/hybrid_estimate | 2 }} with interval ({{ quickstart/hybrid_interval/0 | 2 }}, {{ quickstart/hybrid_interval/1 | 2 }}) and {{ quickstart/effective_trials | 1 }} effective trials. The naive interval excludes the true value {{ quickstart/true_sharpe_of_every_variant | 1 }}, the corrected one contains it (one draw, not evidence of coverage; see Figure 1 for that).

Compare the winner's realised performance on later data with a forecast interval for a new estimate (not with the interval for the in-sample true value):

```python
from wincurse import check_out_of_sample

check = check_out_of_sample(result, later_returns)  # DataFrame containing the winner's column
print(check.realised, check.forecast.hybrid, check.conditional_pvalue)
```

Record every variant while searching, so that $K$ stays honest:

```python
import numpy as np
from wincurse import Ledger

ledger = Ledger()
rng = np.random.default_rng(0)
common = rng.standard_normal(1000)


@ledger.track
def backtest(lookback):
    noise = rng.standard_normal(1000)
    return 0.01 * (0.7 * common + 0.7 * noise) + 0.0001 * lookback


for lookback in (5, 10, 20, 50):
    backtest(lookback)

result = ledger.analyze(periods_per_year=252, n_sim=20_000)
print(ledger.n_trials, "trials;", result.winner_name)
```

## Command line

```bash
wincurse analyze returns.csv --alpha 0.05 --method hybrid --report tearsheet.html
```

`returns.csv` (or Parquet) is a $T\times K$ panel of per-period excess returns, one column per tried variant; a non-numeric first column (dates) becomes the index and the observation frequency is inferred from it (otherwise pass `--periods-per-year`). Options:

* `--method {hybrid,conditional,projection,naive}` selects the headline; all are computed and shown.
* `--select {estimate,t}` selects on the estimate or on the t-statistic; `--statistic {sharpe,mean}`.
* `--cov {hac,iid,normal,bootstrap}` covariance estimator; `--shrink` Ledoit-Wolf constant-correlation shrinkage.
* `--oos later.csv` evaluates the winner out of sample against forecast intervals.
* `--hidden-trials N [--hidden-corr R]` sensitivity to $N$ unlogged trials.
* `--arch` adds Hansen's SPA and the Romano-Wolf StepM; `--coverage-reps N` size of the coverage simulation under the estimated covariance (0 switches it off); `--n-sim`, `--seed`.

## Reproduce all results

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -e ".[dev,docs]"
.venv/bin/python scripts/reproduce_all.py          # downloads data, runs every study, redraws every figure
```

`scripts/reproduce_all.py` runs, in order: the AKM replication (`exp_akm_replication.py`; the fixture comes from `extract_akm_fixture.py`, which downloads the 267 MB replication package once), the coverage study (`exp_coverage.py`, the longest step, tens of minutes with several workers; finished cells are cached in `data/cache/coverage`), the two case studies (`exp_momentum.py`, `exp_technical.py`), the other selection rules (`exp_selection_rules.py`), `make_example.py`, the figures (`make_figures.py`), the screenshots (`make_screenshots.py`, needs Google Chrome) and finally renders this README and `docs/methodology.md` from `docs/templates` and `docs/results.json`. Every number above is read from `docs/results.json`, which also records package versions and data vintages (Ken French files, CRSP vintage {{ momentum/data/crsp_vintage | str }}, last observation {{ momentum/data/last_observation | str }}). Raw downloads go to `data/raw/` and are never committed. Tests (`pytest`, no network access, under a minute and a half) check the numerics against closed forms, `mpmath` at 50 digits, `scipy`, simulations and the AKM replication package.

## Limitations and next steps

* Coverage is exact only under normality with a known covariance; with estimated covariance matrices and real returns it is asymptotic. The study shows where this matters (serial correlation that the covariance estimator ignores, very short samples), not that it never does.
* Selection is the argmax on the same data used for inference. Human discretion, early stopping, grids designed after looking at the data and non-stationarity are not covered, and a decayed out-of-sample Sharpe ratio can have causes other than selection.
* Unlogged trials are not identified for conditional and hybrid inference. Only the projection interval is exact under an assumed covariance; the scenario simulation is not a confidence interval.
* With $K$ close to $T$ the covariance of near-twin trials is poorly determined and the corrected estimate can move substantially with the estimator (momentum case study); shrinkage changes the correlation structure and should be reported next to the unshrunk result.
* The conditional estimator is unstable when the winner barely wins or, for threshold selection, close to the threshold. The hybrid estimator is not exactly median-unbiased (bias at most $\beta/2$).
* Case studies use no transaction costs and the grids reflect post-1990 literature.
* Not implemented: the OSAP (Open Source Asset Pricing) case study, locally simultaneous inference of Zrnic and Fithian, the conditional test of Pav, a hybrid construction for top-$m$ and threshold rules, and a documentation site beyond this file and `docs/methodology.md`.

## References

* Andrews, I., Kitagawa, T. and McCloskey, A. (2024). Inference on winners. *Quarterly Journal of Economics* 139(1), 305-358.
* Bailey, D. H. and Lopez de Prado, M. (2014). The deflated Sharpe ratio: correcting for selection bias, backtest overfitting and non-normality. *Journal of Portfolio Management* 40(5), 94-107.
* Hansen, P. R. (2005). A test for superior predictive ability. *Journal of Business and Economic Statistics* 23(4), 365-380.
* Harvey, C. R. and Liu, Y. (2015). Backtesting. *Journal of Portfolio Management* 42(1), 13-28.
* Ledoit, O. and Wolf, M. (2003). Improved estimation of the covariance matrix of stock returns with an application to portfolio selection. *Journal of Empirical Finance* 10(5), 603-621.
* Lo, A. W. (2002). The statistics of Sharpe ratios. *Financial Analysts Journal* 58(4), 36-52.
* Memmel, C. (2003). Performance hypothesis testing with the Sharpe ratio. *Finance Letters* 1, 21-23.
* Newey, W. K. and West, K. D. (1994). Automatic lag selection in covariance matrix estimation. *Review of Economic Studies* 61(4), 631-653.
* Politis, D. N. and Romano, J. P. (1994). The stationary bootstrap. *Journal of the American Statistical Association* 89(428), 1303-1313.
* Romano, J. P. and Wolf, M. (2005). Stepwise multiple testing as formalized data snooping. *Econometrica* 73(4), 1237-1282.
* Sullivan, R., Timmermann, A. and White, H. (1999). Data-snooping, technical trading rule performance, and the bootstrap. *Journal of Finance* 54(5), 1647-1691.

Data: Kenneth R. French Data Library (49 industry portfolios, Fama-French factors). The AKM replication package is cited above and is the source of the JobStart estimates stored in `tests/data/akm_jobstart.npz` (CC0).

MIT licence, see [LICENSE](LICENSE). Cite via [CITATION.cff](CITATION.cff).
