# Methodology

This note derives what `wincurse` computes, states the assumptions, and documents how
each piece was checked. Numbers come from `docs/results.json` (regenerate everything with
`python scripts/reproduce_all.py`).

## 1. Setting

A researcher tries $K$ variants of a strategy on the same $T$ periods and reports the one
with the largest estimated performance. Let $X \in \mathbb{R}^K$ be the vector of
estimates (annualised Sharpe ratios, or means) and assume, as in Andrews, Kitagawa and
McCloskey (2024, "AKM"),

$$
X \sim N(\mu, \Sigma), \qquad \Sigma \text{ known},
$$

where $\Sigma$ is the covariance of the estimation error. The reported trial is
$k = \arg\max_j X_j$ (or $\arg\max_j X_j/\sqrt{\Sigma_{jj}}$ when selecting on the
t-statistic) and the target is $\mu_k$, the *true* performance of the selected trial. The
naive estimate $X_k$ is biased upwards and $X_k \pm 1.96\sqrt{\Sigma_{kk}}$ covers $\mu_k$
with probability below 95%; with $K = 100$ equally good trials of correlation 0.7 the
simulated coverage is {{ coverage/known_sigma/known_K100_rho0.7_equal_n10000/naive_coverage | pct1 }}.

With an estimated $\Sigma$ and non-normal data all statements below hold asymptotically
(AKM, Section 5), not in finite samples.

## 2. Conditional inference

Fix the selected index $k$ and define

$$
Z = X - \frac{\Sigma_{\cdot k}}{\Sigma_{kk}} X_k ,
$$

so that $Z$ is independent of $X_k$ (it is uncorrelated with it, and the pair is jointly
normal). Write $X_j = Z_j + \tfrac{\Sigma_{jk}}{\Sigma_{kk}} X_k$. Trial $k$ is selected iff
$X_k \ge X_j$ for all $j$, that is

$$
\frac{\Sigma_{kk} - \Sigma_{jk}}{\Sigma_{kk}}\, X_k \;\ge\; Z_j - Z_k \qquad \text{for all } j\neq k .
$$

Since $Z_k = 0$, each $j$ gives a bound on $X_k$:

$$
X_k \ge \frac{\Sigma_{kk} Z_j}{\Sigma_{kk}-\Sigma_{jk}} \quad (\Sigma_{jk} < \Sigma_{kk}),
\qquad
X_k \le \frac{\Sigma_{kk} Z_j}{\Sigma_{kk}-\Sigma_{jk}} \quad (\Sigma_{jk} > \Sigma_{kk}),
$$

and trials with $\Sigma_{jk}=\Sigma_{kk}$ impose only $Z_j \le 0$, which holds. Let
$L(Z)$ be the largest lower bound and $U(Z)$ the smallest upper bound (both are the empty
maximum/minimum, $-\infty$ and $+\infty$, if no trial gives one). Then, conditional on
selection and on $Z$,

$$
X_k \;\sim\; N(\mu_k, \Sigma_{kk}) \text{ truncated to } [L(Z), U(Z)] .
$$

Let $F(x;\mu)$ be the CDF of this truncated normal. It is strictly decreasing in $\mu$
(the truncated normal family has monotone likelihood ratio), so the equation
$F(X_k;\mu)=p$ has a unique root $\hat\mu_p$. Then

$$
\hat\mu_{1/2} \text{ is median-unbiased}, \qquad
\left[\hat\mu_{1-\alpha/2},\ \hat\mu_{\alpha/2}\right] \text{ has conditional coverage } 1-\alpha ,
$$

conditional on selection of $k$ and on $Z$, hence also unconditionally. These are the
**conditional estimator and interval**. In `wincurse`, `selection_bounds` returns
$(L, U)$ and `winner_inference` computes the estimator and interval.

*Numerical details.* When the winner barely beats the runner-up the window
$[L, U]$ lies many standard errors from $\mu_k$ and the textbook formula
$\frac{\Phi(z)-\Phi(a)}{\Phi(b)-\Phi(a)}$ returns `nan`. The CDF is evaluated as a ratio of
survival functions $Q=1-\Phi$ in log space with the scaled complementary error function,
$\log Q(t) = -t^2/2 + \log\bigl(\operatorname{erfcx}(t/\sqrt 2)/2\bigr)$, combining the
large terms as $(t-a)(t+a)/2$. The root in $\mu$ is bracketed by doubling and found by
bisection; for two independent trials with a lead of $10^{-3}$ standard errors the conditional
interval reaches thousands of standard errors (figure `endpoints_vs_gap`). A trial with
$|\Sigma_{kk}-\Sigma_{jk}| \le 10^{-12}\Sigma_{kk}$ is treated as the equality case; a
covariance that is exactly singular in this way but inconsistent with the data raises a
`ValueError` instead of producing arbitrary bounds. Failed inversions are never
skipped: they raise `RootFindingError` in the public functions and are counted as misses in
every coverage computation (`CoverageResult.n_failed`; {{ coverage/known_sigma/known_K1000_rho0.5_equal_n10000/n_failed | int }} failures in the 10,000 replications at K = 1000 and none in any cell of the study).

## 3. Hybrid inference (the default)

The conditional interval can be extremely wide. AKM's hybrid construction additionally
conditions on the event that $\mu_k$ is not far from $X_k$. Let $c_\beta$ be the
$1-\beta$ quantile of $\max_j |\xi_j|/\sqrt{\Sigma_{jj}}$ with $\xi\sim N(0,\Sigma)$
(simulated; for independent trials $c_\beta=\Phi^{-1}\!\bigl(\tfrac{1+(1-\beta)^{1/K}}{2}\bigr)$).
The *projection* interval $X_k \pm c_\alpha\sqrt{\Sigma_{kk}}$ has coverage $1-\alpha$
whatever rule selected $k$. The hybrid procedure conditions in addition on
$|X_k-\mu_k|\le c_\beta\sqrt{\Sigma_{kk}}$, which has probability $1-\beta$, so that $X_k$ is
truncated to

$$
\bigl[\max(L,\ \mu_k - c_\beta\sqrt{\Sigma_{kk}}),\ \min(U,\ \mu_k + c_\beta\sqrt{\Sigma_{kk}})\bigr],
$$

and runs the conditional construction at level $\gamma=\tfrac{\alpha-\beta}{1-\beta}$. The result
lies inside the projection set for level $\beta$, its unconditional coverage lies between
$1-\alpha$ and $(1-\alpha)/(1-\beta)$, and the absolute median bias of the hybrid estimator
is at most $\beta/2$ (AKM, Section 4). The default is $\beta=\alpha/10$.
The hybrid estimator is therefore *nearly*, not exactly, median-unbiased; only the
conditional one is median-unbiased. In the simulations hybrid coverage is
{{ coverage/known_sigma/known_K100_rho0.7_equal_n10000/hybrid_coverage | pct1 }} against
{{ coverage/known_sigma/known_K100_rho0.7_equal_n10000/conditional_coverage | pct1 }} for the conditional
interval (K = 100, correlation 0.7, known covariance).

The hybrid interval is not always shorter than the conditional one in every
configuration. It is confined to the projection set, which makes it the safer default; for
a very narrow win it becomes short but displaced towards $X_k-c_\beta\sqrt{\Sigma_{kk}}$
(figure `endpoints_vs_gap`), because its guarantee is unconditional on the near-tie
event, which is rare.

## 4. Sampling covariance of Sharpe ratios

Let $m_1=E r$, $m_2=E r^2$ and $SR=m_1/\sqrt{m_2-m_1^2}$ per period. The delta method on
the moment vector $(r_t, r_t^2)$ gives the influence function

$$
\psi_t = \frac{m_2}{s^3}\,(r_t-m_1) - \frac{m_1}{2 s^3}\,(r_t^2-m_2), \qquad s^2=m_2-m_1^2,
$$

and $\operatorname{Cov}(\widehat{SR}_i,\widehat{SR}_j)\approx \Omega_{ij}/T$ with $\Omega$ the
long-run covariance of $(\psi_{it},\psi_{jt})$. `wincurse` offers four estimators of $\Omega$:

* `normal`: iid normal returns, $\Omega_{ij}=\rho_{ij}+\tfrac12 SR_iSR_j\rho_{ij}^2$
  (Lo 2002; Memmel 2003; $\operatorname{Var}(\widehat{SR}_i)=(1+SR_i^2/2)/T$);
* `iid`: the sample covariance of $\psi$ (robust to skewness and kurtosis);
* `hac` (default): Bartlett-kernel Newey-West with the automatic bandwidth of Newey and
  West (1994), evaluated as one matrix product and positive semi-definite by construction;
* `bootstrap`: the stationary bootstrap of Politis and Romano (1994).

Everything is computed per period and annualised at the end: multiplying the estimates by
$\sqrt q$ multiplies $\Sigma$ by $q$, and all intervals are equivariant under this joint
scaling. (Plugging an *annualised* Sharpe ratio into $(1+SR^2/2)/T$ overstates the
standard error by 50% for a daily Sharpe ratio of 0.1; a test covers this.) A matrix that is
not positive definite is repaired by raising eigenvalues to $10^{-10}\lambda_{\max}$ while
keeping the variances, with a warning. The optional Ledoit-Wolf shrinkage (2003) is applied
to the influence functions with a *constant-correlation* target, which keeps every variance;
it changes correlations and with them the truncation bounds and $c_\beta$, so the case
studies report results with and without it.

## 5. Forecast intervals for out-of-sample checks

Comparing a realised out-of-sample Sharpe ratio with an interval for the *in-sample* true
Sharpe ratio is a category error: the realised value has its own sampling error. AKM
(Section 4.1, Appendix E) give an interval for a new, independent estimate $X_2$ of the
winner's performance, variance $\Sigma_2$, under the hypothesis that the winner's true
performance is unchanged. Write $\Sigma_{12}=\Sigma_{kk}+\Sigma_{2}$ and $D=X_k-X_2\sim
N(0,\Sigma_{12})$. Redefining $Z = X - \Sigma_{\cdot k} D/\Sigma_{12}$ turns selection into
bounds on $D$ of the same form,

$$
L_{12} = D + \Sigma_{12}\max_{\Sigma_{jk}<\Sigma_{kk}} \frac{X_j-X_k}{\Sigma_{kk}-\Sigma_{jk}},\qquad
U_{12} = D + \Sigma_{12}\min_{\Sigma_{jk}>\Sigma_{kk}} \frac{X_j-X_k}{\Sigma_{kk}-\Sigma_{jk}},
$$

and the set of $X_2$ for which $D$ is not rejected by the truncated normal
$N(0,\Sigma_{12})$ on $[L_{12},U_{12}]$ is the conditional forecast interval; the hybrid
version intersects the window with $\pm c_\beta\sqrt{\Sigma_{12}}$, exactly as in the
authors' `L12_fn.m`, `U12_fn.m` and `empirical.m`. `forecast_interval` implements this and
`check_out_of_sample` evaluates the winner on later data, with $\Sigma_2$ the sampling
variance of the winner's out-of-sample Sharpe ratio. The conditional two-sided p-value is
$2\min(F,1-F)$ with $F$ the truncated CDF at the observed $D$.

## 6. Unlogged trials

The truncation bounds depend on the realised values $Z_j$ of every trial, which are
unobserved for trials that were run but not logged, and integrating them out introduces
their unknown means as nuisance parameters. Conditional and hybrid inference are therefore
not identified from an assumed correlation alone. Only the projection critical value depends
on $\Sigma$ alone. `unlogged_projection` recomputes $c_\alpha$ for the covariance augmented
by $H$ hidden trials, each a noisy copy of a logged trial with a stated correlation to it
(positive semi-definite by construction), and reports the exact projection interval.
`unlogged_scenario` additionally simulates hidden outcomes under stated assumptions (all true
values equal, hidden estimates drawn from their conditional normal law, draws in which a
hidden trial would have won discarded) and is labelled a scenario, not a confidence interval.

## 7. Other selection rules and reported statistics

*Select on one statistic, report another.* Let $(X, Y)$ be jointly normal, select
$k=\arg\max X_j$ and report $Y_k$ (Sharpe ratio selected, Fama-French alpha reported). With
$c_j=\operatorname{Cov}(X_j,Y_k)/\operatorname{Var}(Y_k)$ write $X_j=Z_j+c_jY_k$ with $Z$
independent of $Y_k$; selection is $(c_k-c_j)Y_k\ge Z_j-Z_k$, an interval for $Y_k$, and
the truncated-normal argument of Section 2 applies to $Y_k$ (`winner_inference_cross`; for
$X=Y$ it coincides with `winner_inference`, which a test verifies). The joint covariance of
Sharpe ratios and alphas comes from the stacked influence functions; the influence function
of an OLS intercept is $h_t e_{it}$ with $h_t=e_1'(E[ww'])^{-1}w_t$, $w_t=(1,f_t)$.

*Threshold selection* ("published only if $t>c$"): trial $j$ is reported iff
$X_j>c\sqrt{\Sigma_{jj}}$, which involves $X_j$ alone, so $X_j$ is a normal truncated to
$[c\sqrt{\Sigma_{jj}},\infty)$ and the same inversion gives a median-unbiased estimate and an
exact conditional interval (`threshold_inference`). *Top-$m$ selection*: trial $k$ is
reported iff fewer than $m$ trials exceed it. With $Z$ as in Section 2 the event is a union of
intervals for $X_k$ (one for $m=1$); the truncated normal CDF is monotone in the mean for any
truncation set, so the inversion carries over (`top_m_inference`). Both give conditional
inference only; the hybrid construction is not extended to them.

## 8. Validation

* $K=1$ reduces every method to the textbook interval; independent equal-variance arms match
  `scipy.stats.truncnorm`; the truncated-normal CDF and quantiles match `mpmath` at 50 digits
  in extreme tails (`tests/test_truncnorm.py`).
* The conditional pivot $F(X_k;\mu_k)$ is uniform (Kolmogorov-Smirnov), conditional coverage
  equals $1-\alpha$ within Monte Carlo error and the conditional estimator is median-unbiased;
  hybrid coverage is at least $1-\alpha$ and its median bias at most $\beta/2$ plus Monte
  Carlo error (`tests/test_winners.py`).
* The iid Sharpe covariance matches simulation; the HAC version matches simulation under
  AR(1) and GARCH(1,1) returns where the normal formula does not (`tests/test_performance.py`).
### Replication of AKM

The authors' replication package (Harvard Dataverse,
doi:10.7910/DVN/9GAYB5, CC0) supplies the JobStart estimates $\mu$ (13 policies, winner
policy 2 with estimate {{ akm_replication/setting/winner_estimate | int }} and standard error
{{ akm_replication/setting/winner_se | 2 }}) and the diagonal covariance matrix; the script
`scripts/extract_akm_fixture.py` stores the two small arrays. Table II of the paper prints
the endpoints below next to the exact values of this package (computed with exact
critical values for the diagonal covariance and verified against an independent
50-digit `mpmath` evaluation, `tests/test_winners.py`).

| | paper (Table II) | `wincurse` (exact) |
|---|---|---|
| naive | ({{ akm_replication/printed_table_ii/naive/0 | int }}, {{ akm_replication/printed_table_ii/naive/1 | int }}) | ({{ akm_replication/exact/naive/0 | int }}, {{ akm_replication/exact/naive/1 | int }}) |
| conditional | ({{ akm_replication/printed_table_ii/conditional/0 | int }}, {{ akm_replication/printed_table_ii/conditional/1 | int }}) | ({{ akm_replication/exact/conditional/0 | int }}, {{ akm_replication/exact/conditional/1 | int }}) |
| projection | ({{ akm_replication/printed_table_ii/projection/0 | int }}, {{ akm_replication/printed_table_ii/projection/1 | int }}) | ({{ akm_replication/exact/projection/0 | int }}, {{ akm_replication/exact/projection/1 | int }}) |
| hybrid | ({{ akm_replication/printed_table_ii/hybrid/0 | int }}, {{ akm_replication/printed_table_ii/hybrid/1 | int }}) | ({{ akm_replication/exact/hybrid/0 | int }}, {{ akm_replication/exact/hybrid/1 | int }}) |

The printed numbers are Monte Carlo output (sorted truncated-normal draws, up to
$5\times10^5$ draws, a noisy root search under MATLAB's `rng(1)`), so an exact
implementation differs in the last digits and tests must not be tuned to match them
exactly. Upper endpoints, projection endpoints and point estimates agree to within about
two units on a standard error of {{ akm_replication/setting/winner_se | int }}. The lower
conditional and hybrid endpoints differ by about nine and ten units. Re-implementing the
authors' algorithm and running it with {{ akm_replication/authors_algorithm_emulation_lower_endpoints/n_seeds | int }}
seeds gives lower endpoints with a standard deviation of
{{ akm_replication/authors_algorithm_emulation_lower_endpoints/500000/conditional/sd | 1 }} (conditional) with $R_{cv}=5\times10^5$
draws and {{ akm_replication/authors_algorithm_emulation_lower_endpoints/100000/conditional/sd | 1 }} with $10^5$ draws (the replication code
lists $10^4$, $10^5$ and $5\times10^5$ as alternatives), centred on the exact value (means
{{ akm_replication/authors_algorithm_emulation_lower_endpoints/100000/conditional/mean | 1 }} and
{{ akm_replication/authors_algorithm_emulation_lower_endpoints/100000/hybrid/mean | 1 }} against exact
{{ akm_replication/exact/conditional/0 | 1 }} and {{ akm_replication/exact/hybrid/0 | 1 }}). The printed lower endpoints lie 1.0 and 1.5 standard deviations
from the emulated mean at $10^5$ draws, but more than two at $5\times10^5$, so they are consistent with a smaller
$R_{cv}$ than the one set in the replication code. The forecast intervals of
Table III (replication standard errors 1315 and 2607) are reproduced with upper endpoints
within two units and lower endpoints within about 14 to 22 units (see `results.json`,
`akm_replication/forecast_intervals`).

### Deflated Sharpe Ratio

The Deflated Sharpe Ratio reproduces the worked example of Bailey and Lopez de Prado
(2014): $N=100$ trials, annual Sharpe ratio 2.5 over $T=1250$ days, skewness $-3$, kurtosis
10, variance of the trial Sharpe ratios 0.5 per year: $SR_0=0.1132$ per day and
$DSR=0.900$; with $N=46$ the DSR is $0.9505$ (`tests/test_comparators.py`).

## 9. Monte Carlo design

Returns are daily, ten years ($T=2520$), drawn from a one-factor equicorrelation model
(correlation $\rho$ between all pairs of strategies), 1% daily volatility, so that the annualised
Sharpe ratios of the trials are the stated values. The *least-separated* configuration gives
every trial the same true annual Sharpe ratio ({{ coverage/design/equal_true_sharpe | 1 }}); the *spread*
configuration spaces them evenly from 0 to 1. Two designs are run.

* **Known covariance** ($K\in\{10,100,1000\}$, $\rho\in\{0.3,0.5,0.7,0.9\}$, both
  configurations, {{ coverage/design/reps_known_sigma | int }} replications): the estimates are drawn from $N(\mu,\Sigma)$ with the true
  sampling covariance of the Sharpe ratios and the procedures use it. This is the
  theoretical setting.
* **Estimated covariance** ({{ coverage/design/reps_feasible | int }} replications): every replication simulates the return
  panel, estimates the Sharpe ratios and their covariance (HAC unless stated) and applies the
  procedures with critical values simulated from the *estimated* covariance. For normal
  returns the same replication is also analysed with the true covariance. Monthly data would
  make the covariance of $K=1000$ trials rank-deficient (rank at most $T-1$), so the study uses
  daily data. Student-t(5), GARCH(1,1) and AR(1) ($\phi=0.2$) returns with three covariance
  estimators show robustness and limits.

Every replication is scored. The coverage of a procedure is the share of replications in
which its interval contains the *true* Sharpe ratio of the trial that was selected in that
replication.

## 10. Case-study data

Industry portfolios come from the Ken French Data Library (49 industries, value-weighted
monthly returns, CRSP vintage {{ momentum/data/crsp_vintage | str }}, last observation
{{ momentum/data/last_observation | str }}; raw files are downloaded at run time into
`data/raw/` and never committed). The file holds several stacked blocks and codes missing
industries as -99.99 or -999; only the first (value-weighted monthly) block is read and
missing values become NaN. A variant ranks only industries with a complete formation window
and needs at least $2n$ of them; the common sample of all 256 variants starts
{{ momentum/in_sample/start | str }}. Held industries with a missing return in a holding month contribute zero
({{ momentum/data/grid/held_industry_months_with_missing_return | int }} industry-months in total, all of one industry in 1943-44). Variants with in-sample
correlation above $1-10^{-10}$ with an earlier one would be dropped (AKM's Lemma 1 requires
no two perfectly correlated trials): {{ momentum/n_duplicates_dropped | int }} in the momentum grid and
{{ technical/n_duplicates_dropped | int }} among the technical rules.

The grid ranges (for example 12-month lookbacks) come from literature written after 1990
and partly on overlapping data; the correction accounts for the 256 logged variants and not
for that prior snooping. Overlapping holding periods make the weights, not the returns,
overlap; the automatic HAC bandwidth selected for the momentum panel is
{{ momentum/bandwidth | int }} and for the daily technical rules {{ technical/bandwidth | int }}.

## 11. Limitations

* Inference is asymptotic: coverage is exact only under normality with a known covariance. The
  covariance is estimated (HAC, bootstrap or shrinkage) and the study reports where that
  matters (autocorrelated returns with a covariance estimator that ignores them; very short
  samples).
* Selection is the argmax (or a threshold or top-$m$ rule) on the *same* data used for
  inference; human discretion, early stopping and data-dependent grid design are not covered.
* Unlogged trials are not identified for conditional and hybrid inference; the sensitivity
  analysis is exact only for the projection interval.
* Non-stationarity is not fixed: a decayed out-of-sample Sharpe ratio can be selection,
  regime change or transaction costs, and the forecast interval tests only the first against
  the data under a no-change hypothesis.
* Constant-correlation shrinkage changes the correlation structure and with it the results;
  K close to T leaves the covariance of near-twin trials poorly determined (the momentum case
  study shows this).
* The conditional estimator is extremely unstable when the winner barely wins and for
  threshold selection close to the threshold. The hybrid estimator inherits this only to
  within $\beta/2$ of median bias.

## References

* Andrews, I., Kitagawa, T. and McCloskey, A. (2024). Inference on winners. *Quarterly
  Journal of Economics* 139(1), 305-358.
* Bailey, D. H. and Lopez de Prado, M. (2014). The deflated Sharpe ratio: correcting for
  selection bias, backtest overfitting and non-normality. *Journal of Portfolio Management*
  40(5), 94-107.
* Harvey, C. R. and Liu, Y. (2015). Backtesting. *Journal of Portfolio Management* 42(1),
  13-28.
* Ledoit, O. and Wolf, M. (2003). Improved estimation of the covariance matrix of stock
  returns with an application to portfolio selection. *Journal of Empirical Finance* 10(5),
  603-621.
* Lo, A. W. (2002). The statistics of Sharpe ratios. *Financial Analysts Journal* 58(4),
  36-52.
* Memmel, C. (2003). Performance hypothesis testing with the Sharpe ratio. *Finance Letters*
  1, 21-23.
* Newey, W. K. and West, K. D. (1994). Automatic lag selection in covariance matrix
  estimation. *Review of Economic Studies* 61(4), 631-653.
* Politis, D. N. and Romano, J. P. (1994). The stationary bootstrap. *Journal of the American
  Statistical Association* 89(428), 1303-1313.
