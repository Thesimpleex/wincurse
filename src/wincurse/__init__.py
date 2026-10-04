"""wincurse: selection-corrected Sharpe ratios for the best of many backtests.

Implements the conditional and hybrid inference of Andrews, Kitagawa and McCloskey
(2024, "Inference on Winners") for the performance of the best of K trials, with
delta-method / HAC / bootstrap covariances of Sharpe ratios, comparators (Deflated
Sharpe Ratio, Harvey-Liu haircuts, Bonferroni), diagnostics, a trial ledger and a
command-line tear sheet.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from .analysis import Analysis, DeflatedSharpe, analyze, infer_periods_per_year
from .comparators import (
    ArchTests,
    Haircut,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    harvey_liu_haircut,
    multiple_testing_pvalues,
    spa_stepm,
)
from .critical import (
    equicorrelated_critical_value,
    independent_critical_value,
    projection_critical_value,
)
from .diagnostics import Diagnostics, effective_number_of_trials, winner_margin
from .exceptions import RootFindingError, WincurseWarning
from .ledger import Ledger, default_ledger, track
from .outofsample import OutOfSample, check_out_of_sample, forecast_pvalue
from .performance import (
    JointEstimate,
    PerformanceEstimate,
    alpha_influence_functions,
    estimate_joint,
    estimate_performance,
    influence_functions,
    ledoit_wolf_constant_correlation,
    long_run_covariance,
    nearest_psd,
    newey_west_bandwidth,
    sharpe_ratios,
    stationary_bootstrap_covariance,
)
from .selection import SelectedInference, threshold_inference, top_m_inference
from .sensitivity import (
    ProjectionSensitivity,
    ScenarioSummary,
    augmented_covariance,
    nearest_neighbour_correlation,
    unlogged_projection,
    unlogged_scenario,
)
from .simulation import CoverageResult, simulate_known_sigma
from .truncnorm import mean_quantile, tn_cdf, tn_ppf, tn_sf
from .winners import (
    ForecastInterval,
    Interval,
    WinnerInference,
    forecast_interval,
    selection_bounds,
    winner_inference,
    winner_inference_cross,
)

try:
    __version__ = version("wincurse")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.1.0"

__all__ = [
    "Analysis",
    "ArchTests",
    "CoverageResult",
    "DeflatedSharpe",
    "Diagnostics",
    "ForecastInterval",
    "Haircut",
    "Interval",
    "JointEstimate",
    "Ledger",
    "OutOfSample",
    "PerformanceEstimate",
    "ProjectionSensitivity",
    "RootFindingError",
    "ScenarioSummary",
    "SelectedInference",
    "WincurseWarning",
    "WinnerInference",
    "__version__",
    "alpha_influence_functions",
    "analyze",
    "augmented_covariance",
    "check_out_of_sample",
    "default_ledger",
    "deflated_sharpe_ratio",
    "effective_number_of_trials",
    "equicorrelated_critical_value",
    "estimate_joint",
    "estimate_performance",
    "expected_max_sharpe",
    "forecast_interval",
    "forecast_pvalue",
    "harvey_liu_haircut",
    "independent_critical_value",
    "infer_periods_per_year",
    "influence_functions",
    "ledoit_wolf_constant_correlation",
    "long_run_covariance",
    "mean_quantile",
    "multiple_testing_pvalues",
    "nearest_neighbour_correlation",
    "nearest_psd",
    "newey_west_bandwidth",
    "projection_critical_value",
    "selection_bounds",
    "sharpe_ratios",
    "simulate_known_sigma",
    "spa_stepm",
    "stationary_bootstrap_covariance",
    "threshold_inference",
    "tn_cdf",
    "tn_ppf",
    "tn_sf",
    "top_m_inference",
    "track",
    "unlogged_projection",
    "unlogged_scenario",
    "winner_inference",
    "winner_inference_cross",
    "winner_margin",
]
