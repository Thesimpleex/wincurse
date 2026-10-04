"""Command-line interface: ``wincurse analyze returns.csv --method hybrid --report out.html``."""

from __future__ import annotations

import argparse
import contextlib
import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from rich import box
from rich.console import Console
from rich.table import Table

from . import __version__
from .analysis import Analysis, analyze
from .exceptions import WincurseWarning
from .outofsample import OutOfSample, check_out_of_sample
from .report import write_report
from .simulation import CoverageResult, simulate_known_sigma

__all__ = ["main", "read_returns"]


def read_returns(path: str | Path, dropna: bool = False) -> pd.DataFrame:
    """Read a ``T x K`` return panel from CSV or Parquet.

    A first column that is not numeric (typically dates) becomes the row index.
    """
    path = Path(path)
    if not path.exists():
        raise ValueError(f"file not found: {path}")
    if path.suffix.lower() in (".parquet", ".pq"):
        frame = pd.read_parquet(path)
    else:
        frame = pd.read_csv(path)
        first = frame.columns[0]
        if not pd.api.types.is_numeric_dtype(frame[first]):
            index = frame[first]
            with contextlib.suppress(ValueError, TypeError):
                index = pd.to_datetime(index)
            frame = frame.drop(columns=first).set_axis(index, axis=0)
    try:
        frame = frame.apply(pd.to_numeric)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"non-numeric return columns in {path.name}: {exc}") from exc
    n_missing = int(frame.isna().to_numpy().sum())
    if n_missing:
        if not dropna:
            raise ValueError(
                f"{n_missing} missing values in {path.name}; trials need a common sample "
                "(pass --dropna to drop incomplete rows)"
            )
        frame = frame.dropna()
    return frame


def _fmt(v: float, digits: int = 2) -> str:
    return "n/a" if not math.isfinite(v) else f"{v:.{digits}f}"


def _pct(v: float) -> str:
    return "n/a" if not math.isfinite(v) else f"{v:.0%}"


def _interval(iv) -> str:
    return f"[{_fmt(iv.lower):>6}, {_fmt(iv.upper):>6}]"


def print_analysis(
    a: Analysis,
    console: Console,
    source: str,
    coverage: CoverageResult | None = None,
    statistic_label: str = "Sharpe",
    oos: OutOfSample | None = None,
) -> None:
    """Render the analysis as a terminal summary."""
    p, w, d = a.performance, a.winner, a.diagnostics
    level = f"{100 * (1 - w.alpha):g}%"
    period = f" ({a.period[0]} to {a.period[1]})" if a.period else ""
    cov = {
        "hac": f"HAC, Bartlett bandwidth {p.bandwidth}",
        "iid": "iid influence functions",
        "normal": "iid normal (Lo/Memmel)",
        "bootstrap": "stationary bootstrap",
    }[p.cov_method]
    console.print(
        f"[bold]wincurse {__version__}[/] [dim]{source} | {p.n_obs:,} obs{period} | "
        f"K = {p.n_strategies} trials | {p.periods_per_year:g}/year[/]"
    )
    console.print(
        f"[dim]covariance: {cov}{' + shrinkage' if p.shrinkage is not None else ''} | "
        f"alpha = {w.alpha:g}, beta = {w.beta:g} | seed {a.seed} | "
        f"{a.n_sim:,} critical-value draws[/]"
    )
    console.print()
    console.print(
        f"[bold]Winner[/]  {a.winner_name}  [dim](trial {w.index + 1} of {p.n_strategies}, "
        f"selected on {'t-statistic' if w.selection == 't' else 'estimate'})[/]"
    )
    console.print()
    table = Table(box=box.SIMPLE_HEAD, pad_edge=False, show_edge=False)
    table.add_column("Procedure")
    table.add_column(f"{statistic_label}", justify="right")
    table.add_column(f"{level} interval", justify="right")
    table.add_column("width", justify="right")
    rows = [
        ("Naive", w.estimate, w.naive, "naive"),
        ("Conditional (median-unbiased)", w.conditional_estimate, w.conditional, "conditional"),
        ("Hybrid", w.hybrid_estimate, w.hybrid, "hybrid"),
        ("Projection", float("nan"), w.projection, "projection"),
        ("Bonferroni", float("nan"), w.bonferroni, "bonferroni"),
    ]
    for name, est, iv, key in rows:
        style = "bold" if key == a.method else None
        label = f"{name} *" if key == a.method else name
        table.add_row(label, _fmt(est), _interval(iv), _fmt(iv.width), style=style)
    console.print(table)
    console.print(
        f"[dim]* headline ({a.method}).[/] Corrected {statistic_label} "
        f"[bold]{_fmt(a.headline_estimate)}[/] against naive [bold]{_fmt(w.estimate)}[/]: "
        f"haircut [yellow]{_pct(a.haircut)}[/]"
    )
    if a.dsr is not None and a.haircuts is not None:
        s = a.dsr
        console.print()
        console.print("[bold]Comparators[/]  [dim](tests against zero; no corrected estimate)[/]")
        comp = Table(box=box.SIMPLE_HEAD, pad_edge=False, show_edge=False)
        comp.add_column("Method")
        comp.add_column("Result", justify="right")
        comp.add_row(f"Deflated Sharpe ratio, N = K = {s.n_trials:.0f}", f"{s.dsr:.3f}")
        comp.add_row(
            f"Deflated Sharpe ratio, N = {s.effective_trials:.1f} effective",
            f"{s.dsr_effective:.3f}",
        )
        hb, hy = a.haircuts["bonferroni"], a.haircuts["bhy"]
        comp.add_row("Harvey-Liu haircut, Bonferroni (= Holm)", f"{hb.haircut:.0%}")
        comp.add_row("Harvey-Liu haircut, BHY", f"{hy.haircut:.0%}")
        if a.arch is not None:
            comp.add_row("Hansen SPA p-value (no strategy beats zero)", f"{a.arch.spa_pvalue:.3f}")
            comp.add_row(
                f"Romano-Wolf StepM: winner superior at {a.arch.size:g}",
                "yes" if a.arch.winner_is_superior else "no",
            )
        console.print(comp)
    console.print()
    console.print("[bold]Diagnostics[/]")
    diag = Table(box=box.SIMPLE_HEAD, pad_edge=False, show_edge=False, show_header=False)
    diag.add_column()
    diag.add_column(justify="right")
    diag.add_row("winner margin (closest rival)", f"{d.margin:.2f} SE")
    diag.add_row("gap to second best", f"{d.gap_in_se:.2f} SE")
    diag.add_row("effective number of trials", f"{d.effective_trials:.1f} of {d.n_trials}")
    lo, hi = w.truncation
    diag.add_row(
        "truncation interval of the winner",
        f"[{_fmt(lo)}, {'inf' if not math.isfinite(hi) else _fmt(hi)}]",
    )
    diag.add_row("conditional / naive width", f"{d.conditional_to_naive_width:.1f}x")
    console.print(diag)
    if coverage is not None:
        console.print()
        console.print(
            f"[bold]Coverage under this covariance[/] "
            f"[dim]({coverage.n_rep:,} simulated panels, all "
            f"true values equal, nominal {level})[/]"
        )
        cov_table = Table(box=box.SIMPLE_HEAD, pad_edge=False, show_edge=False, show_header=False)
        cov_table.add_column()
        cov_table.add_column(justify="right")
        for name, v in (
            ("naive", coverage.naive_coverage),
            ("conditional", coverage.conditional_coverage),
            ("hybrid", coverage.hybrid_coverage),
            ("projection", coverage.projection_coverage),
        ):
            colour = "red" if v < 1 - w.alpha - 0.03 else "green"
            cov_table.add_row(name, f"[{colour}]{v:.1%}[/]")
        console.print(cov_table)
    if a.projection_sensitivity:
        console.print()
        console.print(
            "[bold]Unlogged trials[/] "
            "[dim](projection interval, exact for the assumed covariance)[/]"
        )
        sens = Table(box=box.SIMPLE_HEAD, pad_edge=False, show_edge=False)
        sens.add_column("hidden trials", justify="right")
        sens.add_column("critical value", justify="right")
        sens.add_column("interval", justify="right")
        sens.add_column("width vs logged", justify="right")
        for r in a.projection_sensitivity:
            sens.add_row(
                str(r.n_hidden),
                _fmt(r.critical_value),
                f"[{_fmt(r.lower)}, {_fmt(r.upper)}]",
                f"{r.width_ratio:.2f}x",
            )
        console.print(sens)
        if a.scenario is not None:
            sc = a.scenario
            console.print(
                f"[dim]scenario, not an interval: with all true values {sc.true_value:g} and "
                f"hidden-trial "
                f"similarity {sc.correlation:.2f}, the hybrid estimate ranges over "
                f"[{_fmt(sc.hybrid_estimate[0])}, {_fmt(sc.hybrid_estimate[2])}] (5-95%); "
                f"winner stays best in {sc.acceptance:.0%} of draws[/]"
            )
    if oos is not None:
        _print_oos(a, oos, console, statistic_label)
    for message in a.warnings:
        console.print(f"[yellow]warning:[/] {message}")


def _print_oos(a: Analysis, oos: OutOfSample, console: Console, label: str) -> None:
    level = f"{100 * (1 - a.winner.alpha):g}%"
    console.print()
    console.print(
        f"[bold]Out of sample[/] [dim]({oos.n_obs:,} obs; {level} forecast intervals for the "
        f"winner's new {label}, true value unchanged)[/]"
    )
    table = Table(box=box.SIMPLE_HEAD, pad_edge=False, show_edge=False)
    table.add_column("Forecast interval")
    table.add_column("interval", justify="right")
    table.add_column("realised inside", justify="right")
    for name, iv, inside in (
        ("Naive", oos.forecast.naive, oos.in_naive),
        ("Conditional", oos.forecast.conditional, oos.in_conditional),
        ("Hybrid", oos.forecast.hybrid, oos.in_hybrid),
    ):
        mark = "[green]yes[/]" if inside else "[red]no[/]"
        table.add_row(name, _interval(iv), mark)
    console.print(table)
    console.print(
        f"realised {label} [bold]{_fmt(oos.realised)}[/] (standard error {_fmt(oos.se)}); "
        f"conditional p-value for an unchanged true value: {oos.conditional_pvalue:.3f}"
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="wincurse",
        description="Selection-corrected Sharpe ratios for the best of many backtests.",
    )
    parser.add_argument("--version", action="version", version=f"wincurse {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)
    an = sub.add_parser(
        "analyze",
        help="analyze a T x K file of per-period returns (CSV or Parquet)",
        description="Estimate every trial's performance, pick the winner and report "
        "selection-corrected estimates and intervals.",
    )
    an.add_argument("file", help="CSV or Parquet file; one column per trial, one row per period")
    an.add_argument("--alpha", type=float, default=0.05, help="1 - confidence level (default 0.05)")
    an.add_argument(
        "--beta", type=float, default=None, help="hybrid projection level (default alpha/10)"
    )
    an.add_argument(
        "--method",
        choices=["hybrid", "conditional", "projection", "naive"],
        default="hybrid",
        help="procedure shown as the headline (default hybrid)",
    )
    an.add_argument("--report", metavar="HTML", help="write a self-contained HTML tear sheet")
    an.add_argument(
        "--periods-per-year",
        type=float,
        default=None,
        help="observations per year; inferred from a date index if omitted",
    )
    an.add_argument(
        "--select", choices=["estimate", "t"], default="estimate", help="selection rule"
    )
    an.add_argument("--statistic", choices=["sharpe", "mean"], default="sharpe")
    an.add_argument("--cov", choices=["hac", "iid", "normal", "bootstrap"], default="hac")
    an.add_argument(
        "--shrink", action="store_true", help="Ledoit-Wolf constant-correlation shrinkage"
    )
    an.add_argument("--dropna", action="store_true", help="drop rows with missing values")
    an.add_argument("--n-sim", type=int, default=100_000, help="draws for critical values")
    an.add_argument("--seed", type=int, default=0)
    an.add_argument(
        "--coverage-reps", type=int, default=1000, help="coverage simulation size (0 = off)"
    )
    an.add_argument(
        "--oos",
        metavar="FILE",
        help="later returns (CSV or Parquet) containing the winner's column: forecast check",
    )
    an.add_argument("--arch", action="store_true", help="add Hansen SPA and Romano-Wolf StepM")
    an.add_argument("--hidden-trials", type=int, default=0, help="unlogged-trials sensitivity")
    an.add_argument(
        "--hidden-corr",
        type=float,
        default=None,
        help="correlation of a hidden trial with the logged trial it resembles (default: data)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point of the ``wincurse`` command; returns the process exit code."""
    args = _build_parser().parse_args(argv)
    console = Console(highlight=False)
    err = Console(stderr=True, highlight=False)
    try:
        frame = read_returns(args.file, dropna=args.dropna)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", WincurseWarning)
            result = analyze(
                frame,
                args.periods_per_year,
                args.alpha,
                args.beta,
                method=args.method,
                select=args.select,
                statistic=args.statistic,
                cov_method=args.cov,
                shrinkage=args.shrink,
                n_sim=args.n_sim,
                seed=args.seed,
                hidden_trials=args.hidden_trials,
                hidden_correlation=args.hidden_corr,
                arch_tests=args.arch,
            )
        oos = None
        if args.oos:
            oos = check_out_of_sample(result, read_returns(args.oos, dropna=args.dropna))
        coverage = None
        if args.coverage_reps > 0:
            coverage = simulate_known_sigma(
                np.zeros(result.performance.n_strategies),
                result.performance.cov,
                args.coverage_reps,
                args.alpha,
                args.beta,
                select=args.select,
                critical_values=(
                    result.winner.critical_value_alpha,
                    result.winner.critical_value_beta,
                ),
                rng=args.seed,
            )
    except (ValueError, OSError) as exc:
        err.print(f"[red]error:[/] {exc}")
        return 2
    label = "Sharpe" if args.statistic == "sharpe" else "mean"
    print_analysis(result, console, Path(args.file).name, coverage, label, oos)
    if args.report:
        path = write_report(args.report, result, coverage, source=Path(args.file).name, oos=oos)
        console.print()
        console.print(f"[dim]report written to {path}[/]")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
