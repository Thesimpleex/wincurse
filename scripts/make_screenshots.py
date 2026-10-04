"""Regenerate the product screenshots in ``docs/screenshots`` from real program output.

Needs the case-study panels in ``data/cache`` (run ``exp_momentum.py`` and
``exp_technical.py`` first), the ``docs`` extra and a local Google Chrome.
"""

from __future__ import annotations

import sys

from common import CACHE, ROOT, SCREENSHOTS
from screenshots import report_screenshot, terminal_screenshot

from wincurse import analyze, check_out_of_sample
from wincurse.cli import read_returns
from wincurse.report import write_report
from wincurse.simulation import simulate_known_sigma

MOMENTUM = ("momentum_insample.csv", "momentum_oos.csv")
TECHNICAL = ("technical_insample.csv", "technical_oos.csv")


def terminal_shots() -> None:
    jobs = (
        (
            "hero",
            ["analyze", MOMENTUM[0], "--oos", MOMENTUM[1], "--coverage-reps", "1000"],
            "wincurse analyze momentum_insample.csv --oos momentum_oos.csv",
            "Best of 256 industry-momentum variants",
        ),
        (
            "decay",
            ["analyze", TECHNICAL[0], "--oos", TECHNICAL[1], "--coverage-reps", "0", "--arch"],
            "wincurse analyze technical_insample.csv --oos technical_oos.csv --arch",
            "Best of 340 technical trading rules",
        ),
    )
    for name, args, shown, title in jobs:
        for theme in ("dark", "light"):
            terminal_screenshot(
                [sys.executable, "-m", "wincurse.cli", *args],
                SCREENSHOTS / f"{name}-{theme}.png",
                title=title,
                display_command=shown,
                theme=theme,
                columns=112,
                cwd=CACHE,
            )
            print(f"wrote {name}-{theme}.png")


def tearsheet() -> None:
    frame = read_returns(CACHE / MOMENTUM[0])
    later = read_returns(CACHE / MOMENTUM[1])
    result = analyze(frame, n_sim=100_000, seed=0, hidden_trials=100, arch_tests=True)
    coverage = simulate_known_sigma(
        [0.0] * result.performance.n_strategies,
        result.performance.cov,
        1000,
        critical_values=(result.winner.critical_value_alpha, result.winner.critical_value_beta),
        rng=0,
    )
    oos = check_out_of_sample(result, later)
    path = write_report(
        ROOT / "docs" / "tearsheet_momentum.html",
        result,
        coverage,
        source="momentum_insample.csv",
        oos=oos,
    )
    out = SCREENSHOTS / "tearsheet.png"
    report_screenshot(path, out, width=1000, selector="#summary")
    print(f"wrote {out.name}")


def main() -> None:
    SCREENSHOTS.mkdir(parents=True, exist_ok=True)
    terminal_shots()
    tearsheet()


if __name__ == "__main__":
    main()
