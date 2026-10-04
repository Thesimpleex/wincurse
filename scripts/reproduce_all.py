"""Run every experiment, redraw all figures and screenshots, and render the documents.

``python scripts/reproduce_all.py`` needs network access for the Ken French library and, on
first use, the 267 MB AKM replication package. Use ``--quick`` to smoke-test the coverage
study with tiny sizes (results file untouched), ``--jobs N`` for the number of workers of the
coverage study and ``--skip-screenshots`` if Google Chrome is not installed.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def run(script: str, *args: str) -> None:
    started = time.time()
    print(f"\n== {script} {' '.join(args)}", flush=True)
    subprocess.run([sys.executable, str(HERE / script), *args], check=True)
    print(f"-- {script} finished in {time.time() - started:.0f} s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--skip-screenshots", action="store_true")
    args = parser.parse_args()

    if not (HERE.parent / "tests" / "data" / "akm_jobstart.npz").exists():
        run("extract_akm_fixture.py")
    run("exp_akm_replication.py")
    if args.quick:
        run("exp_coverage.py", "--quick", "--no-write", "--jobs", str(args.jobs))
        return
    run("exp_coverage.py", "--jobs", str(args.jobs))
    run("exp_momentum.py")
    run("exp_technical.py")
    run("exp_selection_rules.py")
    run("make_example.py")
    run("make_figures.py")
    if not args.skip_screenshots:
        run("make_screenshots.py")
    run("render_docs.py")


if __name__ == "__main__":
    main()
