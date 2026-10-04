"""The README and methodology are rendered from templates and the committed results file."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import pytest
from render_docs import PLACEHOLDER, TARGETS, TEMPLATES, fmt, lookup, render

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "docs" / "results.json"

needs_results = pytest.mark.skipif(not RESULTS.exists(), reason="docs/results.json not generated")


def test_render_formats_and_lookup():
    data = {"a": {"b": [0.12345, 2]}, "n": 12345.6, "s": "x"}
    assert lookup(data, "a/b/0") == 0.12345
    assert render("{{ a/b/0 | 2 }} {{a/b/0|pct1}} {{ n | int }} {{ s }}", data) == (
        "0.12 12.3% 12,346 x"
    )
    with pytest.raises(ValueError, match="cannot resolve"):
        render("{{ missing/key | 2 }}", data)
    with pytest.raises(ValueError, match="cannot resolve"):
        render("{{ a/b/9 | 2 }}", data)
    with pytest.raises(ValueError, match="unknown format"):
        fmt(1.0, "bogus")
    with pytest.raises(ValueError, match="null"):
        fmt(None, "2")


@needs_results
@pytest.mark.parametrize("template", sorted(TARGETS))
def test_committed_documents_match_their_templates_and_results(template):
    results = json.loads(RESULTS.read_text())
    expected = render((TEMPLATES / template).read_text(), results)
    assert TARGETS[template].read_text(encoding="utf-8") == expected
    assert PLACEHOLDER.search(expected) is None


def _python_blocks(text: str) -> list[str]:
    return re.findall(r"```python\n(.*?)```", text, flags=re.S)


@needs_results
def test_readme_python_blocks_run(monkeypatch):
    monkeypatch.chdir(ROOT)
    blocks = _python_blocks((ROOT / "README.md").read_text(encoding="utf-8"))
    assert len(blocks) == 3
    namespace: dict = {}
    exec(blocks[0], namespace)
    returns = namespace["returns"]
    namespace["later_returns"] = returns.iloc[-250:]
    exec(blocks[1], namespace)
    exec(blocks[2], namespace)
    assert namespace["ledger"].n_trials == 4


@needs_results
def test_quickstart_claims_hold():
    q = json.loads(RESULTS.read_text())["quickstart"]
    truth = q["true_sharpe_of_every_variant"]
    assert not q["naive_interval"][0] <= truth <= q["naive_interval"][1]
    assert q["hybrid_interval"][0] <= truth <= q["hybrid_interval"][1]
    frame = pd.read_csv(ROOT / q["file"], index_col="date", parse_dates=True)
    assert frame.shape == (q["n_obs"], q["n_trials"])


@needs_results
def test_committed_coverage_study_backs_the_claims():
    cov = json.loads(RESULTS.read_text())["coverage"]
    reps = cov["design"]["reps_known_sigma"]
    tol = 4 * (0.95 * 0.05 / reps) ** 0.5
    for cell in cov["known_sigma"].values():
        assert cell["n_failed"] == 0
        assert cell["n_rep"] == reps
        assert abs(cell["conditional_coverage"] - 0.95) < tol
        assert cell["hybrid_coverage"] > 0.95 - tol
        assert cell["naive_coverage"] < cell["conditional_coverage"] + tol
    feas_reps = cov["design"]["reps_feasible"]
    tol = 4 * (0.95 * 0.05 / feas_reps) ** 0.5
    for name, cell in cov["feasible"].items():
        est = cell["estimated_sigma"]
        assert est["n_failed"] == 0
        assert est["n_rep"] == feas_reps
        if "ar1" in name and "hac" not in name:
            continue  # documented limit: serial correlation ignored by the covariance estimator
        assert est["conditional_coverage"] > 0.95 - 2 * tol
        assert est["hybrid_coverage"] > 0.95 - 2 * tol


@needs_results
def test_results_record_provenance():
    results = json.loads(RESULTS.read_text())
    for section in ("coverage", "momentum", "technical", "akm_replication"):
        env = results[section]["environment"]
        assert {"python", "numpy", "scipy", "pandas", "wincurse", "run_date"} <= set(env)
    assert results["momentum"]["data"]["crsp_vintage"]
    assert results["technical"]["data"]["last_observation"]
