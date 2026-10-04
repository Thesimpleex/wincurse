from __future__ import annotations

from html.parser import HTMLParser

import numpy as np
import pandas as pd
import pytest

from wincurse import analyze, check_out_of_sample
from wincurse.cli import main, read_returns
from wincurse.report import render_report, write_report
from wincurse.simulation import simulate_known_sigma

from .conftest import equicorrelated


@pytest.fixture
def panel(rng) -> pd.DataFrame:
    chol = np.linalg.cholesky(equicorrelated(10, 0.4))
    values = rng.standard_normal((900, 10)) @ chol.T * 0.01 + 0.0002
    index = pd.date_range("2018-01-01", periods=900, freq="B")
    return pd.DataFrame(values, index=index, columns=[f"var{i}" for i in range(10)])


@pytest.fixture
def csv_path(panel, tmp_path):
    path = tmp_path / "returns.csv"
    panel.to_csv(path)
    return path


class _Collector(HTMLParser):
    def __init__(self):
        super().__init__()
        self.stack: list[str] = []
        self.errors: list[str] = []
        self.void = {"meta", "br", "hr", "img", "input", "link", "line", "circle", "rect"}

    def handle_starttag(self, tag, attrs):
        if tag not in self.void:
            self.stack.append(tag)

    def handle_startendtag(self, tag, attrs):
        pass

    def handle_endtag(self, tag):
        if tag in self.void:
            return
        if not self.stack or self.stack[-1] != tag:
            self.errors.append(f"unbalanced </{tag}> with stack {self.stack[-3:]}")
        else:
            self.stack.pop()


def test_report_is_self_contained_well_formed_html(panel):
    result = analyze(panel, n_sim=10_000, hidden_trials=6)
    cov = simulate_known_sigma(
        np.zeros(10), result.performance.cov, 300, critical_values=(2.5, 3.1), rng=0
    )
    oos = check_out_of_sample(result, panel.iloc[:300])
    page = render_report(result, cov, source="returns.csv", oos=oos)
    parser = _Collector()
    parser.feed(page)
    assert parser.errors == []
    assert parser.stack == []
    for needle in (
        "Selection-adjusted tear sheet",
        result.winner_name,
        "Coverage under your covariance",
        "Unlogged trials",
        "Out-of-sample check",
        "Deflated Sharpe ratio",
        "Illustrative research, not investment advice",
        "<svg",
    ):
        assert needle in page
    assert "http://" not in page.replace("http://www.w3.org", "")
    assert "<script" not in page
    assert "<link" not in page


def test_report_escapes_user_supplied_names(panel):
    panel = panel.rename(columns={"var0": "<b>x&y</b>"})
    panel.iloc[:, 0] += 0.01
    page = render_report(analyze(panel, n_sim=3000, method="conditional"))
    assert "<b>x&y</b>" not in page
    assert "&lt;b&gt;x&amp;y&lt;/b&gt;" in page


def test_write_report_creates_the_file(panel, tmp_path):
    path = write_report(tmp_path / "out.html", analyze(panel, n_sim=3000))
    assert path.read_text(encoding="utf-8").startswith("<!doctype html>")


def test_read_returns_dates_and_missing_values(tmp_path, panel):
    path = tmp_path / "a.csv"
    panel.to_csv(path)
    frame = read_returns(path)
    assert isinstance(frame.index, pd.DatetimeIndex)
    assert frame.shape == panel.shape
    plain = tmp_path / "b.csv"
    panel.reset_index(drop=True).to_csv(plain, index=False)
    assert read_returns(plain).shape == panel.shape
    holey = panel.copy()
    holey.iloc[3, 2] = np.nan
    holey.to_csv(path)
    with pytest.raises(ValueError, match="missing"):
        read_returns(path)
    assert read_returns(path, dropna=True).shape[0] == panel.shape[0] - 1
    with pytest.raises(ValueError, match="not found"):
        read_returns(tmp_path / "nope.csv")


def test_read_returns_parquet(tmp_path, panel):
    pytest.importorskip("pyarrow")
    path = tmp_path / "r.parquet"
    panel.to_parquet(path)
    assert read_returns(path).shape == panel.shape


def test_cli_analyze_prints_the_summary_and_writes_the_report(csv_path, tmp_path, capsys):
    report = tmp_path / "tear.html"
    code = main(
        [
            "analyze",
            str(csv_path),
            "--n-sim",
            "10000",
            "--coverage-reps",
            "200",
            "--hidden-trials",
            "5",
            "--arch",
            "--report",
            str(report),
        ]
    )
    out = capsys.readouterr().out
    assert code == 0
    for needle in (
        "Winner",
        "Hybrid",
        "Conditional (median-unbiased)",
        "Comparators",
        "Diagnostics",
        "Coverage under this covariance",
        "Unlogged trials",
        "Hansen SPA",
        "Romano-Wolf",
        "report written",
    ):
        assert needle in out
    assert report.exists()
    assert "var" in report.read_text(encoding="utf-8")


def test_cli_out_of_sample_section(csv_path, panel, tmp_path, capsys):
    later = panel.iloc[-300:]
    later_path = tmp_path / "later.csv"
    later.to_csv(later_path)
    code = main(
        [
            "analyze",
            str(csv_path),
            "--n-sim",
            "5000",
            "--coverage-reps",
            "0",
            "--oos",
            str(later_path),
            "--method",
            "conditional",
            "--cov",
            "iid",
        ]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "Out of sample" in out
    assert "conditional p-value" in out
    assert "Coverage under" not in out


def test_cli_reports_errors_with_exit_code_two(tmp_path, capsys):
    code = main(["analyze", str(tmp_path / "missing.csv")])
    assert code == 2
    assert "file not found" in capsys.readouterr().err
    bad = tmp_path / "bad.csv"
    pd.DataFrame(np.ones((20, 3))).to_csv(bad, index=False)
    assert main(["analyze", str(bad), "--periods-per-year", "12", "--n-sim", "1000"]) == 2


def test_cli_requires_a_frequency_without_dates(tmp_path, panel, capsys):
    path = tmp_path / "plain.csv"
    panel.reset_index(drop=True).to_csv(path, index=False)
    assert main(["analyze", str(path), "--n-sim", "1000", "--coverage-reps", "0"]) == 2
    assert "periods_per_year" in capsys.readouterr().err
    ok = main(
        [
            "analyze",
            str(path),
            "--periods-per-year",
            "252",
            "--n-sim",
            "1000",
            "--coverage-reps",
            "0",
        ]
    )
    assert ok == 0


def test_cli_version(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert capsys.readouterr().out.startswith("wincurse")
