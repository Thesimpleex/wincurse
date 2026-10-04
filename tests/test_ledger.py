from __future__ import annotations

import doctest

import numpy as np
import pandas as pd
import pytest

from wincurse import Ledger, default_ledger, ledger, track


def test_module_doctest_runs():
    results = doctest.testmod(ledger)
    assert results.attempted > 0
    assert results.failed == 0


def test_track_records_every_distinct_variant_and_counts_calls():
    book = Ledger()

    @book.track
    def variant(lookback, skip=1):
        return np.linspace(0, 1, 30) * lookback + skip

    first = variant(5)
    variant(10)
    variant(5)
    assert book.n_trials == 2
    assert book.n_calls == 3
    assert len(book) == 2
    assert "variant(lookback=5, skip=1)" in book
    np.testing.assert_array_equal(first, np.linspace(0, 1, 30) * 5 + 1)
    assert list(book.to_frame().columns) == book.names


def test_record_validates_and_overwrites():
    book = Ledger()
    book.record("a", [0.1, 0.2, 0.3], {"x": 1})
    book.record("a", [0.4, 0.5, 0.6])
    assert book.n_trials == 1
    assert book.n_calls == 2
    assert book.to_frame()["a"].tolist() == [0.4, 0.5, 0.6]
    with pytest.raises(ValueError, match="NaN"):
        book.record("b", [0.1, np.nan])
    with pytest.raises(ValueError, match="empty"):
        Ledger().to_frame()


def test_to_frame_requires_aligned_trials():
    book = Ledger()
    book.record("a", np.zeros(10))
    book.record("b", np.zeros(8))
    with pytest.raises(ValueError, match="different lengths"):
        book.to_frame()
    aligned = Ledger()
    index_a = pd.date_range("2020-01-01", periods=5)
    index_b = pd.date_range("2021-01-01", periods=5)
    aligned.record("a", pd.Series(np.arange(5.0), index=index_a))
    aligned.record("b", pd.Series(np.arange(5.0), index=index_b))
    with pytest.raises(ValueError, match="align"):
        aligned.to_frame()


def test_ledger_to_csv_round_trip_and_analyze(tmp_path, rng):
    book = Ledger()
    dates = pd.date_range("2015-01-01", periods=500, freq="B")
    for i in range(6):
        book.record(f"v{i}", pd.Series(rng.standard_normal(500) * 0.01 + 0.0002 * i, index=dates))
    path = book.to_csv(tmp_path / "trials.csv")
    back = pd.read_csv(path, index_col=0)
    np.testing.assert_allclose(back.to_numpy(), book.to_frame().to_numpy())
    result = book.analyze(252.0, n_sim=5000)
    assert result.performance.n_strategies == 6


def test_module_level_track_uses_the_default_ledger():
    before = default_ledger().n_calls

    @track(name="probe")
    def probe():
        return np.ones(5) * np.arange(5)

    probe()
    assert default_ledger().n_calls == before + 1
    assert "probe" in default_ledger()

    private = Ledger()

    @track(ledger=private)
    def other(a):
        return np.arange(5.0) * a

    other(2)
    assert private.n_trials == 1


def test_unprintable_arguments_do_not_merge_distinct_variants():
    book = Ledger()
    weights = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]

    @book.track
    def blend(w, scale=1.0):
        return scale * np.linspace(0.0, 1.0, 20) * (w[0] + 2.0 * w[1])

    blend(weights[0])
    blend(weights[1])
    blend(weights[1])  # an identical rerun is the same trial
    assert book.n_trials == 2
    assert book.n_calls == 3
    assert all("#" in name for name in book.names)
