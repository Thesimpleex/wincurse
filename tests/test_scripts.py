"""Tests of the experiment helpers in ``scripts/`` (data parsing and trial generators)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from casestudy import drop_duplicates
from frenchdata import parse_block
from strategies import (
    channel_rule,
    filter_rule,
    ma_rule,
    momentum_rule,
    momentum_variants,
    momentum_weights,
    technical_variants,
)

MONTHLY = """This file was created using the 202608 CRSP database.

  Average Value Weighted Returns -- Monthly
,A,B,C
192607,   1.00,  -99.99,   3.00
192608,  -2.00,   5.00, -999
192609,   0.50,   0.25,   0.10

  Average Equal Weighted Returns -- Monthly
,A,B,C
192607,   9.00,   9.00,   9.00
"""

DAILY = """header

,Mkt-RF,SMB,HML,RF
19260701,    0.09,   -0.23,   -0.28,    0.01
19260702,    0.45,   -0.34,   -0.03,    0.01

Annual
,Mkt-RF
1927,1.0
"""


def test_french_block_parser_reads_only_the_requested_block():
    frame = parse_block(MONTHLY, "Value Weighted Returns -- Monthly", 6)
    assert list(frame.columns) == ["A", "B", "C"]
    assert frame.shape == (3, 3)
    assert frame.iloc[0, 0] == pytest.approx(0.01)
    assert np.isnan(frame.iloc[0, 1])
    assert np.isnan(frame.iloc[1, 2])
    assert frame.iloc[2, 1] == pytest.approx(0.0025)
    assert frame.index[0] == pd.Timestamp("1926-07-31")
    assert frame.index[-1] == pd.Timestamp("1926-09-30")


def test_french_block_parser_daily_and_errors():
    frame = parse_block(DAILY, None, 8)
    assert frame.shape == (2, 4)
    assert frame.index[1] == pd.Timestamp("1926-07-02")
    assert frame.loc["1926-07-01", "Mkt-RF"] == pytest.approx(0.0009)
    with pytest.raises(ValueError, match="not found"):
        parse_block(MONTHLY, "Nonexistent block", 6)
    with pytest.raises(ValueError, match="no data rows"):
        parse_block("nothing here", None, 6)


def _monthly_returns(rng, t=40, n=8):
    index = pd.date_range("2000-01-31", periods=t, freq="ME")
    return pd.DataFrame(rng.standard_normal((t, n)) * 0.05, index=index)


def test_momentum_weights_form_a_dollar_neutral_ranking(rng):
    r = _monthly_returns(rng)
    w = momentum_weights(r, lookback=3, skip=0, n_leg=2, holding=1)
    valid = ~np.isnan(w).any(axis=1)
    assert valid.sum() == len(r) - 3
    assert np.allclose(w[valid].sum(axis=1), 0.0)
    assert np.allclose(np.abs(w[valid]).sum(axis=1), 2.0)
    # Row t uses returns of months t-3 .. t-1 only: the winner is the best of them.
    t = 10
    signal = np.log1p(r.iloc[t - 3 : t]).sum().to_numpy()
    longs = np.flatnonzero(w[t] > 0)
    assert set(longs) == set(np.argsort(signal)[-2:])
    assert set(np.flatnonzero(w[t] < 0)) == set(np.argsort(signal)[:2])


def test_momentum_has_no_lookahead_and_honours_skip(rng):
    r = _monthly_returns(rng)
    base = momentum_weights(r, 3, 1, 2, 1)
    changed = r.copy()
    changed.iloc[20] += 0.5
    shifted = momentum_weights(changed, 3, 1, 2, 1)
    np.testing.assert_array_equal(base[:22], shifted[:22])
    assert not np.array_equal(base[22], shifted[22])
    no_skip = momentum_weights(changed, 3, 0, 2, 1)
    assert not np.array_equal(momentum_weights(r, 3, 0, 2, 1)[21], no_skip[21])


def test_momentum_holding_period_averages_overlapping_portfolios(rng):
    r = _monthly_returns(rng)
    one = momentum_weights(r, 3, 0, 2, 1)
    three = momentum_weights(r, 3, 0, 2, 3)
    t = 15
    np.testing.assert_allclose(three[t], (one[t] + one[t - 1] + one[t - 2]) / 3)
    assert np.isnan(three[3 + 1]).all()
    assert not np.isnan(three[3 + 2]).any()


def test_momentum_skips_months_with_too_few_ranked_industries(rng):
    r = _monthly_returns(rng, n=6)
    r.iloc[:12, 3:] = np.nan
    w = momentum_weights(r, 2, 0, 3, 1)
    assert np.isnan(w[:14]).all()
    assert not np.isnan(w[15]).any()


def test_momentum_variants_grid(rng):
    r = _monthly_returns(rng, t=80, n=12)
    frame, params = momentum_variants(r, (1, 3), (0, 1), (1, 3), (2, 3))
    assert frame.shape[1] == 16
    assert list(frame.columns) == list(params.index)
    row = params.loc["mom J=3 skip=1 H=3 n=2"]
    assert (row.lookback, row.skip, row.holding, row.n_leg) == (3, 1, 3, 2)
    w = momentum_weights(r, 3, 1, 2, 3)
    expected = np.nansum(w * r.to_numpy(), axis=1)
    got = frame["mom J=3 skip=1 H=3 n=2"].to_numpy()
    ok = ~np.isnan(got)
    np.testing.assert_allclose(got[ok], expected[ok])
    assert int(params["missing_held"].sum()) == 0


def test_moving_average_rule_against_brute_force(rng):
    price = np.cumprod(1 + rng.standard_normal(300) * 0.01)
    pos = ma_rule(price, 5, 20, 0.002)
    for t in (19, 50, 120, 299):
        fast, slow = price[t - 4 : t + 1].mean(), price[t - 19 : t + 1].mean()
        expected = 1.0 if fast > slow * 1.002 else -1.0 if fast < slow * 0.998 else 0.0
        assert pos[t] == expected
    assert (pos[:19] == 0).all()


def test_filter_rule_on_a_hand_checked_path():
    price = np.array([100, 101, 103, 106, 104, 100, 99, 98, 101, 104.0])
    pos = filter_rule(price, 0.03)
    # 100 -> 103 is a 3% rise from the trough: long; falls below 106 * 0.97 = 102.82 at 100:
    # short; rises 3% above the trough 98 only at 101 * ... (98 * 1.03 = 100.94): long again.
    assert pos.tolist() == [0, 0, 1, 1, 1, -1, -1, -1, 1, 1]


def test_channel_and_time_series_momentum_rules():
    price = np.array([10, 11, 12, 11, 10, 9, 8, 9, 10, 11.0])
    # A close above the highest (below the lowest) of the previous three closes buys
    # (sells); in between the last signal is held.
    assert channel_rule(price, 3, 0.0).tolist() == [0, 0, 0, 0, -1, -1, -1, -1, 1, 1]
    assert momentum_rule(price, 2, 0.0).tolist() == [0, 0, 1, 0, -1, -1, -1, 0, 1, 1]


def test_technical_variants_use_yesterdays_position(rng):
    dates = pd.bdate_range("2000-01-03", periods=600)
    excess = pd.Series(rng.standard_normal(600) * 0.01 + 0.0002, index=dates)
    rf = pd.Series(0.0001, index=dates)
    frame, params = technical_variants(excess, rf)
    assert frame.shape == (600, 340)
    assert params.shape[0] == 340
    assert set(params["family"]) == {"ma", "filter", "channel", "tsmom"}
    assert (frame.iloc[0] == 0).all()
    # Every daily return is +/- the market excess return or zero.
    ratio = frame.to_numpy()[1:] / excess.to_numpy()[1:, None]
    assert set(np.unique(np.round(ratio, 10))) <= {-1.0, 0.0, 1.0}
    # No look-ahead: changing the last day's return cannot change earlier strategy returns.
    other = excess.copy()
    other.iloc[-1] += 0.2
    frame2, _ = technical_variants(other, rf)
    np.testing.assert_allclose(frame.iloc[:-1].to_numpy(), frame2.iloc[:-1].to_numpy())
    long_flat = frame[[c for c in frame.columns if c.endswith(" lf")]]
    long_short = frame[[c for c in frame.columns if c.endswith(" ls")]]
    assert long_flat.shape == long_short.shape


def test_drop_duplicates_removes_perfectly_correlated_variants(rng):
    base = rng.standard_normal((200, 3))
    panel = pd.DataFrame(
        {"a": base[:, 0], "b": base[:, 1], "a2": 2.0 * base[:, 0] + 1.0, "c": base[:, 2]}
    )
    kept, dropped = drop_duplicates(panel)
    assert list(kept.columns) == ["a", "b", "c"]
    assert dropped == ["a2"]
