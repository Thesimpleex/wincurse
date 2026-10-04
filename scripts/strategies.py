"""Trial generators for the real-data case studies: industry momentum and technical rules.

Both families produce a ``T x K`` frame of per-period excess returns, one column per
variant, that is the input of ``wincurse``. Every variant uses information up to the
period before the return it earns, so there is no look-ahead.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

# ------------------------------------------------------------------- industry momentum


def momentum_weights(
    returns: pd.DataFrame, lookback: int, skip: int, n_leg: int, holding: int
) -> np.ndarray:
    """Weights ``T x N`` of an industry long-short momentum strategy.

    The signal of industry ``i`` for month ``t`` is its cumulative log return over the
    ``lookback`` months ending ``skip`` months before ``t - 1`` (``skip = 0``: the signal
    ends with the previous month). Industries with a missing return inside the window are
    not ranked. If at least ``2 * n_leg`` industries are ranked, the top ``n_leg`` are
    bought and the bottom ``n_leg`` sold with equal weights ``1 / n_leg``; otherwise the
    month is unavailable (NaN). With ``holding > 1`` months the weights are averaged over
    the portfolios formed in the last ``holding`` months (Jegadeesh and Titman 1993),
    which makes the returns overlap and serially correlated.
    """
    log_r = np.log1p(returns)
    signal = log_r.rolling(lookback, min_periods=lookback).sum().shift(1 + skip).to_numpy()
    t, n = signal.shape
    weights = np.full((t, n), np.nan)
    ranked = ~np.isnan(signal)
    for row in range(t):
        ok = ranked[row]
        if ok.sum() < 2 * n_leg:
            continue
        idx = np.flatnonzero(ok)
        order = idx[np.argsort(signal[row, idx], kind="stable")]
        w = np.zeros(n)
        w[order[:n_leg]] = -1.0 / n_leg
        w[order[-n_leg:]] = 1.0 / n_leg
        weights[row] = w
    if holding > 1:
        acc = np.zeros((t, n))
        for h in range(holding):
            shifted = np.full((t, n), np.nan)
            shifted[h:] = weights[: t - h]
            acc = acc + shifted / holding
        weights = acc
    return weights


def momentum_variants(
    returns: pd.DataFrame,
    lookbacks=(1, 2, 3, 6, 9, 12, 18, 24),
    skips=(0, 1),
    holdings=(1, 3, 6, 12),
    sizes=(3, 5, 7, 10),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """All momentum variants on a grid; returns ``(returns_frame, parameter_table)``.

    A variant is unavailable (NaN) in months where its weights are. Missing industry
    returns of a held industry contribute zero and are counted in ``missing_held``.
    """
    values = returns.to_numpy(dtype=float)
    filled = np.nan_to_num(values, nan=0.0)
    columns: dict[str, np.ndarray] = {}
    rows = []
    for lookback, skip, holding, size in itertools.product(lookbacks, skips, holdings, sizes):
        w = momentum_weights(returns, lookback, skip, size, holding)
        series = np.einsum("ti,ti->t", np.nan_to_num(w, nan=0.0), filled)
        series[np.isnan(w).any(axis=1)] = np.nan
        held_missing = int(np.sum((np.abs(np.nan_to_num(w)) > 0) & np.isnan(values)))
        name = f"mom J={lookback} skip={skip} H={holding} n={size}"
        columns[name] = series
        rows.append(
            {
                "name": name,
                "lookback": lookback,
                "skip": skip,
                "holding": holding,
                "n_leg": size,
                "missing_held": held_missing,
            }
        )
    frame = pd.DataFrame(columns, index=returns.index)
    return frame, pd.DataFrame(rows).set_index("name")


# ------------------------------------------------------------------- technical rules


def _moving_average(price: np.ndarray, window: int) -> np.ndarray:
    csum = np.cumsum(np.insert(price, 0, 0.0))
    out = np.full(price.size, np.nan)
    out[window - 1 :] = (csum[window:] - csum[:-window]) / window
    return out


def _persist(signal: np.ndarray) -> np.ndarray:
    """Hold the last non-zero signal (+1 buy, -1 sell) until the opposite one appears."""
    out = pd.Series(np.where(signal == 0, np.nan, signal)).ffill().fillna(0.0)
    return out.to_numpy()


def ma_rule(price: np.ndarray, fast: int, slow: int, band: float) -> np.ndarray:
    """Buy if the fast average exceeds the slow one by ``band`` (relative), sell if below."""
    f, s = _moving_average(price, fast), _moving_average(price, slow)
    with np.errstate(invalid="ignore"):
        buy = f > s * (1 + band)
        sell = f < s * (1 - band)
    return np.where(buy, 1.0, np.where(sell, -1.0, 0.0))


def filter_rule(price: np.ndarray, x: float) -> np.ndarray:
    """Alexander (1961) filter: long after a rise of ``x`` from the last trough, short after
    a fall of ``x`` from the last peak."""
    pos = np.zeros(price.size)
    state = 0.0
    peak = trough = price[0]
    for i in range(1, price.size):
        p = price[i]
        if state == 1.0:
            peak = max(peak, p)
            if p <= peak * (1 - x):
                state, trough = -1.0, p
        elif state == -1.0:
            trough = min(trough, p)
            if p >= trough * (1 + x):
                state, peak = 1.0, p
        else:
            peak, trough = max(peak, p), min(trough, p)
            if p >= trough * (1 + x):
                state, peak = 1.0, p
            elif p <= peak * (1 - x):
                state, trough = -1.0, p
        pos[i] = state
    return pos


def channel_rule(price: np.ndarray, window: int, band: float) -> np.ndarray:
    """Donchian channel: buy above the previous ``window`` highest close by ``band``,
    sell below the lowest, hold in between."""
    hi = pd.Series(price).rolling(window).max().shift(1).to_numpy()
    lo = pd.Series(price).rolling(window).min().shift(1).to_numpy()
    with np.errstate(invalid="ignore"):
        signal = np.where(
            price > hi * (1 + band), 1.0, np.where(price < lo * (1 - band), -1.0, 0.0)
        )
    return _persist(signal)


def momentum_rule(price: np.ndarray, window: int, band: float) -> np.ndarray:
    """Time-series momentum: long if the ``window``-day return exceeds ``band``, else short."""
    ret = np.full(price.size, np.nan)
    ret[window:] = price[window:] / price[:-window] - 1.0
    with np.errstate(invalid="ignore"):
        return np.where(ret > band, 1.0, np.where(ret < -band, -1.0, 0.0))


def technical_variants(
    market_excess: pd.Series, risk_free: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Several hundred technical trading rules on a market index (``(returns, parameters)``).

    The index is the compounded total market return. A rule observes the closing price of
    day ``t - 1`` and holds ``position in {-1, 0, +1}`` (short, flat, long) for day ``t``;
    its excess return is ``position * market_excess`` (flat positions earn the risk-free
    rate). Each rule also appears as a long-or-flat version (short signals become flat).
    No transaction costs are charged.
    """
    total = (market_excess + risk_free).to_numpy(dtype=float)
    price = np.cumprod(1.0 + total)
    excess = market_excess.to_numpy(dtype=float)
    signals: dict[str, np.ndarray] = {}
    params: dict[str, dict] = {}

    def add(family: str, label: str, signal: np.ndarray, **p) -> None:
        for mode, pos in (("ls", signal), ("lf", np.maximum(signal, 0.0))):
            name = f"{family} {label} {mode}"
            signals[name] = pos
            params[name] = {"family": family, "mode": mode, **p}

    bands = (0.0, 0.005, 0.01)
    for slow in (10, 20, 50, 100, 150, 200):
        for fast in (1, 2, 5, 10, 20, 50):
            if fast >= slow:
                continue
            for band in bands:
                add(
                    "ma",
                    f"{fast}/{slow} b={band:g}",
                    ma_rule(price, fast, slow, band),
                    fast=fast,
                    slow=slow,
                    band=band,
                )
    for x in (0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20):
        add("filter", f"x={x:g}", filter_rule(price, x), x=x)
    for window in (5, 10, 20, 40, 60, 100, 150, 200):
        for band in (0.0, 0.005, 0.01, 0.02):
            add(
                "channel",
                f"n={window} c={band:g}",
                channel_rule(price, window, band),
                window=window,
                band=band,
            )
    for window in (5, 10, 20, 40, 60, 100, 150, 200, 250):
        for band in (0.0, 0.005, 0.01, 0.02):
            add(
                "tsmom",
                f"n={window} c={band:g}",
                momentum_rule(price, window, band),
                window=window,
                band=band,
            )

    lagged = {name: np.concatenate([[0.0], pos[:-1]]) for name, pos in signals.items()}
    frame = pd.DataFrame(
        {name: pos * excess for name, pos in lagged.items()}, index=market_excess.index
    )
    return frame, pd.DataFrame(params).T
