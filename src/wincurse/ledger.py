"""Trial ledger: record every variant that was tried so that ``K`` stays honest.

The corrections in this package depend on how many strategies were searched and
how they co-move. A :class:`Ledger` stores the return series of every backtest run
during a search, either explicitly with :meth:`Ledger.record` or automatically by
decorating the backtest function with :meth:`Ledger.track`.

>>> ledger = Ledger()
>>> @ledger.track
... def momentum(lookback):
...     return np.sin(np.arange(60) / lookback)
>>> _ = momentum(5), momentum(10), momentum(5)
>>> ledger.n_trials, ledger.n_calls
(2, 3)
"""

from __future__ import annotations

import functools
import hashlib
import inspect
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
import pandas as pd

__all__ = ["Ledger", "default_ledger", "track"]

F = TypeVar("F", bound=Callable[..., Any])


class Ledger:
    """Collection of the return series of every trial in a search.

    A trial is identified by its name; recording the same name again replaces the
    series (a re-run of an identical variant is not a new trial) but still increments
    :attr:`n_calls`, so a gap between calls and trials reveals repeated work.
    """

    def __init__(self) -> None:
        self._series: dict[str, pd.Series] = {}
        self._params: dict[str, dict[str, Any]] = {}
        self.n_calls = 0

    def __len__(self) -> int:
        return len(self._series)

    def __contains__(self, name: str) -> bool:
        return name in self._series

    @property
    def n_trials(self) -> int:
        """Number of distinct trials recorded."""
        return len(self._series)

    @property
    def names(self) -> list[str]:
        return list(self._series)

    def record(self, name: str, returns, params: dict[str, Any] | None = None) -> None:
        """Store the per-period returns of one trial under ``name``."""
        series = (
            returns if isinstance(returns, pd.Series) else pd.Series(np.asarray(returns, float))
        )
        series = series.astype(float)
        if series.isna().any():
            raise ValueError(f"trial {name!r} contains NaN returns")
        self.n_calls += 1
        self._series[str(name)] = series
        self._params[str(name)] = dict(params or {})

    def track(self, func: F | None = None, *, name: str | None = None) -> Any:
        """Decorator that records the return series produced by every call of ``func``.

        The decorated function must return the per-period returns of the variant
        (array or Series). The call is recorded under ``name`` (default: the function
        name plus its bound arguments; if some arguments cannot be printed, a short hash
        of the returns is appended so that different variants stay distinct) and the result
        is returned unchanged.
        """

        def decorate(fn: F) -> F:
            signature = inspect.signature(fn)

            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                result = fn(*args, **kwargs)
                bound = signature.bind(*args, **kwargs)
                bound.apply_defaults()
                shown = {
                    key: val
                    for key, val in bound.arguments.items()
                    if isinstance(val, int | float | str | bool | tuple | type(None))
                }
                label = (
                    name
                    or fn.__name__ + "(" + ", ".join(f"{k}={v}" for k, v in shown.items()) + ")"
                )
                if name is None and len(shown) < len(bound.arguments):
                    # Arguments that cannot be printed (arrays, objects) would make different
                    # variants share a label and silently shrink K; tell them apart by output.
                    digest = hashlib.sha1(np.asarray(result, dtype=float).tobytes()).hexdigest()
                    label += f"#{digest[:8]}"
                self.record(label, result, shown)
                return result

            return wrapper  # type: ignore[return-value]

        return decorate if func is None else decorate(func)

    def to_frame(self) -> pd.DataFrame:
        """All trials as a ``T x K`` frame, aligned on the row index.

        Raises
        ------
        ValueError
            If the series have different lengths and no common index to align on.
        """
        if not self._series:
            raise ValueError("the ledger is empty")
        lengths = {len(s) for s in self._series.values()}
        if len(lengths) > 1:
            raise ValueError(f"trials have different lengths {sorted(lengths)}; align them first")
        frame = pd.concat(self._series, axis=1)
        if frame.isna().any().any():
            raise ValueError("trial indices do not align; use a common index")
        return frame

    def to_csv(self, path: str | Path) -> Path:
        """Write the ``T x K`` frame to CSV, ready for ``wincurse analyze``."""
        path = Path(path)
        self.to_frame().to_csv(path)
        return path

    def analyze(self, periods_per_year: float, **kwargs):
        """Run :func:`wincurse.analyze` on the recorded trials."""
        from .analysis import analyze

        return analyze(self.to_frame(), periods_per_year, **kwargs)


_DEFAULT = Ledger()


def default_ledger() -> Ledger:
    """The process-wide ledger used by :func:`track` when no ledger is given."""
    return _DEFAULT


def track(func: F | None = None, *, ledger: Ledger | None = None, name: str | None = None) -> Any:
    """Decorator recording every call into ``ledger`` (default: :func:`default_ledger`)."""
    target = _DEFAULT if ledger is None else ledger
    return target.track(func, name=name)
