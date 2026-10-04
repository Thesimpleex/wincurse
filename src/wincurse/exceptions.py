"""Exceptions and warnings raised by wincurse."""

from __future__ import annotations


class WincurseWarning(UserWarning):
    """A result is usable but a modelling assumption deserves attention."""


class RootFindingError(RuntimeError):
    """A truncated-normal inversion could not be bracketed.

    Failures are never skipped silently: callers either see this exception or a
    non-finite endpoint that is counted as a miss in coverage computations.
    """
