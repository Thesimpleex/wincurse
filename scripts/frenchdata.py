"""Download, cache and parse Ken French Data Library files.

Raw files are downloaded into ``data/raw/`` (gitignored) and never committed. Only the
first monthly or daily block of a file is read; missing values (``-99.99``, ``-999``)
become NaN. Returns are converted from percent to decimals.
"""

from __future__ import annotations

import re
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
FILES = {
    "industries49": "49_Industry_Portfolios_CSV.zip",
    "factors_daily": "F-F_Research_Data_Factors_daily_CSV.zip",
    "factors_monthly": "F-F_Research_Data_Factors_CSV.zip",
}
MISSING = (-99.99, -999.0)


@dataclass(frozen=True)
class FrenchFile:
    """A parsed block together with the provenance recorded in the results file."""

    frame: pd.DataFrame
    source: str
    created: str
    last_observation: str


def download(key: str, *, refresh: bool = False) -> Path:
    """Fetch the zip archive for ``key`` into ``data/raw`` (cached) and return its path."""
    name = FILES[key]
    path = RAW / name
    if refresh or not path.exists():
        RAW.mkdir(parents=True, exist_ok=True)
        request = urllib.request.Request(BASE + name, headers={"User-Agent": "wincurse-research"})
        with urllib.request.urlopen(request, timeout=120) as response:
            path.write_bytes(response.read())
    return path


def _read_text(path: Path) -> str:
    with zipfile.ZipFile(path) as zf:
        member = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
        return zf.read(member).decode("latin-1")


def parse_block(text: str, title: str | None = None, digits: int = 6) -> pd.DataFrame:
    """Parse the first block of rows whose first field has ``digits`` digits.

    If ``title`` is given the block must follow a line containing it (for example
    ``"Average Value Weighted Returns -- Monthly"``). The header is the line directly
    above the first data row.
    """
    lines = text.splitlines()
    start = 0
    if title is not None:
        hits = [i for i, line in enumerate(lines) if title in line]
        if not hits:
            raise ValueError(f"block {title!r} not found")
        start = hits[0]
    pattern = re.compile(rf"^\s*\d{{{digits}}}\s*,")
    first = next((i for i in range(start, len(lines)) if pattern.match(lines[i])), None)
    if first is None:
        raise ValueError("no data rows found")
    header = [c.strip() for c in lines[first - 1].split(",")][1:]
    rows = []
    for line in lines[first:]:
        if not pattern.match(line):
            break
        parts = [p.strip() for p in line.split(",")]
        rows.append([parts[0]] + [float(p) if p else np.nan for p in parts[1 : len(header) + 1]])
    frame = pd.DataFrame(rows, columns=["date", *header]).set_index("date")
    values = frame.to_numpy(dtype=float, copy=True)
    values[np.isin(values, MISSING)] = np.nan
    frame = pd.DataFrame(values / 100.0, index=frame.index, columns=header)
    fmt = "%Y%m" if digits == 6 else "%Y%m%d"
    frame.index = pd.to_datetime(frame.index, format=fmt)
    if digits == 6:
        frame.index = frame.index.to_period("M").to_timestamp("M")
    return frame


def _provenance(text: str) -> str:
    match = re.search(r"created (?:using|by using) the (\d{6}) CRSP database", text)
    return match.group(1) if match else "unknown"


def load(key: str, title: str | None = None, digits: int = 6, *, refresh: bool = False):
    """Download (if needed) and parse one block; returns :class:`FrenchFile`."""
    path = download(key, refresh=refresh)
    text = _read_text(path)
    frame = parse_block(text, title, digits)
    return FrenchFile(
        frame=frame,
        source=BASE + FILES[key],
        created=_provenance(text),
        last_observation=str(frame.index[-1].date()),
    )


def industries_monthly(*, refresh: bool = False) -> FrenchFile:
    """49 industry portfolios, value-weighted monthly returns."""
    return load("industries49", "Average Value Weighted Returns -- Monthly", 6, refresh=refresh)


def factors_daily(*, refresh: bool = False) -> FrenchFile:
    """Daily Mkt-RF, SMB, HML and RF."""
    return load("factors_daily", None, 8, refresh=refresh)


def factors_monthly(*, refresh: bool = False) -> FrenchFile:
    """Monthly Mkt-RF, SMB, HML and RF."""
    return load("factors_monthly", None, 6, refresh=refresh)
