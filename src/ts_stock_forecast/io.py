from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd


@dataclass
class OHLCVFrame:
    """A lightweight container for a single-stock OHLCV dataframe."""

    df: pd.DataFrame
    date_col: str = "Date"

    @property
    def close(self) -> pd.Series:
        return self.df["close"].astype(float)


def read_price_csv(
    path: str | Path,
    date_col: str = "Date",
    tz: Optional[str] = None,
) -> OHLCVFrame:
    """Read a single-stock CSV like the provided `600028.csv`.

    Expected columns include:
      - Date
      - open, high, low, close
      - (optional) volume, amount, vwap, pb, pe_ttm, etc.

    The function:
      1) parses Date into datetime
      2) sorts by Date ascending
      3) sets Date as index
      4) lowercases column names for consistency
    """
    path = Path(path)
    df = pd.read_csv(path)
    if date_col not in df.columns:
        raise ValueError(f"Date column '{date_col}' not found. Available: {list(df.columns)}")

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col)
    if tz:
        df[date_col] = df[date_col].dt.tz_localize(tz, nonexistent="shift_forward", ambiguous="NaT")

    # normalize column names (keep original Date column name for index)
    cols = {c: c.lower() for c in df.columns if c != date_col}
    df = df.rename(columns=cols)

    df = df.set_index(date_col)
    return OHLCVFrame(df=df, date_col=date_col)
