from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import pandas as pd


@dataclass(frozen=True)
class TrainTestSplit:
    train: pd.Series
    test: pd.Series


def train_test_split_series(series: pd.Series, train_ratio: float = 0.8) -> TrainTestSplit:
    """Chronological split (no shuffling)."""
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be in (0,1)")
    n = len(series)
    cut = int(n * train_ratio)
    return TrainTestSplit(train=series.iloc[:cut], test=series.iloc[cut:])
