from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd


def plot_forecast(
    y_train: pd.Series,
    y_test: pd.Series,
    y_pred: pd.Series,
    title: str,
    out_path: Optional[str | Path] = None,
) -> None:
    """Plot train/test and forecast."""
    plt.figure()
    y_train.plot(label="train")
    y_test.plot(label="test")
    y_pred.plot(label="forecast")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=200)
    plt.show()
