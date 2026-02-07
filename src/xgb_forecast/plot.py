from __future__ import annotations

from typing import Optional
import matplotlib.pyplot as plt
import pandas as pd


def plot_predictions(
    y_true: pd.DataFrame,
    y_pred: pd.DataFrame,
    outpath: Optional[str] = None,
    title: str = "XGBoost multi-horizon return prediction",
) -> None:
    """
    Plot predicted vs actual returns for each target on the test period.
    """
    targets = [c for c in y_true.columns if c in y_pred.columns]
    if not targets:
        raise ValueError("No common target columns to plot.")

    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(111)

    for t in targets:
        ax.plot(y_true.index, y_true[t], label=f"Actual {t}")
        ax.plot(y_pred.index, y_pred[t], linestyle="--", label=f"Pred {t}")

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Return")
    ax.legend(ncol=2)
    ax.grid(True)
    fig.tight_layout()

    if outpath:
        fig.savefig(outpath, dpi=150)
    else:
        plt.show()

    plt.close(fig)
