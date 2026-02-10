from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class ForecastConfig:
    """Configuration for the XGBoost multi-horizon return forecasting pipeline."""
    # Basic settings
    horizons: List[int] = (1, 2, 3, 4, 5)
    feature_select_k: int = 30
    train_ratio: float = 0.7
    val_ratio: float = 0.1
    random_state: int = 42

    # RandomizedSearchCV settings
    n_splits_tscv: int = 3
    n_iter_search: int = 15

    # XGBoost early stopping
    early_stopping_rounds: int = 50

    def target_names(self) -> List[str]:
        # 返回 ['target_1d', 'target_2d', 'target_3d', 'target_4d', 'target_5d']
        return [f"target_{h}d" for h in self.horizons]
