from __future__ import annotations

from typing import Dict, List, Tuple, Any

import pandas as pd
import numpy as np
import joblib
import xgboost as xgb


def predict_on_df(
    models: Dict[str, xgb.XGBRegressor],
    features_dict: Dict[str, List[str]],
    df: pd.DataFrame,
    train_means: pd.Series,
) -> pd.DataFrame:
    """
    Predict each target for every row in df.
    Missing values are filled with train_means (computed on train+val of the same feature set).
    """
    out = pd.DataFrame(index=df.index)
    for target, model in models.items():
        feats = features_dict[target]
        X = df[feats].fillna(train_means[feats])
        out[target] = model.predict(X)
    return out


def predict_last_row_prices(
    models: Dict[str, xgb.XGBRegressor],
    features_dict: Dict[str, List[str]],
    last_row: pd.DataFrame,
    base_price: float,
    train_means: pd.Series,
) -> pd.DataFrame:
    """
    Predict last-row future returns and convert to implied prices: price = base_price*(1+ret).
    Returns a small table with horizon, predicted_return, implied_price.
    """
    rows = []
    for target, model in models.items():
        feats = features_dict[target]
        X = last_row[feats].fillna(train_means[feats])
        pred_ret = float(model.predict(X)[0])
        rows.append({
            "target": target,
            "pred_return": pred_ret,
            "implied_price": base_price * (1.0 + pred_ret),
        })
    return pd.DataFrame(rows).sort_values("target").reset_index(drop=True)


def save_bundle(path: str, bundle: dict) -> None:
    joblib.dump(bundle, path)


def load_bundle(path: str) -> dict:
    return joblib.load(path)
