from __future__ import annotations

from dataclasses import asdict
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.feature_selection import SelectKBest, mutual_info_regression, f_regression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from .config import ForecastConfig


def time_split(df: pd.DataFrame, train_ratio: float, val_ratio: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split in chronological order into train/val/test with ratios (train, val, rest)."""
    n = len(df)
    train_size = int(n * train_ratio)
    val_size = int(n * val_ratio)
    train_df = df.iloc[:train_size].copy()
    val_df = df.iloc[train_size: train_size + val_size].copy()
    test_df = df.iloc[train_size + val_size:].copy()
    return train_df, val_df, test_df


def select_features(train_data: pd.DataFrame, val_data: pd.DataFrame, features: List[str], target: str,
                    method: str = "mutual_info", k: int = 30) -> List[str]:
    """Select top-k features using SelectKBest (mutual_info_regression by default)."""
    combined = pd.concat([train_data, val_data], ignore_index=True)
    X = combined[features].fillna(combined[features].mean())
    y = combined[target].fillna(combined[target].mean())

    k = min(k, len(features))
    try:
        score_func = mutual_info_regression if method == "mutual_info" else f_regression
        selector = SelectKBest(score_func=score_func, k=k)
        selector.fit(X, y)
        idx = selector.get_support(indices=True)
        return [features[i] for i in idx] if idx is not None else features[:k]
    except Exception:
        return features[:k]


def hyperparameter_tuning(train_data: pd.DataFrame, val_data: pd.DataFrame, features: List[str], target: str,
                          config: ForecastConfig) -> Dict[str, Any]:
    """RandomizedSearchCV with TimeSeriesSplit (mirrors the notebook settings)."""
    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        random_state=config.random_state,
        n_jobs=-1,
        enable_categorical=True,
    )

    param_grid = {
        "n_estimators": [200, 400, 600],
        "learning_rate": [0.01, 0.03, 0.05],
        "max_depth": [4, 5, 6],
        "subsample": [0.8, 0.9],
        "colsample_bytree": [0.8, 0.9],
        "reg_alpha": [0, 0.1, 0.5],
        "reg_lambda": [0.1, 0.5, 1],
        "gamma": [0, 0.1],
        "min_child_weight": [1, 3],
    }

    tscv = TimeSeriesSplit(n_splits=config.n_splits_tscv)

    rs = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=config.n_iter_search,
        scoring="neg_root_mean_squared_error",
        cv=tscv,
        verbose=0,
        n_jobs=-1,
        random_state=config.random_state,
    )

    X_train = train_data[features]
    y_train = train_data[target]
    train_mean = X_train.mean()
    rs.fit(X_train.fillna(train_mean), y_train)

    # Evaluate on val for reporting (not used by search cv itself)
    X_val = val_data[features].fillna(train_mean)
    y_val = val_data[target]
    pred = rs.predict(X_val)
    rmse = float(np.sqrt(mean_squared_error(y_val, pred)))

    best = dict(rs.best_params_)
    best["_val_rmse"] = rmse
    return best


def train_multi_horizon(df_model_ready: pd.DataFrame, all_features: List[str], targets: List[str],
                        config: ForecastConfig) -> Tuple[Dict[str, xgb.XGBRegressor],
                                                         Dict[str, List[str]],
                                                         Dict[str, Dict[str, Any]],
                                                         pd.DataFrame]:
    """
    Train one model per target. Returns:
      - models dict
      - selected_features dict
      - best_params dict (includes _val_rmse)
      - metrics DataFrame (rmse/mae/r2 on test set)
    """
    train_df, val_df, test_df = time_split(df_model_ready, config.train_ratio, config.val_ratio)
    train_val_df = pd.concat([train_df, val_df], ignore_index=True)

    models: Dict[str, xgb.XGBRegressor] = {}
    selected_features_dict: Dict[str, List[str]] = {}
    best_params_dict: Dict[str, Dict[str, Any]] = {}
    rows = []

    for target in targets:
        selected = select_features(train_df, val_df, all_features, target, method="mutual_info", k=config.feature_select_k)
        selected_features_dict[target] = selected

        best_params = hyperparameter_tuning(train_df, val_df, selected, target, config)
        best_params_dict[target] = best_params

        # pop helper key
        fit_params = {k: v for k, v in best_params.items() if not k.startswith("_")}

        final_model = xgb.XGBRegressor(
            objective="reg:squarederror",
            random_state=config.random_state,
            n_jobs=-1,
            early_stopping_rounds=config.early_stopping_rounds,
            **fit_params,
        )

        X_train_val = train_val_df[selected]
        y_train_val = train_val_df[target]
        X_test = test_df[selected]
        y_test = test_df[target]

        train_val_mean = X_train_val.mean()
        X_train_val_filled = X_train_val.fillna(train_val_mean)
        X_test_filled = X_test.fillna(train_val_mean)

        X_val_eval = val_df[selected].fillna(train_val_mean)
        y_val_eval = val_df[target]

        final_model.fit(
            X_train_val_filled, y_train_val,
            eval_set=[(X_val_eval, y_val_eval)],
            verbose=False,
        )

        pred = final_model.predict(X_test_filled)

        rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
        mae = float(mean_absolute_error(y_test, pred))
        r2 = float(r2_score(y_test, pred))

        rows.append({"target": target, "rmse": rmse, "mae": mae, "r2": r2, "val_rmse": best_params.get("_val_rmse")})

        models[target] = final_model

    metrics = pd.DataFrame(rows).sort_values("target").reset_index(drop=True)
    return models, selected_features_dict, best_params_dict, metrics
