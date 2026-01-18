from __future__ import annotations

import pandas as pd


def add_basic_ts_features(df: pd.DataFrame, price_col: str = "close", lags=(1, 2, 3, 5, 10)) -> pd.DataFrame:
    """Create a minimal feature set for panel or single-series ML.

    - log return lags
    - rolling mean / std of returns

    This is intentionally minimal so that it works on the sample CSV.
    """
    out = df.copy()
    out["ret_1d"] = out[price_col].astype(float).pct_change()

    for k in lags:
        out[f"ret_lag_{k}"] = out["ret_1d"].shift(k)

    out["ret_ma_5"] = out["ret_1d"].rolling(5).mean()
    out["ret_std_5"] = out["ret_1d"].rolling(5).std()

    return out


def make_forward_return_targets(df: pd.DataFrame, price_col: str = "close", horizons=(1, 2, 3, 4, 5)) -> pd.DataFrame:
    out = df.copy()
    price = out[price_col].astype(float)
    for h in horizons:
        out[f"target_ret_tplus{h}"] = price.pct_change(h).shift(-h)
    return out
