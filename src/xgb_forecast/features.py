from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
import pandas_ta as ta


DEFAULT_PRICE_COLS = ["open", "high", "low", "close", "volume"]


def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators and engineered features (mirrors the notebook).
    Expects a Date-indexed DataFrame containing at least open/high/low/close/volume.
    """
    out = df.copy()

    # ensure numeric
    for col in DEFAULT_PRICE_COLS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    # MACD (appends MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9)
    out.ta.macd(close="close", fast=12, slow=26, signal=9, append=True)

    # RSI
    out["RSI_14"] = out.ta.rsi(close="close", length=14)
    out["RSI_21"] = out.ta.rsi(close="close", length=21)

    # Moving averages
    out["MA_5"] = out.ta.sma(close="close", length=5)
    out["MA_10"] = out.ta.sma(close="close", length=10)
    out["MA_20"] = out.ta.sma(close="close", length=20)
    out["MA_50"] = out.ta.sma(close="close", length=50)

    # Bollinger Bands (BBL_20_2.0, BBM_20_2.0, BBU_20_2.0, BBB_20_2.0, BBP_20_2.0)
    out.ta.bbands(close="close", length=20, std=2, append=True)

    # Williams %R
    out["WILLR_14"] = out.ta.willr(high="high", low="low", close="close", length=14)

    # Stochastic (STOCHk_14_3_3, STOCHd_14_3_3)
    out.ta.stoch(high="high", low="low", close="close", k=14, d=3, append=True)

    # ATR
    out["ATR_14"] = out.ta.atr(high="high", low="low", close="close", length=14)

    # Momentum / ROC
    out["MOM_10"] = out.ta.mom(close="close", length=10)
    out["ROC_10"] = out.ta.roc(close="close", length=10)

    # Volume indicators
    out["OBV"] = out.ta.obv(close="close", volume="volume")
    out["AD"] = out.ta.ad(high="high", low="low", close="close", volume="volume")

    # Price position
    out["price_position"] = (out["close"] - out["low"]) / (out["high"] - out["low"])

    # Volatility
    out["volatility_20"] = out["close"].rolling(20).std()

    # Price changes
    out["price_change_1d"] = out["close"].pct_change(1)
    out["price_change_5d"] = out["close"].pct_change(5)
    out["price_change_10d"] = out["close"].pct_change(10)

    # Advanced feature engineering
    out["RSI_MA_interaction"] = out["RSI_14"] * out["MA_20"] / out["close"]
    out["MACD_volume_interaction"] = (
        out["MACD_12_26_9"] * out["volume"] / out["volume"].rolling(20).mean()
    )
    out["price_momentum"] = out["close"] / out["MA_20"] * out["MOM_10"]

    out["trend_strength"] = (
        (out["MA_5"] - out["MA_20"]) / out["MA_20"] + (out["MA_10"] - out["MA_50"]) / out["MA_50"]
    )
    out["momentum_score"] = out["RSI_14"] * out["MOM_10"] / 100
    out["volatility_score"] = out["ATR_14"] / out["close"] * out["volatility_20"]

    out["bid_ask_spread_proxy"] = (out["high"] - out["low"]) / out["close"]
    out["volume_price_trend"] = out["OBV"] / out["close"]
    out["liquidity_score"] = out["volume"] / out["volume"].rolling(20).mean()

    out["price_acceleration"] = out["price_change_1d"].diff()
    out["volume_acceleration"] = out["volume"].pct_change().diff()

    out["relative_strength"] = out["close"] / out["close"].rolling(50).mean()
    out["relative_volume"] = out["volume"] / out["volume"].rolling(50).mean()

    out["volatility_ratio"] = out["volatility_20"] / out["volatility_20"].rolling(50).mean()
    out["price_volatility_interaction"] = out["price_change_1d"] * out["volatility_20"]

    out["trend_consistency"] = (
        (out["MA_5"] > out["MA_10"]).astype(int)
        + (out["MA_10"] > out["MA_20"]).astype(int)
        + (out["MA_20"] > out["MA_50"]).astype(int)
    )

    out["price_outlier"] = np.abs(out["price_change_1d"]) > out["price_change_1d"].rolling(20).std() * 2
    out["volume_outlier"] = out["volume"] > out["volume"].rolling(20).mean() * 2

    return out


def make_targets(df: pd.DataFrame, horizons: List[int]) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Create target columns: target_{h}d = close.shift(-h)/close - 1.
    Returns:
      df_model_ready, targets, all_features
    """
    out = df.copy()
    targets = []
    for h in horizons:
        name = f"target_{h}d"
        out[name] = (out["close"].shift(-h) / out["close"]) - 1
        targets.append(name)

    all_features = [c for c in out.columns if c not in targets]
    out = out.dropna(subset=targets).copy()

    # ensure numeric
    for col in all_features:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    return out, targets, all_features
