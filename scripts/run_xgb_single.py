from __future__ import annotations

import argparse

import pandas as pd

from ts_stock_forecast.io import read_price_csv
from ts_stock_forecast.features import add_basic_ts_features, make_forward_return_targets
from ts_stock_forecast.models.xgb import train_single_horizon


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, default="data/sample/600028.csv")
    p.add_argument("--horizon", type=int, default=1)
    args = p.parse_args()

    frame = read_price_csv(args.csv)
    df = frame.df.reset_index().rename(columns={"Date": "date"})
    # ensure date column exists after rename; read_price_csv sets index and lowers cols
    df = frame.df.copy().reset_index().rename(columns={frame.date_col: "date"})

    df = add_basic_ts_features(df, price_col="close")
    df = make_forward_return_targets(df, price_col="close", horizons=(args.horizon,))

    target_col = f"target_ret_tplus{args.horizon}"
    feature_cols = [c for c in df.columns if c.startswith("ret_") and c != target_col]

    # drop rows with NaNs created by lags/targets
    df = df.dropna(subset=feature_cols + [target_col]).reset_index(drop=True)

    model, cv_rmse = train_single_horizon(df, feature_cols=feature_cols, target_col=target_col)
    print(f"Trained XGB for horizon T+{args.horizon}. TimeSeriesCV RMSE (return space): {cv_rmse:.6f}")


if __name__ == "__main__":
    main()
