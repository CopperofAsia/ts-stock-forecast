from __future__ import annotations

import argparse

from ts_stock_forecast.io import read_price_csv
from ts_stock_forecast.split import train_test_split_series
from ts_stock_forecast.metrics import rmse, mae
from ts_stock_forecast.models.baselines import mean_forecast, naive_forecast, seasonal_naive_forecast, drift_forecast


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, default="data/sample/600028.csv")
    p.add_argument("--train_ratio", type=float, default=0.8)
    p.add_argument("--season", type=int, default=5)
    args = p.parse_args()

    frame = read_price_csv(args.csv)
    y = frame.close
    split = train_test_split_series(y, train_ratio=args.train_ratio)

    methods = [
        mean_forecast(split.train, split.test),
        naive_forecast(split.train, split.test),
        seasonal_naive_forecast(split.train, split.test, season_length=args.season),
        drift_forecast(split.train, split.test),
    ]

    for m in methods:
        print(f"{m.name:>16} | RMSE={rmse(split.test, m.pred):.6f} | MAE={mae(split.test, m.pred):.6f}")


if __name__ == "__main__":
    main()
