from __future__ import annotations

import argparse
from pathlib import Path

from ts_stock_forecast.io import read_price_csv
from ts_stock_forecast.split import train_test_split_series
from ts_stock_forecast.metrics import rmse, mae
from ts_stock_forecast.plotting import plot_forecast
from ts_stock_forecast.models.arima import ARIMAConfig, fit_forecast


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, default="data/sample/600028.csv")
    p.add_argument("--train_ratio", type=float, default=0.8)
    p.add_argument("--p", type=int, default=1)
    p.add_argument("--d", type=int, default=1)
    p.add_argument("--q", type=int, default=1)
    p.add_argument("--out", type=str, default="figures/arima_forecast.png")
    args = p.parse_args()

    frame = read_price_csv(args.csv)
    y = frame.close

    split = train_test_split_series(y, train_ratio=args.train_ratio)
    cfg = ARIMAConfig(order=(args.p, args.d, args.q), seasonal_order=None)
    res = fit_forecast(split.train, steps=len(split.test), cfg=cfg)

    y_pred = res.pred
    y_pred.index = split.test.index  # align index

    print(f"RMSE: {rmse(split.test, y_pred):.6f}")
    print(f"MAE : {mae(split.test, y_pred):.6f}")

    plot_forecast(split.train, split.test, y_pred, title=f"ARIMA{cfg.order} Forecast", out_path=args.out)


if __name__ == "__main__":
    main()
