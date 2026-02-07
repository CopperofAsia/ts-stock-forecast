from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from xgb_forecast.data import load_csvs, basic_clean, pick_single_stock
from xgb_forecast.features import add_technical_features, make_targets
from xgb_forecast.evaluate import load_bundle, predict_last_row_prices


def main() -> None:
    parser = argparse.ArgumentParser(description="Load a trained run and predict the last row (T+1..T+5).")
    parser.add_argument("--data", required=True, help="Path to a CSV file or a folder containing CSVs.")
    parser.add_argument("--bundle", required=True, help="Path to models.joblib produced by scripts/train.py")
    parser.add_argument("--stock_id", default=None, help="Optional: choose a specific stock_id (filename stem).")
    parser.add_argument("--index", type=int, default=1, help="If stock_id not set, choose the N-th stock (1-based).")
    args = parser.parse_args()

    bundle = load_bundle(args.bundle)
    models = bundle["models"]
    features_dict = bundle["selected_features"]
    train_means = pd.Series(bundle["train_means"])

    panel = load_csvs(args.data)
    panel = basic_clean(panel)
    stock_df = pick_single_stock(panel, stock_id=args.stock_id, index=args.index)

    feat_df = add_technical_features(stock_df).dropna().copy()

    # Make targets just to align columns; we won't use target rows at the end
    df_ready, targets, all_features = make_targets(feat_df, horizons=[1, 2, 3, 4, 5])

    last_row = df_ready.iloc[[-1]]
    base_price = float(last_row["close"].iloc[0])

    table = predict_last_row_prices(models, features_dict, last_row, base_price, train_means)
    print("\nLast-row prediction (return + implied price):")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
