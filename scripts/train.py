from __future__ import annotations

import argparse
from pathlib import Path
import json

import pandas as pd

from xgb_forecast.config import ForecastConfig
from xgb_forecast.data import load_csvs, basic_clean, pick_single_stock
from xgb_forecast.features import add_technical_features, make_targets
from xgb_forecast.model import train_multi_horizon, time_split
from xgb_forecast.evaluate import save_bundle, predict_on_df
from xgb_forecast.plot import plot_predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="Train multi-horizon XGBoost models for stock return forecasting.")
    parser.add_argument("--data", required=True, help="Path to a CSV file or a folder containing CSVs.")
    parser.add_argument("--stock_id", default=None, help="Optional: choose a specific stock_id (filename stem).")
    parser.add_argument("--index", type=int, default=1, help="If stock_id not set, choose the N-th stock (1-based).")
    parser.add_argument("--outdir", required=True, help="Output directory to save models and metrics.")
    parser.add_argument("--k", type=int, default=30, help="Number of selected features (SelectKBest).")
    args = parser.parse_args()

    cfg = ForecastConfig(feature_select_k=args.k)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) load + clean
    panel = load_csvs(args.data)
    panel = basic_clean(panel)
    stock_df = pick_single_stock(panel, stock_id=args.stock_id, index=args.index)

    # 2) features
    feat_df = add_technical_features(stock_df)
    feat_df = feat_df.dropna().copy()

    # 3) targets
    df_model_ready, targets, all_features = make_targets(feat_df, list(cfg.horizons))

    # 4) train
    models, feats_dict, params_dict, metrics = train_multi_horizon(df_model_ready, all_features, targets, cfg)

    metrics_path = outdir / "metrics.csv"
    metrics.to_csv(metrics_path, index=False)

    # 5) save a single bundle (models + metadata)
    train_df, val_df, test_df = time_split(df_model_ready, cfg.train_ratio, cfg.val_ratio)
    train_val_df = pd.concat([train_df, val_df], ignore_index=True)
    train_means = train_val_df[all_features].mean()

    bundle = {
        "config": cfg.__dict__,
        "targets": targets,
        "all_features": all_features,
        "selected_features": feats_dict,
        "best_params": params_dict,
        "train_means": train_means.to_dict(),
        "models": models,
    }
    save_bundle(str(outdir / "models.joblib"), bundle)

    # 6) plot on test set
    y_true = test_df[targets].copy()
    y_pred = predict_on_df(models, feats_dict, test_df, train_means)
    fig_path = Path("reports/figures")
    fig_path.mkdir(parents=True, exist_ok=True)
    plot_file = fig_path / f"{outdir.name}_predictions.png"
    plot_predictions(y_true, y_pred, outpath=str(plot_file), title=f"Predicted vs actual returns ({outdir.name})")

    # 7) metadata
    meta = {
        "data": str(args.data),
        "stock_id": args.stock_id,
        "index": args.index,
        "n_rows": int(len(df_model_ready)),
        "date_min": str(df_model_ready.index.min()),
        "date_max": str(df_model_ready.index.max()),
        "outputs": {
            "metrics_csv": str(metrics_path),
            "bundle": str(outdir / "models.joblib"),
            "plot": str(plot_file),
        },
    }
    (outdir / "run_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nDone.")


if __name__ == "__main__":
    main()
