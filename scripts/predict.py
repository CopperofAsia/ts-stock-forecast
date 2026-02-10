# scripts/predict.py
# ------------------------------------------------------------
# Predict multi-horizon returns using a trained XGBoost bundle,
# and save ALL outputs to a user-specified outdir.
#
# Fixes:
# 1) feature_names mismatch -> align to booster.feature_names
# 2) model dict keys like "target_1d" -> robust horizon parsing
# ------------------------------------------------------------

import argparse
import re
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from xgb_forecast.data import load_csvs, basic_clean, pick_single_stock
from xgb_forecast.features import add_technical_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load a trained run and predict the last row (T+1..T+H)."
    )
    parser.add_argument("--data", required=True, help="Path to a CSV file or a folder containing CSVs.")
    parser.add_argument("--bundle", required=True, help="Path to models.joblib produced by scripts/train.py")
    parser.add_argument("--stock_id", default=None, help="Optional: choose a specific stock_id (filename stem).")
    parser.add_argument("--index", type=int, default=1, help="If stock_id not set, choose the N-th stock (1-based).")
    parser.add_argument("--outdir", required=True, help="Directory to save prediction outputs.")
    parser.add_argument("--verbose", action="store_true", help="Print diagnostic info.")
    return parser.parse_args()


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _parse_horizon(key) -> int:
    """
    Convert model key to horizon int.
    Supports:
      - 1, 2, ...
      - "1", "2"
      - "target_1d", "target_5d"
      - "h1", "H3"
      - "t+4" / "T+4"
    """
    if isinstance(key, int):
        return key
    s = str(key)

    # Direct numeric string
    if s.isdigit():
        return int(s)

    # Find first integer substring
    m = re.search(r"(\d+)", s)
    if m:
        return int(m.group(1))

    raise ValueError(f"Cannot parse horizon from model key: {key!r}")


def _align_to_booster_features(
    X_last: pd.DataFrame,
    booster_feature_names: list[str],
    train_means: pd.Series | None,
    verbose: bool = False,
    horizon_label: str = "?",
) -> pd.DataFrame:
    """
    Align a single-row dataframe to the exact feature set and order used in training.
    - Drops extra columns not seen during training.
    - Adds missing training columns and fills with train_means (if available) or 0.
    """
    X_num = X_last.select_dtypes(include=[np.number]).copy()

    expected = list(booster_feature_names or [])
    if not expected:
        if verbose:
            print(f"[{horizon_label}] Warning: booster.feature_names missing; using numeric columns as-is.")
        return X_num

    Xh = X_num.reindex(columns=expected)

    if train_means is not None and len(train_means) > 0:
        fill = train_means.reindex(expected)
        Xh = Xh.fillna(fill)

    Xh = Xh.fillna(0.0)

    if verbose:
        missing = [c for c in expected if c not in X_num.columns]
        extra = [c for c in X_num.columns if c not in expected]
        if missing:
            print(f"[{horizon_label}] Missing {len(missing)} features -> filled by train_means/0. e.g. {missing[:6]}")
        if extra:
            print(f"[{horizon_label}] Dropped {len(extra)} extra numeric features. e.g. {extra[:6]}")

    return Xh


def main() -> None:
    args = parse_args()

    outdir = Path(args.outdir)
    _ensure_dir(outdir)

    bundle = joblib.load(args.bundle)
    models = bundle["models"]
    train_means = pd.Series(bundle.get("train_means", {}))

    # Load data
    panel = load_csvs(args.data)
    panel = basic_clean(panel)
    stock_df = pick_single_stock(panel, stock_id=args.stock_id, index=args.index)

    # Feature engineering
    feat_df = add_technical_features(stock_df).dropna().copy()

    # Last row as T
    X_last = feat_df.iloc[[-1]]
    last_close = float(stock_df["close"].iloc[-1])

    preds: dict[int, float] = {}

    # Sort models by parsed horizon
    items = list(models.items())
    items_sorted = sorted(items, key=lambda kv: _parse_horizon(kv[0]))

    for key, model in items_sorted:
        h = _parse_horizon(key)
        horizon_label = str(key)

        booster_feats = model.get_booster().feature_names
        Xh = _align_to_booster_features(
            X_last=X_last,
            booster_feature_names=booster_feats,
            train_means=train_means,
            verbose=args.verbose,
            horizon_label=horizon_label,
        )
        preds[h] = float(model.predict(Xh)[0])

    # Save predicted returns
    pred_ret = (
        pd.Series(preds, name="pred_return")
        .sort_index()
        .rename_axis("horizon")
        .to_frame()
    )
    pred_ret.to_csv(outdir / "forecast_returns.csv")

    # Build implied price path (simple compounding from last_close)
    prices = [last_close]
    for r in pred_ret["pred_return"].tolist():
        prices.append(prices[-1] * (1.0 + r))

    pred_price = pd.DataFrame(
        {
            "horizon": ["T"] + [f"T+{i}" for i in pred_ret.index.tolist()],
            "price": prices,
        }
    )
    pred_price.to_csv(outdir / "forecast_prices.csv", index=False)

    # Save summary
    with open(outdir / "forecast_summary.txt", "w", encoding="utf-8") as f:
        f.write("XGBoost multi-horizon forecast\n")
        f.write("=" * 50 + "\n")
        f.write(f"Last close price: {last_close:.4f}\n\n")
        for h, r in pred_ret["pred_return"].items():
            f.write(f"T+{int(h)}: return = {float(r):.6f} ({float(r):.4%})\n")

    # Plot price path
    fig_path = Path("reports/figures")
    plt.figure(figsize=(9, 4.5))
    plt.plot(pred_price["horizon"], pred_price["price"], marker="o")
    plt.title("Forecasted Price Path")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(fig_path / f"{outdir.name}_predictions.png", dpi=150)
    plt.close()

    print("\nDone.")


if __name__ == "__main__":
    main()
