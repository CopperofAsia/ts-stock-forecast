from __future__ import annotations

import glob
from pathlib import Path
from typing import Union, List, Optional

import pandas as pd


def load_csvs(path: Union[str, Path]) -> pd.DataFrame:
    """
    Load one CSV or a folder of CSVs and concatenate into a single DataFrame.

    - If `path` is a folder: loads all *.csv inside.
    - If `path` is a file: loads that file only.
    - Adds a `stock_id` column inferred from filename (without extension) if not present.
    """
    path = Path(path)
    files: List[Path]
    if path.is_dir():
        files = [Path(p) for p in glob.glob(str(path / "*.csv"))]
    else:
        files = [path]

    if not files:
        raise FileNotFoundError(f"No csv files found at: {path}")

    dfs = []
    for fp in files:
        df = pd.read_csv(fp)
        if "stock_id" not in df.columns:
            df["stock_id"] = fp.stem
        dfs.append(df)

    panel = pd.concat(dfs, axis=0, ignore_index=True)
    return panel


def basic_clean(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Basic cleaning consistent with the notebook:
    - drop pct_chg_b if exists (often duplicates pct_chg_f)
    - parse Date -> datetime
    - sort by stock_id, Date
    """
    df = panel.copy()

    if "pct_chg_b" in df.columns:
        df = df.drop(columns=["pct_chg_b"])

    if "Date" not in df.columns:
        raise ValueError("Input must contain a 'Date' column.")

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["stock_id", "Date"]).reset_index(drop=True)

    return df


def pick_single_stock(panel: pd.DataFrame, stock_id: Optional[str] = None, index: int = 1) -> pd.DataFrame:
    """
    Pick one stock from a multi-stock panel.
    - If `stock_id` is provided, filter by it.
    - Else pick the `index`-th stock_id (1-based) sorted lexicographically.
    Returns a Date-indexed DataFrame.
    """
    if "stock_id" not in panel.columns:
        raise ValueError("panel must have a 'stock_id' column.")

    if stock_id is None:
        ids = sorted(panel["stock_id"].astype(str).unique().tolist())
        if index < 1 or index > len(ids):
            raise IndexError(f"index out of range: {index}, available={len(ids)}")
        stock_id = ids[index - 1]

    df = panel[panel["stock_id"].astype(str) == str(stock_id)].copy()
    df = df.sort_values("Date").set_index("Date")
    return df
