from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd


def load_dataset(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def split_features_target(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
) -> Tuple[pd.DataFrame, pd.Series]:
    return df[feature_cols], df[target_col]
