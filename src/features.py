from __future__ import annotations

from typing import Iterable, List

import pandas as pd


def add_lag_rolling_features(
    df: pd.DataFrame,
    group_col: str,
    time_col: str,
    target_col: str,
    lags: Iterable[int],
    rolling_windows: Iterable[int],
) -> pd.DataFrame:
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.sort_values([group_col, time_col])

    for lag in lags:
        df[f"{target_col}_lag_{lag}"] = (
            df.groupby(group_col)[target_col].shift(lag)
        )

    for window in rolling_windows:
        df[f"{target_col}_roll_mean_{window}"] = (
            df.groupby(group_col)[target_col]
            .shift(1)
            .rolling(window=window, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
        df[f"{target_col}_roll_std_{window}"] = (
            df.groupby(group_col)[target_col]
            .shift(1)
            .rolling(window=window, min_periods=1)
            .std()
            .reset_index(level=0, drop=True)
        )

    return df


def get_feature_columns(
    df: pd.DataFrame,
    target_col: str,
    exclude_cols: List[str],
) -> List[str]:
    return [
        col
        for col in df.columns
        if col not in exclude_cols and col != target_col
    ]
