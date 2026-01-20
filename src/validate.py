from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from src.config import load_config
from src.data import load_dataset, split_features_target
from src.features import add_lag_rolling_features, get_feature_columns
from src.metrics import regression_metrics


def leave_one_group_out_cv(
    df: pd.DataFrame,
    group_col: str,
    feature_cols: list[str],
    target_col: str,
    model_params: dict,
) -> dict:
    metrics = []
    groups = df[group_col].unique()
    for group in groups:
        train_df = df[df[group_col] != group]
        test_df = df[df[group_col] == group]

        x_train, y_train = split_features_target(train_df, feature_cols, target_col)
        x_test, y_test = split_features_target(test_df, feature_cols, target_col)

        model = XGBRegressor(**model_params)
        model.fit(x_train, y_train)

        preds = model.predict(x_test)
        metrics.append(regression_metrics(y_test, preds))

    mean_metrics = {
        key: float(np.mean([m[key] for m in metrics])) for key in metrics[0]
    }
    return {"per_group": metrics, "mean": mean_metrics}


def main(config_path: str) -> None:
    config = load_config(config_path)
    df = load_dataset(config.data.path)

    df = add_lag_rolling_features(
        df,
        group_col=config.data.rock_id_col,
        time_col=config.data.time_col,
        target_col=config.data.target_col,
        lags=config.features.lags,
        rolling_windows=config.features.rolling_windows,
    )

    df = df.dropna().reset_index(drop=True)

    exclude_cols = [config.data.rock_id_col, config.data.time_col]
    feature_cols = get_feature_columns(df, config.data.target_col, exclude_cols)

    results = leave_one_group_out_cv(
        df=df,
        group_col=config.data.rock_id_col,
        feature_cols=feature_cols,
        target_col=config.data.target_col,
        model_params=config.model.params,
    )

    output = Path("reports") / "validation_metrics.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("Leave-One-Rock-Out metrics:", results["mean"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config YAML")
    args = parser.parse_args()
    main(args.config)
