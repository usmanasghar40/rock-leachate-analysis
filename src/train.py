from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
from xgboost import XGBRegressor

from src.config import load_config
from src.data import load_dataset, split_features_target
from src.features import add_lag_rolling_features, get_feature_columns


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
    x, y = split_features_target(df, feature_cols, config.data.target_col)

    model = XGBRegressor(**config.model.params)
    model.fit(x, y)

    model_path = Path(config.training.output_model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)

    features_path = Path(config.training.output_features_path)
    features_path.parent.mkdir(parents=True, exist_ok=True)
    features_path.write_text(json.dumps(feature_cols, indent=2), encoding="utf-8")

    print(f"Saved model to {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config YAML")
    args = parser.parse_args()
    main(args.config)
