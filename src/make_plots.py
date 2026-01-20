from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from xgboost import XGBRegressor

from src.config import load_config
from src.data import load_dataset, split_features_target
from src.features import add_lag_rolling_features, get_feature_columns


def plot_predictions(y_true, y_pred, output_path: Path) -> None:
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.3, s=8)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], color="red", linewidth=1)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Actual vs Predicted")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_feature_importance(feature_cols, importances, output_path: Path) -> None:
    order = np.argsort(importances)[::-1][:15]
    top_features = [feature_cols[i] for i in order]
    top_importances = importances[order]

    plt.figure(figsize=(8, 6))
    plt.barh(top_features[::-1], top_importances[::-1], color="#4C72B0")
    plt.xlabel("Importance")
    plt.title("Top Feature Importances")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


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
    preds = model.predict(x)

    figures_dir = Path("reports") / "figures"
    plot_predictions(y.to_numpy(), preds, figures_dir / "pred_vs_actual.png")
    plot_feature_importance(
        feature_cols, model.feature_importances_, figures_dir / "feature_importance.png"
    )

    print(f"Saved plots to {figures_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config YAML")
    args = parser.parse_args()
    main(args.config)
