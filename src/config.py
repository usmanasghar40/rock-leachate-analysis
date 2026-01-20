from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import yaml


@dataclass
class DataConfig:
    path: str
    rock_id_col: str
    time_col: str
    target_col: str


@dataclass
class FeatureConfig:
    lags: List[int]
    rolling_windows: List[int]


@dataclass
class ModelConfig:
    type: str
    params: Dict[str, Any]


@dataclass
class TrainingConfig:
    test_size: float
    output_model_path: str
    output_features_path: str


@dataclass
class ProjectConfig:
    data: DataConfig
    features: FeatureConfig
    model: ModelConfig
    training: TrainingConfig


def load_config(path: str | Path) -> ProjectConfig:
    with open(path, "r", encoding="utf-8") as file:
        raw = yaml.safe_load(file)

    return ProjectConfig(
        data=DataConfig(**raw["data"]),
        features=FeatureConfig(**raw["features"]),
        model=ModelConfig(**raw["model"]),
        training=TrainingConfig(**raw["training"]),
    )
