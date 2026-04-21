"""YAML-driven application configuration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from peekaboo.models.registry import MODEL_MAPPINGS


class InputConfig(BaseModel):
    paths: list[Path] = Field(default_factory=list)
    live_interface: str | None = None


class FeatureConfig(BaseModel):
    data_size_mode: str = "dot11_frame_len"
    leakage_debug: bool = False
    mac_encoding: str = "hashed"
    impute_numeric: float = -1.0
    impute_categorical: str = "missing"
    include_missing_indicators: bool = False


class LabelConfig(BaseModel):
    mode: str = "binary_one_vs_rest"
    target_id: str | None = None
    positive_label: str = "target"
    negative_label: str = "other"


class FilterConfig(BaseModel):
    known_targets_only: bool = False
    include_ap_originated: bool = True
    include_management: bool = True
    include_control: bool = True
    include_data: bool = True
    channel_frequency: int | None = None
    start_time: float | str | None = None
    end_time: float | str | None = None
    rssi_min: float | None = None
    rssi_max: float | None = None
    min_frame_size: int | None = None
    ap_macs: list[str] = Field(default_factory=list)


class ModelConfig(BaseModel):
    model_id: str = "leveraging_bag"
    checkpoint_path: Path = Path("runs/model.pkl")
    params: dict[str, Any] = Field(default_factory=dict)


class DataConfig(BaseModel):
    normalized_path: Path = Path("runs/records.parquet")
    features_path: Path = Path("runs/features.parquet")
    labeled_path: Path = Path("runs/labeled.parquet")
    train_path: Path = Path("runs/train.parquet")
    test_path: Path = Path("runs/test.parquet")
    predictions_path: Path = Path("runs/predictions.parquet")
    rolling_path: Path = Path("runs/rolling.parquet")
    metrics_path: Path = Path("runs/metrics.json")


class SamplingConfig(BaseModel):
    percentage: float = 100.0
    balance_strategy: str = "none"
    class_weight_column: str = "sample_weight"


class SplitConfig(BaseModel):
    train_fraction: float = 0.9
    chronological: bool = True


class WindowConfig(BaseModel):
    frame_count: int = 100
    time_seconds: int = 30
    min_frames: int = 5
    present_ratio_threshold: float = 0.5
    mean_probability_threshold: float = 0.5
    max_probability_threshold: float = 0.8


class PresenceConfig(BaseModel):
    target_classes: list[str] = Field(default_factory=list)
    all_targets: bool = False


class ComparisonConfig(BaseModel):
    models: list[str] = Field(default_factory=lambda: list(MODEL_MAPPINGS))
    train_fractions: list[float] = Field(default_factory=lambda: [0.01, 0.1, 0.5, 0.9])
    output_dir: Path | None = None
    prepare_if_missing: bool = True

    @field_validator("models")
    @classmethod
    def validate_models(cls, value: list[str]) -> list[str]:
        unknown = sorted(set(value) - set(MODEL_MAPPINGS))
        if unknown:
            raise ValueError(f"Unsupported comparison model id(s): {', '.join(unknown)}")
        if not value:
            raise ValueError("comparison.models must include at least one model id")
        return value

    @field_validator("train_fractions")
    @classmethod
    def validate_train_fractions(cls, value: list[float]) -> list[float]:
        if not value:
            raise ValueError("comparison.train_fractions must include at least one fraction")
        invalid = [fraction for fraction in value if fraction <= 0 or fraction >= 1]
        if invalid:
            raise ValueError("comparison.train_fractions values must satisfy 0 < fraction < 1")
        return value


class AppConfig(BaseModel):
    input: InputConfig = Field(default_factory=InputConfig)
    target_registry_path: Path | None = None
    output_dir: Path = Path("runs/default")
    random_seed: int = 42
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    labeling: LabelConfig = Field(default_factory=LabelConfig)
    filters: FilterConfig = Field(default_factory=FilterConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    split: SplitConfig = Field(default_factory=SplitConfig)
    windowing: WindowConfig = Field(default_factory=WindowConfig)
    presence: PresenceConfig = Field(default_factory=PresenceConfig)
    comparison: ComparisonConfig = Field(default_factory=ComparisonConfig)

    @model_validator(mode="after")
    def default_comparison_output_dir(self) -> AppConfig:
        if self.comparison.output_dir is None:
            self.comparison.output_dir = self.output_dir / "comparison"
        return self


def _validate_config(data: dict[str, Any]) -> AppConfig:
    if hasattr(AppConfig, "model_validate"):
        return AppConfig.model_validate(data)  # type: ignore[attr-defined]
    return AppConfig.parse_obj(data)


def config_to_dict(config: AppConfig) -> dict[str, Any]:
    if hasattr(config, "model_dump"):
        return config.model_dump(mode="json")  # type: ignore[attr-defined]
    return config.dict()


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        if config_path.suffix.lower() == ".json":
            data = json.load(handle)
        else:
            data = yaml.safe_load(handle) or {}
    return _validate_config(data)


def write_run_config(config: AppConfig, output_dir: str | Path) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    run_config = output_path / "run_config.json"
    with run_config.open("w", encoding="utf-8") as handle:
        json.dump(config_to_dict(config), handle, indent=2, sort_keys=True, default=str)
    return run_config
