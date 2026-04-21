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


class CalibrationConfig(BaseModel):
    objective: str = "f1"
    present_ratio_thresholds: list[float] = Field(
        default_factory=lambda: [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    )
    mean_probability_thresholds: list[float] = Field(
        default_factory=lambda: [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    )
    max_probability_thresholds: list[float] = Field(
        default_factory=lambda: [0.5, 0.6, 0.7, 0.8, 0.9]
    )
    window_types: list[str] = Field(default_factory=lambda: ["frame_count", "time"])
    min_truth_frames: int = 1
    prepare_predictions_if_missing: bool = True
    output_dir: Path | None = None

    @field_validator("objective")
    @classmethod
    def validate_objective(cls, value: str) -> str:
        allowed = {"f1", "mcc", "precision", "recall"}
        if value not in allowed:
            raise ValueError(f"calibration.objective must be one of {', '.join(sorted(allowed))}")
        return value

    @field_validator(
        "present_ratio_thresholds",
        "mean_probability_thresholds",
        "max_probability_thresholds",
    )
    @classmethod
    def validate_thresholds(cls, value: list[float]) -> list[float]:
        if not value:
            raise ValueError("calibration threshold grids must not be empty")
        invalid = [threshold for threshold in value if threshold < 0 or threshold > 1]
        if invalid:
            raise ValueError("calibration threshold values must satisfy 0 <= threshold <= 1")
        return sorted(set(value))

    @field_validator("window_types")
    @classmethod
    def validate_window_types(cls, value: list[str]) -> list[str]:
        allowed = {"frame_count", "time"}
        if not value:
            raise ValueError("calibration.window_types must include at least one window type")
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise ValueError(f"Unsupported calibration window type(s): {', '.join(unknown)}")
        return list(dict.fromkeys(value))

    @field_validator("min_truth_frames")
    @classmethod
    def validate_min_truth_frames(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("calibration.min_truth_frames must be greater than 0")
        return value


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
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)
    comparison: ComparisonConfig = Field(default_factory=ComparisonConfig)

    @model_validator(mode="after")
    def default_derived_output_dirs(self) -> AppConfig:
        if self.calibration.output_dir is None:
            self.calibration.output_dir = self.output_dir / "calibration"
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
