"""YAML-driven application configuration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from pydantic import BaseModel, Field


class InputConfig(BaseModel):
    paths: List[Path] = Field(default_factory=list)
    live_interface: Optional[str] = None


class FeatureConfig(BaseModel):
    data_size_mode: str = "dot11_frame_len"
    leakage_debug: bool = False
    mac_encoding: str = "hashed"
    impute_numeric: float = -1.0
    impute_categorical: str = "missing"
    include_missing_indicators: bool = False


class LabelConfig(BaseModel):
    mode: str = "binary_one_vs_rest"
    target_id: Optional[str] = None
    positive_label: str = "target"
    negative_label: str = "other"


class FilterConfig(BaseModel):
    known_targets_only: bool = False
    include_ap_originated: bool = True
    include_management: bool = True
    include_control: bool = True
    include_data: bool = True
    channel_frequency: Optional[int] = None
    start_time: Optional[Union[float, str]] = None
    end_time: Optional[Union[float, str]] = None
    rssi_min: Optional[float] = None
    rssi_max: Optional[float] = None
    min_frame_size: Optional[int] = None
    ap_macs: List[str] = Field(default_factory=list)


class ModelConfig(BaseModel):
    model_id: str = "leveraging_bag"
    checkpoint_path: Path = Path("runs/model.pkl")
    params: Dict[str, Any] = Field(default_factory=dict)


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


class AppConfig(BaseModel):
    input: InputConfig = Field(default_factory=InputConfig)
    target_registry_path: Optional[Path] = None
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


def _validate_config(data: Dict[str, Any]) -> AppConfig:
    if hasattr(AppConfig, "model_validate"):
        return AppConfig.model_validate(data)  # type: ignore[attr-defined]
    return AppConfig.parse_obj(data)


def config_to_dict(config: AppConfig) -> Dict[str, Any]:
    if hasattr(config, "model_dump"):
        return config.model_dump(mode="json")  # type: ignore[attr-defined]
    return config.dict()


def load_config(path: Union[str, Path]) -> AppConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        if config_path.suffix.lower() == ".json":
            data = json.load(handle)
        else:
            data = yaml.safe_load(handle) or {}
    return _validate_config(data)


def write_run_config(config: AppConfig, output_dir: Union[str, Path]) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    run_config = output_path / "run_config.json"
    with run_config.open("w", encoding="utf-8") as handle:
        json.dump(config_to_dict(config), handle, indent=2, sort_keys=True, default=str)
    return run_config
