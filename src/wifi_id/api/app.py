"""Small programmatic API for embedding the pipeline."""

from __future__ import annotations

from pathlib import Path

from wifi_id.config import AppConfig
from wifi_id.data.readers import iter_rows
from wifi_id.data.writers import write_rows
from wifi_id.inference.classify import classify_rows
from wifi_id.models.base import load_checkpoint


def classify_dataset(config: AppConfig, input_path: str | Path, output_path: str | Path) -> int:
    model = load_checkpoint(config.model.checkpoint_path)
    predictions = classify_rows(
        iter_rows(input_path),
        model,
        config.features,
        label_mode=config.labeling.mode,
        positive_label=config.labeling.positive_label,
    )
    return write_rows(output_path, predictions)

