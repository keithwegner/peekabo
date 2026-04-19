"""Small programmatic API for embedding the pipeline."""

from __future__ import annotations

from pathlib import Path

from peekaboo.config import AppConfig
from peekaboo.data.readers import iter_rows
from peekaboo.data.writers import write_rows
from peekaboo.inference.classify import classify_rows
from peekaboo.models.base import load_checkpoint


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
