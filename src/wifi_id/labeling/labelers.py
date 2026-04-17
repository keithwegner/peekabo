"""Label assignment strategies."""

from __future__ import annotations

from typing import Any, Iterable, Iterator

from wifi_id.config import LabelConfig
from wifi_id.labeling.targets import TargetRegistry


def label_for_row(
    row: dict[str, Any],
    registry: TargetRegistry,
    config: LabelConfig,
) -> str | None:
    source_target_id = registry.target_id_for_mac(row.get("source_mac"))

    if config.mode == "binary_one_vs_rest":
        if config.target_id is None:
            return config.positive_label if source_target_id is not None else config.negative_label
        return config.positive_label if source_target_id == config.target_id else config.negative_label

    if config.mode == "per_target_binary":
        if config.target_id is None:
            raise ValueError("per_target_binary requires labeling.target_id")
        return config.positive_label if source_target_id == config.target_id else config.negative_label

    if config.mode == "multiclass_known_targets_only":
        return source_target_id

    if config.mode == "multiclass_with_other":
        return source_target_id or config.negative_label

    raise ValueError(f"Unsupported label mode: {config.mode}")


def iter_labeled_rows(
    rows: Iterable[dict[str, Any]],
    registry: TargetRegistry,
    config: LabelConfig,
) -> Iterator[dict[str, Any]]:
    for row in rows:
        label = label_for_row(row, registry, config)
        if label is None:
            continue
        output = dict(row)
        output["label"] = label
        output["source_target_id"] = registry.target_id_for_mac(row.get("source_mac"))
        output["label_mode"] = config.mode
        yield output

