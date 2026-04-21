"""Rolling target-presence aggregation."""

from __future__ import annotations

import json
from collections.abc import Iterable
from statistics import mean
from typing import Any

from peekaboo.config import WindowConfig
from peekaboo.parsing.records import RollingPresenceRow


def rolling_frame_count(
    predictions: Iterable[dict[str, Any]],
    *,
    target_class: str,
    config: WindowConfig,
) -> list[dict[str, Any]]:
    rows = list(predictions)
    output: list[dict[str, Any]] = []
    for start in range(0, len(rows), config.frame_count):
        chunk = rows[start : start + config.frame_count]
        if chunk:
            output.append(
                summarize_presence_window(
                    chunk,
                    start,
                    start + len(chunk) - 1,
                    target_class,
                    config,
                )
            )
    return output


def rolling_time(
    predictions: Iterable[dict[str, Any]],
    *,
    target_class: str,
    config: WindowConfig,
) -> list[dict[str, Any]]:
    buckets: dict[int, list[dict[str, Any]]] = {}
    for row in predictions:
        timestamp = row.get("timestamp")
        if timestamp is None:
            continue
        bucket = int(float(timestamp) // config.time_seconds)
        buckets.setdefault(bucket, []).append(row)
    output = []
    for bucket, rows in sorted(buckets.items()):
        start = bucket * config.time_seconds
        end = start + config.time_seconds
        output.append(summarize_presence_window(rows, start, end, target_class, config))
    return output


def rolling_aggregates(
    predictions: Iterable[dict[str, Any]],
    *,
    target_class: str,
    config: WindowConfig,
) -> list[dict[str, Any]]:
    rows = list(predictions)
    return rolling_aggregates_for_targets(rows, target_classes=[target_class], config=config)


def rolling_aggregates_for_targets(
    predictions: Iterable[dict[str, Any]],
    *,
    target_classes: Iterable[str],
    config: WindowConfig,
) -> list[dict[str, Any]]:
    rows = list(predictions)
    output: list[dict[str, Any]] = []
    for target_class in target_classes:
        output.extend(
            _rolling_aggregates_one_target(rows, target_class=target_class, config=config)
        )
    return output


def _rolling_aggregates_one_target(
    rows: list[dict[str, Any]],
    *,
    target_class: str,
    config: WindowConfig,
) -> list[dict[str, Any]]:
    frame_rows = rolling_frame_count(rows, target_class=target_class, config=config)
    for row in frame_rows:
        row["window_type"] = "frame_count"
    time_rows = rolling_time(rows, target_class=target_class, config=config)
    for row in time_rows:
        row["window_type"] = "time"
    return frame_rows + time_rows


def summarize_presence_window(
    rows: list[dict[str, Any]],
    window_start: float | int,
    window_end: float | int,
    target_class: str,
    config: WindowConfig,
) -> dict[str, Any]:
    probabilities = [_target_probability(row, target_class) for row in rows]
    positives = [row for row in rows if row.get("predicted_class") == target_class]
    positive_ratio = len(positives) / len(rows) if rows else None
    mean_probability = mean(probabilities) if probabilities else None
    max_probability = max(probabilities) if probabilities else None
    state = _presence_state(len(rows), positive_ratio, mean_probability, max_probability, config)
    return RollingPresenceRow(
        window_start=window_start,
        window_end=window_end,
        target_id=target_class,
        frame_count=len(rows),
        mean_probability=mean_probability,
        max_probability=max_probability,
        positive_prediction_ratio=positive_ratio,
        state=state,
    ).to_dict()


def _target_probability(row: dict[str, Any], target_class: str) -> float:
    raw = row.get("probabilities_json")
    if raw:
        try:
            probabilities = json.loads(raw)
            if target_class in probabilities:
                return float(probabilities[target_class])
        except Exception:
            pass
    if row.get("predicted_class") == target_class:
        return float(row.get("confidence") or 1.0)
    return 0.0


def _presence_state(
    frame_count: int,
    positive_ratio: float | None,
    mean_probability: float | None,
    max_probability: float | None,
    config: WindowConfig,
) -> str:
    if frame_count < config.min_frames:
        return "uncertain"
    ratio_ok = positive_ratio is not None and positive_ratio >= config.present_ratio_threshold
    mean_ok = mean_probability is not None and mean_probability >= config.mean_probability_threshold
    max_ok = max_probability is not None and max_probability >= config.max_probability_threshold
    if ratio_ok and (mean_ok or max_ok):
        return "present"
    if not ratio_ok and not mean_ok and not max_ok:
        return "absent"
    return "uncertain"
