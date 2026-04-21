"""Streaming target-presence inference."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from peekaboo.config import AppConfig, WindowConfig
from peekaboo.inference.aggregate import summarize_presence_window
from peekaboo.labeling.targets import TargetRegistry

MULTICLASS_LABEL_MODES = {"multiclass_known_targets_only", "multiclass_with_other"}


class StreamingPresenceEngine:
    """Convert prediction rows into completed presence windows."""

    def __init__(self, *, target_class: str, config: WindowConfig) -> None:
        self.target_class = target_class
        self.config = config
        self._frame_rows: list[dict[str, Any]] = []
        self._frame_start = 0
        self._time_bucket: int | None = None
        self._time_rows: list[dict[str, Any]] = []

    def process(self, prediction: dict[str, Any]) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        self._frame_rows.append(prediction)
        if len(self._frame_rows) >= self.config.frame_count:
            events.append(self._flush_frame_window(final=False))

        timestamp = prediction.get("timestamp")
        if timestamp is not None:
            bucket = int(float(timestamp) // self.config.time_seconds)
            if self._time_bucket is None:
                self._time_bucket = bucket
            elif bucket != self._time_bucket:
                events.append(self._flush_time_window(final=False))
                self._time_bucket = bucket
            self._time_rows.append(prediction)
        return events

    def flush(self) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        if self._frame_rows:
            events.append(self._flush_frame_window(final=True))
        if self._time_rows:
            events.append(self._flush_time_window(final=True))
        return events

    def _flush_frame_window(self, *, final: bool) -> dict[str, Any]:
        rows = self._frame_rows
        window_start = self._frame_start
        window_end = self._frame_start + len(rows) - 1
        self._frame_start += len(rows)
        self._frame_rows = []
        event = summarize_presence_window(
            rows,
            window_start,
            window_end,
            self.target_class,
            self.config,
        )
        event["window_type"] = "frame_count"
        event["final_window"] = final
        return event

    def _flush_time_window(self, *, final: bool) -> dict[str, Any]:
        rows = self._time_rows
        bucket = self._time_bucket or 0
        self._time_rows = []
        event = summarize_presence_window(
            rows,
            bucket * self.config.time_seconds,
            (bucket + 1) * self.config.time_seconds,
            self.target_class,
            self.config,
        )
        event["window_type"] = "time"
        event["final_window"] = final
        return event


class MultiTargetStreamingPresenceEngine:
    """Run one streaming presence engine per target class."""

    def __init__(self, *, target_classes: Iterable[str], config: WindowConfig) -> None:
        self.target_classes = unique_target_classes(target_classes)
        self.engines = {
            target_class: StreamingPresenceEngine(target_class=target_class, config=config)
            for target_class in self.target_classes
        }

    def process(self, prediction: dict[str, Any]) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        for engine in self.engines.values():
            events.extend(engine.process(prediction))
        return events

    def flush(self) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        for engine in self.engines.values():
            events.extend(engine.flush())
        return events


class PresenceStateTracker:
    """Track state changes so terminal output stays concise."""

    def __init__(self) -> None:
        self._last_state: dict[tuple[str, str], str] = {}

    def is_change(self, event: dict[str, Any]) -> bool:
        key = (str(event.get("target_id")), str(event.get("window_type")))
        state = str(event.get("state"))
        previous = self._last_state.get(key)
        self._last_state[key] = state
        return previous != state


def resolve_presence_target_classes(
    config: AppConfig,
    *,
    target_classes: Iterable[str] | None = None,
    all_targets: bool | None = None,
    registry: TargetRegistry | None = None,
) -> list[str]:
    explicit_targets = unique_target_classes(target_classes or [])
    if explicit_targets:
        return explicit_targets
    if all_targets is True:
        return _all_registry_targets(config, registry=registry)
    if all_targets is False:
        return [_default_presence_target_class(config)]
    if config.presence.target_classes:
        return unique_target_classes(config.presence.target_classes)
    if config.presence.all_targets:
        return _all_registry_targets(config, registry=registry)
    return [_default_presence_target_class(config)]


def unique_target_classes(target_classes: Iterable[str]) -> list[str]:
    selected: list[str] = []
    seen: set[str] = set()
    for target_class in target_classes:
        value = str(target_class)
        if value and value not in seen:
            selected.append(value)
            seen.add(value)
    return selected


def format_presence_event(event: dict[str, Any]) -> str:
    return (
        f"{event.get('window_type')} target={event.get('target_id')} "
        f"state={event.get('state')} frames={event.get('frame_count')} "
        f"mean={_fmt(event.get('mean_probability'))} "
        f"max={_fmt(event.get('max_probability'))} "
        f"positive_ratio={_fmt(event.get('positive_prediction_ratio'))}"
    )


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _all_registry_targets(
    config: AppConfig,
    *,
    registry: TargetRegistry | None,
) -> list[str]:
    if config.labeling.mode not in MULTICLASS_LABEL_MODES:
        raise ValueError(
            "--all-targets requires a multiclass label mode "
            "(multiclass_known_targets_only or multiclass_with_other)"
        )
    if registry is None:
        if config.target_registry_path is None:
            raise ValueError("--all-targets requires target_registry_path")
        registry = TargetRegistry.from_file(config.target_registry_path)
    target_classes = sorted(registry.enabled_target_ids())
    if not target_classes:
        raise ValueError("--all-targets found no enabled targets in the target registry")
    return target_classes


def _default_presence_target_class(config: AppConfig) -> str:
    if config.labeling.mode in {"binary_one_vs_rest", "per_target_binary"}:
        return config.labeling.positive_label
    return config.labeling.target_id or config.labeling.positive_label
