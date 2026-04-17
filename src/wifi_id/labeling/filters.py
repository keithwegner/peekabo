"""Dataset filtering helpers."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from wifi_id.config import FilterConfig
from wifi_id.labeling.targets import TargetRegistry
from wifi_id.parsing.dot11 import frame_family, normalize_mac


def passes_filters(
    row: dict[str, Any],
    config: FilterConfig,
    *,
    registry: TargetRegistry | None = None,
) -> bool:
    if config.known_targets_only:
        if registry is None or registry.target_for_mac(row.get("source_mac")) is None:
            return False

    family = frame_family(row.get("frame_type"))
    if family == "management" and not config.include_management:
        return False
    if family == "control" and not config.include_control:
        return False
    if family == "data" and not config.include_data:
        return False

    if not config.include_ap_originated and is_ap_originated(row, config):
        return False

    if config.channel_frequency is not None and row.get("channel_frequency") != config.channel_frequency:
        return False

    timestamp = row.get("timestamp")
    if config.start_time is not None and timestamp is not None:
        if float(timestamp) < _parse_time(config.start_time):
            return False
    if config.end_time is not None and timestamp is not None:
        if float(timestamp) > _parse_time(config.end_time):
            return False

    signal = row.get("rssi")
    if signal is None:
        signal = row.get("ssi")
    if config.rssi_min is not None and signal is not None and float(signal) < config.rssi_min:
        return False
    if config.rssi_max is not None and signal is not None and float(signal) > config.rssi_max:
        return False

    frame_size = row.get("data_size") or row.get("dot11_frame_len") or row.get("packet_len")
    if config.min_frame_size is not None and frame_size is not None:
        if int(frame_size) < config.min_frame_size:
            return False

    return True


def is_ap_originated(row: dict[str, Any], config: FilterConfig) -> bool:
    source = normalize_mac(row.get("source_mac"))
    ap_macs = {normalize_mac(mac) for mac in config.ap_macs}
    if source is not None and source in ap_macs:
        return True
    return bool(row.get("from_ds")) and not bool(row.get("to_ds"))


def _parse_time(value: float | str) -> float:
    if isinstance(value, (float, int)):
        return float(value)
    return datetime.fromisoformat(value).timestamp()

