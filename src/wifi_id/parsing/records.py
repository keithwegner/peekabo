"""Stable row contracts shared across the pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class PacketRecord:
    timestamp: float | None
    source_file: str
    packet_index: int
    packet_len: int | None = None
    capture_len: int | None = None
    original_len: int | None = None
    radiotap_len: int | None = None
    dot11_frame_len: int | None = None
    payload_len: int | None = None
    hour: int | None = None
    data_rate: float | None = None
    channel_frequency: int | None = None
    ssi: float | None = None
    rssi: float | None = None
    frame_type: int | None = None
    frame_subtype: int | None = None
    to_ds: bool | None = None
    from_ds: bool | None = None
    addr1: str | None = None
    addr2: str | None = None
    addr3: str | None = None
    addr4: str | None = None
    source_mac: str | None = None
    destination_mac: str | None = None
    sequence_number: int | None = None
    retry: bool | None = None
    protected: bool | None = None
    parse_ok: bool = True
    parse_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FeatureRow:
    timestamp: float | None
    source_file: str
    packet_index: int
    hour: int | None
    data_rate: float | None
    ssi: float | None
    frame_type: int | None
    frame_subtype: int | None
    source_mac: str | None
    destination_mac: str | None
    data_size: int | None
    total_packet_len: int | None = None
    dot11_frame_len: int | None = None
    payload_len: int | None = None
    channel_frequency: int | None = None
    retry: bool | None = None
    protected: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PredictionRow:
    timestamp: float | None
    packet_index: int
    label_mode: str
    predicted_class: str | None
    confidence: float | None
    ground_truth: str | None
    source_mac: str | None
    destination_mac: str | None
    features_json: str
    probabilities_json: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RollingPresenceRow:
    window_start: float | int
    window_end: float | int
    target_id: str
    frame_count: int
    mean_probability: float | None
    max_probability: float | None
    positive_prediction_ratio: float | None
    state: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
