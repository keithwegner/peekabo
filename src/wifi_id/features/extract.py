"""Paper-based feature extraction."""

from __future__ import annotations

import hashlib
from typing import Any, Iterable, Iterator

from wifi_id.config import FeatureConfig
from wifi_id.parsing.records import FeatureRow, PacketRecord

PAPER_FEATURES = [
    "hour",
    "data_rate",
    "ssi",
    "frame_type",
    "frame_subtype",
    "source_mac",
    "destination_mac",
    "data_size",
]

DEFAULT_MODEL_FEATURES = [
    "hour",
    "data_rate",
    "ssi",
    "frame_type",
    "frame_subtype",
    "data_size",
]

LEAKAGE_MODEL_FEATURES = ["source_mac", "destination_mac"]


def choose_signal_value(row: dict[str, Any]) -> float | None:
    value = row.get("ssi")
    if value is None:
        value = row.get("rssi")
    return None if value is None else float(value)


def choose_data_size(row: dict[str, Any], mode: str = "dot11_frame_len") -> int | None:
    candidates: list[str]
    if mode == "dot11_frame_len":
        candidates = ["dot11_frame_len", "capture_len", "packet_len", "total_packet_len"]
    elif mode == "payload_len":
        candidates = ["payload_len", "dot11_frame_len", "capture_len", "packet_len"]
    elif mode in {"packet_len", "total_packet_len"}:
        candidates = ["packet_len", "total_packet_len", "capture_len"]
    elif mode == "capture_len":
        candidates = ["capture_len", "packet_len", "total_packet_len"]
    else:
        candidates = [mode, "dot11_frame_len", "capture_len", "packet_len", "total_packet_len"]

    for key in candidates:
        value = row.get(key)
        if value is not None:
            return int(value)
    return None


def record_to_feature_row(record: PacketRecord | dict[str, Any], config: FeatureConfig) -> FeatureRow:
    row = record.to_dict() if isinstance(record, PacketRecord) else dict(record)
    return FeatureRow(
        timestamp=row.get("timestamp"),
        source_file=str(row.get("source_file") or ""),
        packet_index=int(row.get("packet_index") or 0),
        hour=row.get("hour"),
        data_rate=row.get("data_rate"),
        ssi=choose_signal_value(row),
        frame_type=row.get("frame_type"),
        frame_subtype=row.get("frame_subtype"),
        source_mac=row.get("source_mac"),
        destination_mac=row.get("destination_mac"),
        data_size=choose_data_size(row, config.data_size_mode),
        total_packet_len=row.get("packet_len") or row.get("total_packet_len"),
        dot11_frame_len=row.get("dot11_frame_len"),
        payload_len=row.get("payload_len"),
        channel_frequency=row.get("channel_frequency"),
        retry=row.get("retry"),
        protected=row.get("protected"),
    )


def iter_feature_rows(
    records: Iterable[PacketRecord | dict[str, Any]],
    config: FeatureConfig,
) -> Iterator[dict[str, Any]]:
    for record in records:
        yield record_to_feature_row(record, config).to_dict()


def model_feature_names(config: FeatureConfig) -> list[str]:
    names = list(DEFAULT_MODEL_FEATURES)
    if config.leakage_debug:
        if config.mac_encoding == "hashed":
            names.extend(["source_mac_hash", "destination_mac_hash"])
        else:
            names.extend(LEAKAGE_MODEL_FEATURES)
    return names


def row_to_model_features(row: dict[str, Any], config: FeatureConfig) -> dict[str, Any]:
    features: dict[str, Any] = {}
    for name in DEFAULT_MODEL_FEATURES:
        value = row.get(name)
        if value is None:
            features[name] = config.impute_numeric if name in {"hour", "data_rate", "ssi", "data_size"} else config.impute_categorical
            if config.include_missing_indicators:
                features[f"{name}__missing"] = 1
        else:
            features[name] = value
            if config.include_missing_indicators:
                features[f"{name}__missing"] = 0

    if config.leakage_debug:
        if config.mac_encoding == "hashed":
            features["source_mac_hash"] = stable_mac_hash(row.get("source_mac"))
            features["destination_mac_hash"] = stable_mac_hash(row.get("destination_mac"))
        else:
            features["source_mac"] = row.get("source_mac") or config.impute_categorical
            features["destination_mac"] = row.get("destination_mac") or config.impute_categorical
    return features


def stable_mac_hash(mac: str | None) -> str:
    if not mac:
        return "missing"
    digest = hashlib.sha256(str(mac).lower().encode("utf-8")).hexdigest()
    return digest[:16]

