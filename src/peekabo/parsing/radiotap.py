"""Radiotap metadata extraction."""

from __future__ import annotations

from typing import Any


def extract_radiotap_fields_from_layer(layer: Any) -> dict[str, Any]:
    """Extract the small Radiotap subset used by the application."""
    data_rate = getattr(layer, "Rate", None)
    channel_frequency = getattr(layer, "ChannelFrequency", None)
    antenna_signal = getattr(layer, "dBm_AntSignal", None)
    db_antenna_signal = getattr(layer, "dB_AntSignal", None)
    radiotap_len = getattr(layer, "len", None)
    return {
        "radiotap_len": None if radiotap_len is None else int(radiotap_len),
        "data_rate": None if data_rate is None else float(data_rate),
        "channel_frequency": None if channel_frequency is None else int(channel_frequency),
        "ssi": None if db_antenna_signal is None else float(db_antenna_signal),
        "rssi": None if antenna_signal is None else float(antenna_signal),
    }


def extract_radiotap_fields(packet: Any) -> dict[str, Any]:
    try:
        from scapy.layers.dot11 import RadioTap  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency availability
        raise RuntimeError("Scapy is required for Radiotap parsing") from exc

    defaults = {
        "radiotap_len": None,
        "data_rate": None,
        "channel_frequency": None,
        "ssi": None,
        "rssi": None,
    }
    if not packet.haslayer(RadioTap):
        return defaults
    return defaults | extract_radiotap_fields_from_layer(packet[RadioTap])
