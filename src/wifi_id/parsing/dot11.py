"""802.11 MAC-header extraction and address-role mapping."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from wifi_id.parsing.records import PacketRecord


FRAME_FAMILY_BY_TYPE = {
    0: "management",
    1: "control",
    2: "data",
    3: "extension",
}


def normalize_mac(mac: str | None) -> str | None:
    if mac is None:
        return None
    value = str(mac).strip().lower()
    if not value or value in {"none", "00:00:00:00:00:00"}:
        return None
    return value


def map_logical_addresses(
    to_ds: bool,
    from_ds: bool,
    addr1: str | None,
    addr2: str | None,
    addr3: str | None,
    addr4: str | None = None,
) -> tuple[str | None, str | None]:
    """Map physical 802.11 address fields to logical source and destination."""
    a1 = normalize_mac(addr1)
    a2 = normalize_mac(addr2)
    a3 = normalize_mac(addr3)
    a4 = normalize_mac(addr4)

    if not to_ds and not from_ds:
        return a2, a1
    if to_ds and not from_ds:
        return a2, a3
    if not to_ds and from_ds:
        return a3, a1
    return a4, a3


def frame_family(frame_type: int | None) -> str | None:
    if frame_type is None:
        return None
    return FRAME_FAMILY_BY_TYPE.get(int(frame_type), "unknown")


def _packet_len(packet: Any) -> int | None:
    try:
        return len(bytes(packet))
    except Exception:
        return None


def _hour_from_timestamp(timestamp: float | None) -> int | None:
    if timestamp is None:
        return None
    return datetime.fromtimestamp(float(timestamp)).hour


def _flag_int(value: Any) -> int:
    try:
        return int(value or 0)
    except Exception:
        return 0


def extract_dot11_fields(packet: Any) -> dict[str, Any]:
    """Extract Dot11 fields from a Scapy packet.

    The import is intentionally local so non-packet utilities can be imported
    without Scapy installed.
    """
    try:
        from scapy.layers.dot11 import Dot11  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency availability
        raise RuntimeError("Scapy is required for Dot11 parsing") from exc

    if not packet.haslayer(Dot11):
        raise ValueError("packet has no 802.11 Dot11 layer")

    dot11 = packet[Dot11]
    fc = _flag_int(getattr(dot11, "FCfield", 0))
    to_ds = bool(fc & 0x01)
    from_ds = bool(fc & 0x02)
    retry = bool(fc & 0x08)
    protected = bool(fc & 0x40)
    addr1 = normalize_mac(getattr(dot11, "addr1", None))
    addr2 = normalize_mac(getattr(dot11, "addr2", None))
    addr3 = normalize_mac(getattr(dot11, "addr3", None))
    addr4 = normalize_mac(getattr(dot11, "addr4", None))
    source, destination = map_logical_addresses(to_ds, from_ds, addr1, addr2, addr3, addr4)
    sc = getattr(dot11, "SC", None)
    sequence_number = None if sc is None else int(sc) >> 4

    try:
        payload_len = len(bytes(dot11.payload))
    except Exception:
        payload_len = None

    return {
        "frame_type": getattr(dot11, "type", None),
        "frame_subtype": getattr(dot11, "subtype", None),
        "to_ds": to_ds,
        "from_ds": from_ds,
        "addr1": addr1,
        "addr2": addr2,
        "addr3": addr3,
        "addr4": addr4,
        "source_mac": source,
        "destination_mac": destination,
        "sequence_number": sequence_number,
        "retry": retry,
        "protected": protected,
        "payload_len": payload_len,
    }


def parse_packet_to_record(
    packet: Any,
    source_file: str,
    packet_index: int,
    radiotap_fields: dict[str, Any],
) -> PacketRecord:
    timestamp = getattr(packet, "time", None)
    timestamp_value = None if timestamp is None else float(timestamp)
    packet_len = _packet_len(packet)
    capture_len = len(getattr(packet, "original", b"") or b"") or packet_len
    original_len = getattr(packet, "wirelen", None) or packet_len
    radiotap_len = radiotap_fields.get("radiotap_len")

    try:
        dot11_fields = extract_dot11_fields(packet)
    except Exception as exc:
        return PacketRecord(
            timestamp=timestamp_value,
            source_file=source_file,
            packet_index=packet_index,
            packet_len=packet_len,
            capture_len=capture_len,
            original_len=original_len,
            radiotap_len=radiotap_len,
            hour=_hour_from_timestamp(timestamp_value),
            data_rate=radiotap_fields.get("data_rate"),
            channel_frequency=radiotap_fields.get("channel_frequency"),
            ssi=radiotap_fields.get("ssi"),
            rssi=radiotap_fields.get("rssi"),
            parse_ok=False,
            parse_error=str(exc),
        )

    dot11_frame_len = None
    if packet_len is not None and radiotap_len is not None:
        dot11_frame_len = max(int(packet_len) - int(radiotap_len), 0)

    return PacketRecord(
        timestamp=timestamp_value,
        source_file=source_file,
        packet_index=packet_index,
        packet_len=packet_len,
        capture_len=capture_len,
        original_len=original_len,
        radiotap_len=radiotap_len,
        dot11_frame_len=dot11_frame_len,
        payload_len=dot11_fields.get("payload_len"),
        hour=_hour_from_timestamp(timestamp_value),
        data_rate=radiotap_fields.get("data_rate"),
        channel_frequency=radiotap_fields.get("channel_frequency"),
        ssi=radiotap_fields.get("ssi"),
        rssi=radiotap_fields.get("rssi"),
        frame_type=dot11_fields.get("frame_type"),
        frame_subtype=dot11_fields.get("frame_subtype"),
        to_ds=dot11_fields.get("to_ds"),
        from_ds=dot11_fields.get("from_ds"),
        addr1=dot11_fields.get("addr1"),
        addr2=dot11_fields.get("addr2"),
        addr3=dot11_fields.get("addr3"),
        addr4=dot11_fields.get("addr4"),
        source_mac=dot11_fields.get("source_mac"),
        destination_mac=dot11_fields.get("destination_mac"),
        sequence_number=dot11_fields.get("sequence_number"),
        retry=dot11_fields.get("retry"),
        protected=dot11_fields.get("protected"),
    )

