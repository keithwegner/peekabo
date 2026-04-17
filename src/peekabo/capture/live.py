"""Passive live monitor-mode packet source."""

from __future__ import annotations

from typing import Iterator

from peekabo.parsing.dot11 import parse_packet_to_record
from peekabo.parsing.radiotap import extract_radiotap_fields
from peekabo.parsing.records import PacketRecord


def iter_live_records(
    interface: str,
    *,
    timeout_seconds: float | None = None,
    max_packets: int | None = None,
) -> Iterator[PacketRecord]:
    """Yield live packets from an already-configured monitor-mode interface.

    This is intentionally passive. It does not configure the interface, hop
    channels, inject frames, or transmit probe traffic.
    """
    try:
        from scapy.sendrecv import sniff  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency availability
        raise RuntimeError("Scapy is required for live classification") from exc

    index = 0
    while max_packets is None or index < max_packets:
        packets = sniff(iface=interface, count=1, timeout=timeout_seconds, store=True)
        if not packets:
            break
        packet = packets[0]
        radiotap = extract_radiotap_fields(packet)
        yield parse_packet_to_record(packet, f"live:{interface}", index, radiotap)
        index += 1

