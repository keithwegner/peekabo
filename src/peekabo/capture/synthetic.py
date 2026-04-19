"""Synthetic 802.11 captures for examples and tests."""

from __future__ import annotations

from pathlib import Path

AP_MAC = "66:55:44:33:22:11"
IPHONE_MAC = "aa:bb:cc:dd:ee:ff"
LG_TV_MAC = "11:22:33:44:55:66"
UNKNOWN_MAC = "22:33:44:55:66:77"
INTERNET_MAC = "ff:ff:ff:ff:ff:ff"


def write_synthetic_capture(path: str | Path, *, packet_count: int = 120) -> Path:
    """Write a deterministic passive-safe Radiotap/Dot11 PCAP fixture."""
    try:
        from scapy.layers.dot11 import Dot11, RadioTap  # type: ignore
        from scapy.packet import Raw  # type: ignore
        from scapy.utils import wrpcap  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency availability
        raise RuntimeError("Scapy is required to generate the synthetic capture") from exc

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    packets = []
    for index in range(packet_count):
        source_mac, destination_mac, rssi, rate, subtype, payload_len = _packet_spec(index)
        packet = (
            RadioTap(
                present="Rate+Channel+dBm_AntSignal",
                Rate=rate,
                ChannelFrequency=2412,
                ChannelFlags=0x00A0,
                dBm_AntSignal=rssi,
            )
            / Dot11(
                type=2,
                subtype=subtype,
                FCfield=0x41,
                addr1=AP_MAC,
                addr2=source_mac,
                addr3=destination_mac,
                SC=index << 4,
            )
            / Raw(bytes([index + 1]) * payload_len)
        )
        packet.time = 1_700_000_000 + index
        packets.append(packet)

    wrpcap(str(output), packets)
    return output


def _packet_spec(index: int) -> tuple[str, str, int, int, int, int]:
    if index % 3 == 0:
        return (
            IPHONE_MAC,
            INTERNET_MAC,
            -38 - (index % 5),
            54 if index % 2 else 48,
            0,
            72 + (index % 4) * 4,
        )
    if index % 3 == 1:
        return (
            LG_TV_MAC,
            INTERNET_MAC,
            -63 - (index % 4),
            12 if index % 2 else 18,
            4,
            24 + (index % 3) * 3,
        )
    return (
        UNKNOWN_MAC,
        INTERNET_MAC,
        -72 - (index % 6),
        6 if index % 2 else 9,
        8,
        36 + (index % 5) * 2,
    )
