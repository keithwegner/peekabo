"""Synthetic 802.11 captures for examples and tests."""

from __future__ import annotations

from pathlib import Path

AP_MAC = "66:55:44:33:22:11"
IPHONE_MAC = "aa:bb:cc:dd:ee:ff"
LG_TV_MAC = "11:22:33:44:55:66"
UNKNOWN_MAC = "22:33:44:55:66:77"
INTERNET_MAC = "ff:ff:ff:ff:ff:ff"


def write_synthetic_capture(path: str | Path) -> Path:
    """Write a tiny passive-safe Radiotap/Dot11 PCAP fixture."""
    try:
        from scapy.layers.dot11 import Dot11, RadioTap  # type: ignore
        from scapy.packet import Raw  # type: ignore
        from scapy.utils import wrpcap  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency availability
        raise RuntimeError("Scapy is required to generate the synthetic capture") from exc

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    packet_specs = [
        (IPHONE_MAC, INTERNET_MAC, -41, 24, 12),
        (LG_TV_MAC, INTERNET_MAC, -55, 18, 24),
        (UNKNOWN_MAC, INTERNET_MAC, -67, 12, 36),
        (IPHONE_MAC, INTERNET_MAC, -44, 36, 48),
        (LG_TV_MAC, INTERNET_MAC, -53, 24, 60),
    ]
    packets = []
    for index, (source_mac, destination_mac, rssi, rate, payload_len) in enumerate(packet_specs):
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
                subtype=0,
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
