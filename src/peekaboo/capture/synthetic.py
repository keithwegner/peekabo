"""Synthetic 802.11 captures for examples and tests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

AP_MAC = "66:55:44:33:22:11"
IPHONE_MAC = "aa:bb:cc:dd:ee:ff"
LG_TV_MAC = "11:22:33:44:55:66"
UNKNOWN_MAC = "22:33:44:55:66:77"
SMART_SPEAKER_MAC = "02:33:44:55:66:88"
INTERNET_MAC = "ff:ff:ff:ff:ff:ff"
PHONE_SERVICE_MAC = "02:00:00:00:10:01"
TV_SERVICE_MAC = "02:00:00:00:20:01"
IOT_SERVICE_MAC = "02:00:00:00:30:01"
MULTICAST_MAC = "01:00:5e:00:00:fb"

_BASE_TIMESTAMP = 1_700_000_000
_STORY_PACKET_COUNT = 120
_CYCLE_SECONDS = 7_200
_CHANNEL_FLAGS = 0x00A0
_PROTECTED_TO_DS = 0x41
_RETRY = 0x08


@dataclass(frozen=True)
class _SyntheticScene:
    name: str
    start: int
    stop: int
    timestamp_offset: int


@dataclass(frozen=True)
class _SyntheticPacketSpec:
    source_mac: str
    destination_mac: str
    rssi: int
    rate: int
    subtype: int
    payload_len: int
    channel_frequency: int
    timestamp: float
    retry: bool = False


_SCENES = (
    _SyntheticScene("phone_arrival", 0, 18, 0),
    _SyntheticScene("quiet_house", 18, 36, 240),
    _SyntheticScene("phone_browse_burst", 36, 64, 900),
    _SyntheticScene("tv_streaming", 64, 84, 1_800),
    _SyntheticScene("edge_of_house", 84, 100, 3_600),
    _SyntheticScene("evening_return", 100, 120, 5_400),
)


def write_synthetic_capture(path: str | Path, *, packet_count: int = 120) -> Path:
    """Write a deterministic passive-safe Radiotap/Dot11 PCAP fixture.

    The first 120 packets form one complete fake household traffic story. Larger
    captures repeat that story with increasing timestamps so tests and examples
    can scale without changing the shape of the demo.
    """
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
        spec = _packet_spec(index)
        fcfield = _PROTECTED_TO_DS | (_RETRY if spec.retry else 0)
        packet = (
            RadioTap(
                present="Rate+Channel+dBm_AntSignal",
                Rate=spec.rate,
                ChannelFrequency=spec.channel_frequency,
                ChannelFlags=_CHANNEL_FLAGS,
                dBm_AntSignal=spec.rssi,
            )
            / Dot11(
                type=2,
                subtype=spec.subtype,
                FCfield=fcfield,
                addr1=AP_MAC,
                addr2=spec.source_mac,
                addr3=spec.destination_mac,
                SC=index << 4,
            )
            / Raw(bytes([_payload_byte(index)]) * spec.payload_len)
        )
        packet.time = spec.timestamp
        packets.append(packet)

    wrpcap(str(output), packets)
    return output


def _packet_spec(index: int) -> _SyntheticPacketSpec:
    scene, local_index, cycle = _scene_for_index(index)
    timestamp = _timestamp(scene, local_index, cycle)

    if scene.name == "phone_arrival":
        if local_index % 4 in {0, 1}:
            return _iphone_frame(
                timestamp,
                rssi=-72 + min(local_index * 2, 34),
                rate=24 if local_index < 5 else (48 if local_index % 2 else 54),
                subtype=8 if local_index % 3 else 0,
                payload_len=68 + (local_index % 5) * 12,
                channel_frequency=2412,
                retry=local_index < 4,
            )
        if local_index % 4 == 2:
            return _tv_frame(
                timestamp,
                rssi=-62 - (local_index % 3),
                rate=18,
                subtype=4,
                payload_len=28 + (local_index % 2) * 8,
                channel_frequency=2412,
            )
        return _background_frame(
            timestamp,
            source_mac=UNKNOWN_MAC,
            rssi=-78 - (local_index % 4),
            rate=6,
            subtype=8,
            payload_len=34 + (local_index % 3) * 4,
            channel_frequency=2412,
        )

    if scene.name == "quiet_house":
        if local_index % 6 == 0:
            return _iphone_frame(
                timestamp,
                rssi=-50 - (local_index % 5),
                rate=36,
                subtype=4,
                payload_len=24,
                channel_frequency=2437,
            )
        if local_index % 3 == 0:
            return _tv_frame(
                timestamp,
                rssi=-64 - (local_index % 4),
                rate=12,
                subtype=4,
                payload_len=26,
                channel_frequency=2437,
            )
        return _background_frame(
            timestamp,
            source_mac=SMART_SPEAKER_MAC if local_index % 2 else UNKNOWN_MAC,
            rssi=-69 - (local_index % 7),
            rate=9 if local_index % 2 else 6,
            subtype=0,
            payload_len=30 + (local_index % 4) * 5,
            channel_frequency=2437,
            destination_mac=MULTICAST_MAC if local_index % 5 == 0 else IOT_SERVICE_MAC,
        )

    if scene.name == "phone_browse_burst":
        if local_index % 5 in {0, 1, 2, 3}:
            return _iphone_frame(
                timestamp,
                rssi=-39 - (local_index % 6),
                rate=54 if local_index % 3 else 48,
                subtype=8,
                payload_len=118 + (local_index % 7) * 18,
                channel_frequency=2412 if local_index < 18 else 2437,
                retry=local_index in {9, 10, 21},
            )
        return _background_frame(
            timestamp,
            source_mac=UNKNOWN_MAC,
            rssi=-74 - (local_index % 5),
            rate=12,
            subtype=0,
            payload_len=44 + (local_index % 4) * 7,
            channel_frequency=2412,
        )

    if scene.name == "tv_streaming":
        if local_index % 5 in {0, 1, 2, 3}:
            return _tv_frame(
                timestamp,
                rssi=-58 - (local_index % 7),
                rate=18 if local_index % 2 else 24,
                subtype=8,
                payload_len=150 + (local_index % 6) * 22,
                channel_frequency=2462,
                retry=local_index in {6, 7, 13},
            )
        return _iphone_frame(
            timestamp,
            rssi=-48 - (local_index % 3),
            rate=36,
            subtype=4,
            payload_len=28 + (local_index % 3) * 6,
            channel_frequency=2462,
        )

    if scene.name == "edge_of_house":
        if local_index % 2 == 0:
            return _iphone_frame(
                timestamp,
                rssi=-76 - (local_index % 8),
                rate=12 if local_index % 4 else 9,
                subtype=0 if local_index % 3 else 8,
                payload_len=54 + (local_index % 5) * 8,
                channel_frequency=2462,
                retry=True,
            )
        return _background_frame(
            timestamp,
            source_mac=UNKNOWN_MAC,
            rssi=-82 - (local_index % 5),
            rate=6,
            subtype=8,
            payload_len=38 + (local_index % 4) * 6,
            channel_frequency=2462,
            retry=local_index % 3 == 1,
        )

    if local_index % 5 in {0, 1}:
        return _iphone_frame(
            timestamp,
            rssi=-44 - (local_index % 5),
            rate=54,
            subtype=8,
            payload_len=96 + (local_index % 6) * 20,
            channel_frequency=2437 if local_index < 10 else 2412,
        )
    if local_index % 5 in {2, 3}:
        return _tv_frame(
            timestamp,
            rssi=-61 - (local_index % 4),
            rate=18,
            subtype=8 if local_index % 2 else 0,
            payload_len=104 + (local_index % 5) * 16,
            channel_frequency=2437,
        )
    return _background_frame(
        timestamp,
        source_mac=SMART_SPEAKER_MAC,
        rssi=-70 - (local_index % 6),
        rate=9,
        subtype=0,
        payload_len=42 + (local_index % 3) * 10,
        channel_frequency=2437,
        destination_mac=IOT_SERVICE_MAC,
    )


def _scene_for_index(index: int) -> tuple[_SyntheticScene, int, int]:
    story_index = index % _STORY_PACKET_COUNT
    cycle = index // _STORY_PACKET_COUNT
    for scene in _SCENES:
        if scene.start <= story_index < scene.stop:
            return scene, story_index - scene.start, cycle
    last_scene = _SCENES[-1]
    return last_scene, last_scene.stop - last_scene.start - 1, cycle


def _timestamp(scene: _SyntheticScene, local_index: int, cycle: int) -> float:
    burst_jitter = 0.35 if local_index % 4 == 0 else 0.0
    return (
        _BASE_TIMESTAMP
        + (cycle * _CYCLE_SECONDS)
        + scene.timestamp_offset
        + local_index * 2.0
        + burst_jitter
    )


def _iphone_frame(
    timestamp: float,
    *,
    rssi: int,
    rate: int,
    subtype: int,
    payload_len: int,
    channel_frequency: int,
    retry: bool = False,
) -> _SyntheticPacketSpec:
    return _SyntheticPacketSpec(
        source_mac=IPHONE_MAC,
        destination_mac=PHONE_SERVICE_MAC,
        rssi=rssi,
        rate=rate,
        subtype=subtype,
        payload_len=payload_len,
        channel_frequency=channel_frequency,
        timestamp=timestamp,
        retry=retry,
    )


def _tv_frame(
    timestamp: float,
    *,
    rssi: int,
    rate: int,
    subtype: int,
    payload_len: int,
    channel_frequency: int,
    retry: bool = False,
) -> _SyntheticPacketSpec:
    return _SyntheticPacketSpec(
        source_mac=LG_TV_MAC,
        destination_mac=TV_SERVICE_MAC,
        rssi=rssi,
        rate=rate,
        subtype=subtype,
        payload_len=payload_len,
        channel_frequency=channel_frequency,
        timestamp=timestamp,
        retry=retry,
    )


def _background_frame(
    timestamp: float,
    *,
    source_mac: str,
    rssi: int,
    rate: int,
    subtype: int,
    payload_len: int,
    channel_frequency: int,
    destination_mac: str = INTERNET_MAC,
    retry: bool = False,
) -> _SyntheticPacketSpec:
    return _SyntheticPacketSpec(
        source_mac=source_mac,
        destination_mac=destination_mac,
        rssi=rssi,
        rate=rate,
        subtype=subtype,
        payload_len=payload_len,
        channel_frequency=channel_frequency,
        timestamp=timestamp,
        retry=retry,
    )


def _payload_byte(index: int) -> int:
    return (index % 251) + 1
