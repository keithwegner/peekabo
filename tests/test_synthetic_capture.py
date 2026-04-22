from pathlib import Path

import pytest

from peekaboo.capture.sources import iter_packet_records
from peekaboo.capture.synthetic import (
    IPHONE_MAC,
    LG_TV_MAC,
    SMART_SPEAKER_MAC,
    UNKNOWN_MAC,
    write_synthetic_capture,
)

pytest.importorskip("scapy")


def test_synthetic_capture_has_rich_metadata_variation(tmp_path: Path):
    capture_path = tmp_path / "synthetic-demo.pcap"

    write_synthetic_capture(capture_path)
    rows = [record.to_dict() for record in iter_packet_records([capture_path])]

    assert len(rows) == 120
    assert {row["source_mac"] for row in rows} == {
        IPHONE_MAC,
        LG_TV_MAC,
        SMART_SPEAKER_MAC,
        UNKNOWN_MAC,
    }
    assert len({row["destination_mac"] for row in rows}) >= 5
    assert {row["channel_frequency"] for row in rows} == {2412, 2437, 2462}
    assert len({row["data_rate"] for row in rows}) >= 8
    assert {row["frame_subtype"] for row in rows} == {0, 4, 8}
    assert len({row["hour"] for row in rows}) >= 2
    assert all(row["protected"] for row in rows)
    assert any(row["retry"] for row in rows)

    rssi_values = [float(row["rssi"]) for row in rows if row["rssi"] is not None]
    dot11_lengths = [
        int(row["dot11_frame_len"]) for row in rows if row["dot11_frame_len"] is not None
    ]
    assert max(rssi_values) - min(rssi_values) >= 40
    assert max(dot11_lengths) - min(dot11_lengths) >= 200


def test_synthetic_capture_repeats_story_for_larger_counts(tmp_path: Path):
    capture_path = tmp_path / "synthetic-long.pcap"

    write_synthetic_capture(capture_path, packet_count=260)
    rows = [record.to_dict() for record in iter_packet_records([capture_path])]

    assert len(rows) == 260
    assert rows[-1]["timestamp"] > rows[0]["timestamp"]
    assert all(row["parse_ok"] for row in rows)
