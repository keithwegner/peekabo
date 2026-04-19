from peekaboo.capture.inspect import summarize_packet_records
from peekaboo.labeling.targets import TargetRegistry


def test_inspection_summarizes_successful_records_and_target_matches():
    registry = TargetRegistry.from_dict(
        {
            "targets": [
                {
                    "target_id": "phone",
                    "label": "phone",
                    "mac_addresses": ["aa:bb:cc:dd:ee:ff"],
                }
            ]
        }
    )
    rows = [
        {
            "parse_ok": True,
            "frame_type": 2,
            "protected": True,
            "data_rate": 54.0,
            "channel_frequency": 2412,
            "rssi": -42,
            "source_mac": "aa:bb:cc:dd:ee:ff",
            "destination_mac": "ff:ff:ff:ff:ff:ff",
        },
        {
            "parse_ok": True,
            "frame_type": 0,
            "protected": False,
            "data_rate": 24.0,
            "channel_frequency": 2412,
            "ssi": -60,
            "source_mac": "11:22:33:44:55:66",
            "destination_mac": "ff:ff:ff:ff:ff:ff",
        },
    ]

    summary = summarize_packet_records(rows, capture_paths=["sample.pcap"], registry=registry)

    assert summary["capture_file_count"] == 1
    assert summary["packets_scanned"] == 2
    assert summary["parse_successes"] == 2
    assert summary["parse_failures"] == 0
    assert summary["frame_family_counts"]["data"] == 1
    assert summary["frame_family_counts"]["management"] == 1
    assert summary["protected_frame_count"] == 1
    assert summary["radiotap_coverage"]["data_rate"]["coverage"] == 1.0
    assert summary["radiotap_coverage"]["rssi_or_ssi"]["coverage"] == 1.0
    assert summary["channel_frequencies"] == [2412]
    assert summary["source_mac_count"] == 2
    assert summary["destination_mac_count"] == 1
    assert summary["target_match_counts"] == {"phone": 1}
    assert summary["warnings"] == []


def test_inspection_warns_for_missing_fields_parse_failures_and_no_targets():
    registry = TargetRegistry.from_dict(
        {
            "targets": [
                {
                    "target_id": "phone",
                    "label": "phone",
                    "mac_addresses": ["aa:bb:cc:dd:ee:ff"],
                }
            ]
        }
    )
    rows = [
        {"parse_ok": False, "parse_error": "packet has no 802.11 Dot11 layer"},
        {"parse_ok": True, "frame_type": 2, "protected": False, "source_mac": "00:11:22:33:44:55"},
    ]

    summary = summarize_packet_records(rows, capture_paths=["sample.pcap"], registry=registry)

    assert summary["packets_scanned"] == 2
    assert summary["parse_failure_rate"] == 0.5
    assert summary["radiotap_coverage"]["data_rate"]["coverage"] == 0.0
    assert summary["target_match_total"] == 0
    assert any("20%" in warning for warning in summary["warnings"])
    assert any("data-rate" in warning for warning in summary["warnings"])
    assert any("RSSI/SSI" in warning for warning in summary["warnings"])
    assert any("No protected" in warning for warning in summary["warnings"])
    assert any("No source MACs matched" in warning for warning in summary["warnings"])


def test_inspection_honors_max_packets():
    rows = [{"parse_ok": True, "frame_type": 2, "protected": True} for _ in range(5)]

    summary = summarize_packet_records(rows, capture_paths=["sample.pcap"], max_packets=2)

    assert summary["packets_scanned"] == 2
    assert summary["dot11_frame_count"] == 2
