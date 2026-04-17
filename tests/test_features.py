from peekabo.config import FeatureConfig
from peekabo.features.extract import model_feature_names, record_to_feature_row, row_to_model_features
from peekabo.parsing.records import PacketRecord


def test_feature_extraction_data_size_fallback_and_signal_choice():
    record = PacketRecord(
        timestamp=1.0,
        source_file="x.pcap",
        packet_index=7,
        packet_len=128,
        dot11_frame_len=None,
        capture_len=120,
        payload_len=42,
        rssi=-55,
        source_mac="aa:bb:cc:dd:ee:ff",
        destination_mac="11:22:33:44:55:66",
    )
    feature = record_to_feature_row(record, FeatureConfig()).to_dict()
    assert feature["data_size"] == 120
    assert feature["ssi"] == -55.0


def test_default_model_features_drop_mac_addresses():
    config = FeatureConfig(leakage_debug=False)
    row = {
        "hour": 1,
        "data_rate": 54.0,
        "ssi": -50,
        "frame_type": 2,
        "frame_subtype": 8,
        "data_size": 200,
        "source_mac": "aa:bb:cc:dd:ee:ff",
        "destination_mac": "11:22:33:44:55:66",
    }
    features = row_to_model_features(row, config)
    assert "source_mac" not in features
    assert "destination_mac" not in features
    assert model_feature_names(config) == [
        "hour",
        "data_rate",
        "ssi",
        "frame_type",
        "frame_subtype",
        "data_size",
    ]


def test_leakage_debug_adds_hashed_mac_features():
    config = FeatureConfig(leakage_debug=True, mac_encoding="hashed")
    features = row_to_model_features(
        {"source_mac": "aa:bb:cc:dd:ee:ff", "destination_mac": "11:22:33:44:55:66"},
        config,
    )
    assert "source_mac_hash" in features
    assert features["source_mac_hash"] != "aa:bb:cc:dd:ee:ff"

