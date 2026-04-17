from peekabo.parsing.dot11 import frame_family, map_logical_addresses, normalize_mac


def test_logical_address_mapping_by_ds_bits():
    assert map_logical_addresses(False, False, "d", "s", "bssid") == ("s", "d")
    assert map_logical_addresses(True, False, "ap", "client", "server") == ("client", "server")
    assert map_logical_addresses(False, True, "client", "ap", "server") == ("server", "client")
    assert map_logical_addresses(True, True, "ra", "ta", "da", "sa") == ("sa", "da")


def test_mac_normalization_and_frame_family():
    assert normalize_mac("AA:BB:CC:DD:EE:FF") == "aa:bb:cc:dd:ee:ff"
    assert normalize_mac("00:00:00:00:00:00") is None
    assert frame_family(0) == "management"
    assert frame_family(1) == "control"
    assert frame_family(2) == "data"

