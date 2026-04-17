from wifi_id.parsing.radiotap import extract_radiotap_fields_from_layer


class FakeRadioTap:
    len = 18
    Rate = 54
    ChannelFrequency = 2412
    dBm_AntSignal = -48
    dB_AntSignal = 32


def test_extract_radiotap_fields_from_layer():
    fields = extract_radiotap_fields_from_layer(FakeRadioTap())
    assert fields["radiotap_len"] == 18
    assert fields["data_rate"] == 54.0
    assert fields["channel_frequency"] == 2412
    assert fields["rssi"] == -48.0
    assert fields["ssi"] == 32.0

