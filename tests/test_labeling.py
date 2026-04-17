from peekabo.config import FilterConfig, LabelConfig
from peekabo.labeling.filters import is_ap_originated, passes_filters
from peekabo.labeling.labelers import label_for_row
from peekabo.labeling.targets import TargetRegistry


def registry():
    return TargetRegistry.from_dict(
        {
            "targets": [
                {
                    "target_id": "phone",
                    "label": "phone",
                    "mac_addresses": ["aa:bb:cc:dd:ee:ff", "aa:bb:cc:dd:ee:00"],
                    "enabled": True,
                }
            ]
        }
    )


def test_binary_and_multiclass_label_modes():
    reg = registry()
    row = {"source_mac": "aa:bb:cc:dd:ee:00"}
    assert label_for_row(row, reg, LabelConfig(mode="binary_one_vs_rest", target_id="phone")) == "target"
    assert label_for_row(row, reg, LabelConfig(mode="multiclass_known_targets_only")) == "phone"
    assert label_for_row({"source_mac": "00:11:22:33:44:55"}, reg, LabelConfig(mode="multiclass_known_targets_only")) is None
    assert label_for_row({"source_mac": "00:11:22:33:44:55"}, reg, LabelConfig(mode="multiclass_with_other")) == "other"


def test_ap_filter_heuristic_can_exclude_ap_originated_rows():
    config = FilterConfig(include_ap_originated=False)
    row = {"from_ds": True, "to_ds": False, "frame_type": 2}
    assert is_ap_originated(row, config)
    assert not passes_filters(row, config)

