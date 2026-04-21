import json

import pytest

from peekaboo.config import AppConfig, LabelConfig, PresenceConfig, WindowConfig
from peekaboo.inference.presence import (
    MultiTargetStreamingPresenceEngine,
    PresenceStateTracker,
    StreamingPresenceEngine,
    format_presence_event,
    resolve_presence_target_classes,
)
from peekaboo.labeling.targets import TargetRegistry


def prediction(timestamp: float, predicted_class: str = "target", probability: float = 0.9):
    return {
        "timestamp": timestamp,
        "predicted_class": predicted_class,
        "confidence": probability,
        "probabilities_json": json.dumps({"target": probability, "other": 1 - probability}),
    }


def multiclass_prediction(timestamp: float, predicted_class: str = "iphone_5_user1"):
    return {
        "timestamp": timestamp,
        "predicted_class": predicted_class,
        "confidence": 0.9,
        "probabilities_json": json.dumps(
            {"iphone_5_user1": 0.9, "lg_tv": 0.1, "other": 0.0}
        ),
    }


def test_streaming_presence_emits_frame_windows_and_flushes_partial():
    engine = StreamingPresenceEngine(
        target_class="target",
        config=WindowConfig(frame_count=3, min_frames=1),
    )

    assert engine.process(prediction(1)) == []
    assert engine.process(prediction(2)) == []
    events = engine.process(prediction(3))

    assert len(events) == 1
    assert events[0]["window_type"] == "frame_count"
    assert events[0]["window_start"] == 0
    assert events[0]["window_end"] == 2
    assert events[0]["state"] == "present"
    assert not events[0]["final_window"]

    final_events = engine.flush()
    assert len(final_events) == 1
    assert final_events[0]["window_type"] == "time"
    assert final_events[0]["final_window"]


def test_streaming_presence_emits_time_windows_when_bucket_advances():
    engine = StreamingPresenceEngine(
        target_class="target",
        config=WindowConfig(frame_count=100, time_seconds=10, min_frames=1),
    )

    assert engine.process(prediction(1)) == []
    assert engine.process(prediction(8)) == []
    events = engine.process(prediction(12))

    assert len(events) == 1
    assert events[0]["window_type"] == "time"
    assert events[0]["window_start"] == 0
    assert events[0]["window_end"] == 10
    assert not events[0]["final_window"]

    final_events = engine.flush()
    assert len(final_events) == 2
    assert {event["window_type"] for event in final_events} == {"frame_count", "time"}
    assert all(event["final_window"] for event in final_events)


def test_presence_state_tracker_reports_changes_without_duplicates():
    tracker = PresenceStateTracker()
    present_event = {"target_id": "target", "window_type": "frame_count", "state": "present"}
    duplicate_event = {"target_id": "target", "window_type": "frame_count", "state": "present"}
    absent_event = {"target_id": "target", "window_type": "frame_count", "state": "absent"}

    assert tracker.is_change(present_event)
    assert not tracker.is_change(duplicate_event)
    assert tracker.is_change(absent_event)
    assert "state=absent" in format_presence_event(absent_event)


def test_presence_target_resolution_precedence_and_all_targets():
    registry = TargetRegistry.from_dict(
        {
            "targets": [
                {
                    "target_id": "iphone_5_user1",
                    "mac_addresses": ["aa:bb:cc:dd:ee:ff"],
                    "enabled": True,
                },
                {
                    "target_id": "lg_tv",
                    "mac_addresses": ["11:22:33:44:55:66"],
                    "enabled": True,
                },
            ]
        }
    )
    config = AppConfig(
        labeling=LabelConfig(mode="multiclass_with_other", target_id="configured"),
        presence=PresenceConfig(target_classes=["from_config"], all_targets=True),
    )

    assert resolve_presence_target_classes(
        config,
        target_classes=["cli", "cli", "second"],
        all_targets=True,
        registry=registry,
    ) == ["cli", "second"]
    assert resolve_presence_target_classes(
        config,
        target_classes=[],
        all_targets=True,
        registry=registry,
    ) == ["iphone_5_user1", "lg_tv"]
    assert resolve_presence_target_classes(
        config,
        target_classes=[],
        all_targets=False,
        registry=registry,
    ) == ["configured"]
    assert resolve_presence_target_classes(
        config,
        target_classes=None,
        all_targets=None,
        registry=registry,
    ) == ["from_config"]


def test_all_targets_requires_registry_and_multiclass_mode():
    with pytest.raises(ValueError, match="multiclass label mode"):
        resolve_presence_target_classes(
            AppConfig(labeling=LabelConfig(mode="binary_one_vs_rest")),
            all_targets=True,
        )

    with pytest.raises(ValueError, match="target_registry_path"):
        resolve_presence_target_classes(
            AppConfig(labeling=LabelConfig(mode="multiclass_with_other")),
            all_targets=True,
        )


def test_multi_target_streaming_presence_emits_and_flushes_each_target():
    engine = MultiTargetStreamingPresenceEngine(
        target_classes=["iphone_5_user1", "lg_tv"],
        config=WindowConfig(frame_count=2, time_seconds=10, min_frames=1),
    )

    assert engine.process(multiclass_prediction(1, "iphone_5_user1")) == []
    events = engine.process(multiclass_prediction(2, "lg_tv"))
    final_events = engine.flush()

    assert {event["target_id"] for event in events} == {"iphone_5_user1", "lg_tv"}
    assert {event["window_type"] for event in events} == {"frame_count"}
    assert {event["target_id"] for event in final_events} == {"iphone_5_user1", "lg_tv"}
    assert {event["window_type"] for event in final_events} == {"time"}
    assert all(event["final_window"] for event in final_events)


def test_presence_state_tracker_tracks_changes_per_target():
    tracker = PresenceStateTracker()
    iphone_event = {"target_id": "iphone_5_user1", "window_type": "frame_count", "state": "present"}
    lg_event = {"target_id": "lg_tv", "window_type": "frame_count", "state": "present"}

    assert tracker.is_change(iphone_event)
    assert not tracker.is_change(iphone_event)
    assert tracker.is_change(lg_event)
