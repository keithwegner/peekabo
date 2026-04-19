import json

from peekaboo.config import WindowConfig
from peekaboo.inference.presence import (
    PresenceStateTracker,
    StreamingPresenceEngine,
    format_presence_event,
)


def prediction(timestamp: float, predicted_class: str = "target", probability: float = 0.9):
    return {
        "timestamp": timestamp,
        "predicted_class": predicted_class,
        "confidence": probability,
        "probabilities_json": json.dumps({"target": probability, "other": 1 - probability}),
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
