from pathlib import Path

import yaml
from typer.testing import CliRunner

from peekaboo.capture.synthetic import IPHONE_MAC, LG_TV_MAC, write_synthetic_capture
from peekaboo.cli import app
from peekaboo.data.readers import read_all_rows
from peekaboo.models.base import OnlineModel
from peekaboo.parsing.records import PacketRecord


class AlwaysTargetEstimator:
    def predict_proba_one(self, x):
        return {"target": 1.0, "other": 0.0}

    def predict_one(self, x):
        return "target"

    def learn_one(self, x, y):
        return None


class AlwaysIphoneEstimator:
    def predict_proba_one(self, x):
        return {"iphone_5_user1": 0.9, "lg_tv": 0.1, "other": 0.0}

    def predict_one(self, x):
        return "iphone_5_user1"

    def learn_one(self, x, y):
        return None


def test_presence_replay_outputs_jsonl_and_honors_max_packets(tmp_path: Path):
    capture_path = tmp_path / "synthetic-demo.pcap"
    config_path = tmp_path / "config.yaml"
    output_dir = tmp_path / "runs"
    write_synthetic_capture(capture_path)
    _write_config(config_path, capture_path, output_dir)

    runner = CliRunner()
    for command in ["ingest", "features", "label", "train-online"]:
        result = runner.invoke(app, [command, "--config", str(config_path)])
        assert result.exit_code == 0, result.output

    result = runner.invoke(
        app,
        [
            "presence-replay",
            "--config",
            str(config_path),
            "--max-packets",
            "25",
            "--quiet",
        ],
    )

    assert result.exit_code == 0, result.output
    predictions = read_all_rows(output_dir / "replay_predictions.jsonl")
    presence = read_all_rows(output_dir / "replay_presence.jsonl")
    assert len(predictions) == 25
    assert presence
    assert {row["state"] for row in presence} <= {"present", "absent", "uncertain"}
    assert {row["target_id"] for row in presence} == {"target"}
    assert "source_mac" not in predictions[0]["features_json"]
    assert "destination_mac" not in predictions[0]["features_json"]


def test_multitarget_presence_replay_and_classify_file_outputs_all_targets(tmp_path: Path):
    capture_path = tmp_path / "synthetic-demo.pcap"
    config_path = tmp_path / "config.yaml"
    output_dir = tmp_path / "runs"
    write_synthetic_capture(capture_path)
    _write_multitarget_config(config_path, capture_path, output_dir)

    runner = CliRunner()
    for command in ["ingest", "features", "label", "train-online"]:
        result = runner.invoke(app, [command, "--config", str(config_path)])
        assert result.exit_code == 0, result.output

    replay_result = runner.invoke(
        app,
        [
            "presence-replay",
            "--config",
            str(config_path),
            "--all-targets",
            "--max-packets",
            "30",
            "--quiet",
        ],
    )
    classify_result = runner.invoke(
        app,
        ["classify-file", "--config", str(config_path), "--all-targets"],
    )

    assert replay_result.exit_code == 0, replay_result.output
    assert classify_result.exit_code == 0, classify_result.output
    predictions = read_all_rows(output_dir / "replay_predictions.jsonl")
    presence = read_all_rows(output_dir / "replay_presence.jsonl")
    rolling = read_all_rows(output_dir / "rolling.parquet")
    assert len(predictions) == 30
    assert {row["target_id"] for row in presence} == {"iphone_5_user1", "lg_tv"}
    assert {row["target_id"] for row in rolling} == {"iphone_5_user1", "lg_tv"}
    assert "source_mac" not in predictions[0]["features_json"]
    assert "destination_mac" not in predictions[0]["features_json"]


def test_presence_replay_repeat_target_class_limits_output(tmp_path: Path):
    capture_path = tmp_path / "synthetic-demo.pcap"
    config_path = tmp_path / "config.yaml"
    output_dir = tmp_path / "runs"
    write_synthetic_capture(capture_path)
    _write_multitarget_config(config_path, capture_path, output_dir)

    runner = CliRunner()
    for command in ["ingest", "features", "label", "train-online"]:
        result = runner.invoke(app, [command, "--config", str(config_path)])
        assert result.exit_code == 0, result.output

    result = runner.invoke(
        app,
        [
            "presence-replay",
            "--config",
            str(config_path),
            "--target-class",
            "lg_tv",
            "--max-packets",
            "20",
            "--predictions-output",
            str(output_dir / "subset_predictions.jsonl"),
            "--presence-output",
            str(output_dir / "subset_presence.jsonl"),
            "--quiet",
        ],
    )

    assert result.exit_code == 0, result.output
    assert {row["target_id"] for row in read_all_rows(output_dir / "subset_presence.jsonl")} == {
        "lg_tv"
    }


def test_presence_replay_all_targets_rejects_binary_label_mode(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    _write_config(config_path, tmp_path / "missing.pcap", tmp_path / "runs")

    result = CliRunner().invoke(
        app,
        ["presence-replay", "--config", str(config_path), "--all-targets"],
    )

    assert result.exit_code != 0
    assert "multiclass label mode" in result.output


def test_presence_live_requires_interface(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    _write_config(config_path, tmp_path / "missing.pcap", tmp_path / "runs")

    result = CliRunner().invoke(app, ["presence-live", "--config", str(config_path)])

    assert result.exit_code != 0
    assert "monitor-mode interface" in result.output


def test_presence_live_uses_passive_live_records_without_hardware(tmp_path: Path, monkeypatch):
    output_dir = tmp_path / "runs"
    config_path = tmp_path / "config.yaml"
    checkpoint_path = output_dir / "model.pkl"
    _write_config(config_path, tmp_path / "missing.pcap", output_dir)
    OnlineModel("fake", AlwaysTargetEstimator(), [], {}).save(checkpoint_path)

    def fake_live_records(interface, *, timeout_seconds=None, max_packets=None):
        del interface, timeout_seconds
        for index in range(max_packets or 5):
            yield PacketRecord(
                timestamp=float(index),
                source_file="live:test0",
                packet_index=index,
                data_rate=54.0,
                rssi=-40,
                frame_type=2,
                frame_subtype=0,
                source_mac="aa:bb:cc:dd:ee:ff",
                destination_mac="ff:ff:ff:ff:ff:ff",
                dot11_frame_len=128,
            )

    monkeypatch.setattr("peekaboo.cli.iter_live_records", fake_live_records)
    result = CliRunner().invoke(
        app,
        [
            "presence-live",
            "--config",
            str(config_path),
            "--interface",
            "test0",
            "--max-packets",
            "5",
            "--quiet",
        ],
    )

    assert result.exit_code == 0, result.output
    assert len(read_all_rows(output_dir / "live_predictions.jsonl")) == 5
    assert read_all_rows(output_dir / "live_presence.jsonl")


def test_presence_live_all_targets_uses_passive_live_records_without_hardware(
    tmp_path: Path,
    monkeypatch,
):
    output_dir = tmp_path / "runs"
    config_path = tmp_path / "config.yaml"
    checkpoint_path = output_dir / "model.pkl"
    _write_multitarget_config(config_path, tmp_path / "missing.pcap", output_dir)
    OnlineModel("fake", AlwaysIphoneEstimator(), [], {}).save(checkpoint_path)

    def fake_live_records(interface, *, timeout_seconds=None, max_packets=None):
        del interface, timeout_seconds
        for index in range(max_packets or 5):
            yield PacketRecord(
                timestamp=float(index),
                source_file="live:test0",
                packet_index=index,
                data_rate=54.0,
                rssi=-40,
                frame_type=2,
                frame_subtype=0,
                source_mac=IPHONE_MAC,
                destination_mac="ff:ff:ff:ff:ff:ff",
                dot11_frame_len=128,
            )

    monkeypatch.setattr("peekaboo.cli.iter_live_records", fake_live_records)
    result = CliRunner().invoke(
        app,
        [
            "presence-live",
            "--config",
            str(config_path),
            "--interface",
            "test0",
            "--all-targets",
            "--max-packets",
            "5",
            "--quiet",
        ],
    )

    assert result.exit_code == 0, result.output
    assert {row["target_id"] for row in read_all_rows(output_dir / "live_presence.jsonl")} == {
        "iphone_5_user1",
        "lg_tv",
    }


def _write_config(config_path: Path, capture_path: Path, output_dir: Path) -> None:
    targets_path = output_dir / "targets.yaml"
    targets_path.parent.mkdir(parents=True, exist_ok=True)
    targets_path.write_text(
        yaml.safe_dump(
            {
                "targets": [
                    {
                        "target_id": "iphone_5_user1",
                        "label": "phone",
                        "mac_addresses": ["aa:bb:cc:dd:ee:ff"],
                        "enabled": True,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    config_path.write_text(
        yaml.safe_dump(_base_config(capture_path, output_dir, targets_path)),
        encoding="utf-8",
    )


def _write_multitarget_config(config_path: Path, capture_path: Path, output_dir: Path) -> None:
    targets_path = output_dir / "targets.yaml"
    targets_path.parent.mkdir(parents=True, exist_ok=True)
    targets_path.write_text(
        yaml.safe_dump(
            {
                "targets": [
                    {
                        "target_id": "iphone_5_user1",
                        "label": "phone",
                        "mac_addresses": [IPHONE_MAC],
                        "enabled": True,
                    },
                    {
                        "target_id": "lg_tv",
                        "label": "tv",
                        "mac_addresses": [LG_TV_MAC],
                        "enabled": True,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    data = _base_config(capture_path, output_dir, targets_path)
    data["labeling"] = {
        "mode": "multiclass_with_other",
        "target_id": None,
        "positive_label": "target",
        "negative_label": "other",
    }
    data["presence"] = {"target_classes": [], "all_targets": True}
    config_path.write_text(
        yaml.safe_dump(data),
        encoding="utf-8",
    )


def _base_config(capture_path: Path, output_dir: Path, targets_path: Path) -> dict:
    return {
        "input": {"paths": [str(capture_path)], "live_interface": None},
        "target_registry_path": str(targets_path),
        "output_dir": str(output_dir),
        "random_seed": 42,
        "model": {
            "model_id": "leveraging_bag",
            "checkpoint_path": str(output_dir / "model.pkl"),
            "params": {"n_models": 5, "base_model": {"grace_period": 5}},
        },
        "data": {
            "normalized_path": str(output_dir / "records.parquet"),
            "features_path": str(output_dir / "features.parquet"),
            "labeled_path": str(output_dir / "labeled.parquet"),
            "train_path": str(output_dir / "train.parquet"),
            "test_path": str(output_dir / "test.parquet"),
            "predictions_path": str(output_dir / "predictions.parquet"),
            "rolling_path": str(output_dir / "rolling.parquet"),
            "metrics_path": str(output_dir / "metrics.json"),
        },
        "windowing": {
            "frame_count": 10,
            "time_seconds": 10,
            "min_frames": 2,
            "present_ratio_threshold": 0.3,
            "mean_probability_threshold": 0.3,
            "max_probability_threshold": 0.8,
        },
    }
