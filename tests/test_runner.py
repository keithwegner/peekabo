import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from peekaboo.capture.synthetic import IPHONE_MAC, write_synthetic_capture
from peekaboo.cli import app
from peekaboo.config import AppConfig, DataConfig
from peekaboo.data.readers import read_all_rows, read_json
from peekaboo.features.extract import row_to_model_features
from peekaboo.runner import plan_stages, stage_outputs_exist

pytest.importorskip("scapy")


def test_plan_stages_profiles_and_slices():
    assert plan_stages(profile="full") == [
        "inspect",
        "ingest",
        "features",
        "label",
        "split",
        "train",
        "eval",
        "classify",
        "presence",
        "report",
    ]
    assert plan_stages(profile="prepare") == ["inspect", "ingest", "features", "label", "split"]
    assert plan_stages(profile="train-eval") == ["train", "eval", "classify", "report"]
    assert plan_stages(profile="full", from_stage="features", to_stage="train") == [
        "features",
        "label",
        "split",
        "train",
    ]
    with pytest.raises(ValueError, match="not part of profile"):
        plan_stages(profile="presence-replay", from_stage="inspect")


def test_stage_outputs_exist_requires_all_outputs(tmp_path: Path):
    config = AppConfig(
        output_dir=tmp_path,
        data=DataConfig(
            normalized_path=tmp_path / "records.parquet",
            features_path=tmp_path / "features.parquet",
            labeled_path=tmp_path / "labeled.parquet",
            train_path=tmp_path / "train.parquet",
            test_path=tmp_path / "test.parquet",
            predictions_path=tmp_path / "predictions.parquet",
            rolling_path=tmp_path / "rolling.parquet",
            metrics_path=tmp_path / "metrics.json",
        ),
    )

    assert not stage_outputs_exist(config, "inspect")
    (tmp_path / "inspect.json").write_text("{}", encoding="utf-8")
    assert not stage_outputs_exist(config, "inspect")
    (tmp_path / "inspect.md").write_text("# inspect\n", encoding="utf-8")
    assert stage_outputs_exist(config, "inspect")


def test_run_dry_run_prints_plan_without_artifacts(tmp_path: Path):
    capture_path = tmp_path / "synthetic-demo.pcap"
    config_path = tmp_path / "config.yaml"
    targets_path = tmp_path / "targets.yaml"
    output_dir = tmp_path / "runs"
    write_synthetic_capture(capture_path)
    _write_config(config_path, targets_path, capture_path, output_dir)

    result = CliRunner().invoke(app, ["run", "--config", str(config_path), "--dry-run"])

    assert result.exit_code == 0, result.output
    assert "Dry run planned stages" in result.output
    assert "inspect -> ingest -> features" in result.output
    assert not output_dir.exists()


def test_run_full_synthetic_pipeline_outputs_manifest_and_artifacts(tmp_path: Path):
    capture_path = tmp_path / "synthetic-demo.pcap"
    config_path = tmp_path / "config.yaml"
    targets_path = tmp_path / "targets.yaml"
    output_dir = tmp_path / "runs"
    write_synthetic_capture(capture_path)
    _write_config(config_path, targets_path, capture_path, output_dir)

    result = CliRunner().invoke(app, ["run", "--config", str(config_path), "--quiet"])

    assert result.exit_code == 0, result.output
    records = read_all_rows(output_dir / "records.parquet")
    features = read_all_rows(output_dir / "features.parquet")
    labels = read_all_rows(output_dir / "labeled.parquet")
    train_rows = read_all_rows(output_dir / "train.parquet")
    test_rows = read_all_rows(output_dir / "test.parquet")
    predictions = read_all_rows(output_dir / "predictions.parquet")
    rolling = read_all_rows(output_dir / "rolling.parquet")
    replay_predictions = read_all_rows(output_dir / "replay_predictions.jsonl")
    replay_presence = read_all_rows(output_dir / "replay_presence.jsonl")
    manifest = read_json(output_dir / "run_manifest.json")

    assert len(records) == 120
    assert len(features) == 120
    assert len(labels) == 120
    assert train_rows
    assert test_rows
    assert predictions
    assert rolling
    assert replay_predictions
    assert replay_presence
    assert (output_dir / "inspect.json").exists()
    assert (output_dir / "inspect.md").exists()
    assert (output_dir / "model.pkl").exists()
    assert (output_dir / "metrics.json").exists()
    assert (output_dir / "report.md").exists()
    assert (output_dir / "run_summary.md").exists()
    assert manifest["status"] == "completed"
    assert [stage["name"] for stage in manifest["stages"]] == manifest["stage_order"]

    feature_payload = json.loads(predictions[0]["features_json"])
    assert "source_mac" not in feature_payload
    assert "destination_mac" not in feature_payload
    model_features = row_to_model_features(features[0], AppConfig().features)
    assert "source_mac" not in model_features
    assert "destination_mac" not in model_features


def test_run_skip_existing_and_force_behavior(tmp_path: Path):
    capture_path = tmp_path / "synthetic-demo.pcap"
    config_path = tmp_path / "config.yaml"
    targets_path = tmp_path / "targets.yaml"
    output_dir = tmp_path / "runs"
    write_synthetic_capture(capture_path)
    _write_config(config_path, targets_path, capture_path, output_dir)
    runner = CliRunner()

    first = runner.invoke(app, ["run", "--config", str(config_path), "--profile", "prepare"])
    assert first.exit_code == 0, first.output

    blocked = runner.invoke(app, ["run", "--config", str(config_path), "--profile", "prepare"])
    assert blocked.exit_code == 1
    assert "Output files already exist" in blocked.output

    skipped = runner.invoke(
        app,
        ["run", "--config", str(config_path), "--profile", "prepare", "--skip-existing"],
    )
    assert skipped.exit_code == 0, skipped.output
    manifest = read_json(output_dir / "run_manifest.json")
    assert {stage["status"] for stage in manifest["stages"]} == {"skipped"}

    forced = runner.invoke(
        app,
        ["run", "--config", str(config_path), "--profile", "prepare", "--force", "--quiet"],
    )
    assert forced.exit_code == 0, forced.output
    manifest = read_json(output_dir / "run_manifest.json")
    assert {stage["status"] for stage in manifest["stages"]} == {"completed"}


def test_run_missing_capture_fails_and_records_manifest(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    targets_path = tmp_path / "targets.yaml"
    output_dir = tmp_path / "runs"
    _write_config(config_path, targets_path, tmp_path / "missing.pcap", output_dir)

    result = CliRunner().invoke(app, ["run", "--config", str(config_path)])

    assert result.exit_code == 1
    assert "No capture files found" in result.output
    manifest = read_json(output_dir / "run_manifest.json")
    assert manifest["status"] == "failed"
    assert manifest["stages"][0]["name"] == "inspect"
    assert manifest["stages"][0]["status"] == "failed"
    assert "No capture files found" in manifest["stages"][0]["error"]


def _write_config(
    config_path: Path,
    targets_path: Path,
    capture_path: Path,
    output_dir: Path,
) -> None:
    targets_path.write_text(
        yaml.safe_dump(
            {
                "targets": [
                    {
                        "target_id": "iphone_5_user1",
                        "label": "phone",
                        "mac_addresses": [IPHONE_MAC],
                        "enabled": True,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    config_path.write_text(
        yaml.safe_dump(
            {
                "input": {"paths": [str(capture_path)], "live_interface": None},
                "target_registry_path": str(targets_path),
                "output_dir": str(output_dir),
                "random_seed": 42,
                "features": {
                    "data_size_mode": "dot11_frame_len",
                    "leakage_debug": False,
                    "impute_numeric": -1.0,
                    "impute_categorical": "missing",
                },
                "labeling": {
                    "mode": "binary_one_vs_rest",
                    "target_id": "iphone_5_user1",
                    "positive_label": "target",
                    "negative_label": "other",
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
                "model": {
                    "model_id": "leveraging_bag",
                    "checkpoint_path": str(output_dir / "model.pkl"),
                    "params": {"n_models": 5, "base_model": {"grace_period": 5}},
                },
                "split": {"train_fraction": 0.75, "chronological": True},
                "windowing": {
                    "frame_count": 20,
                    "time_seconds": 20,
                    "min_frames": 5,
                    "present_ratio_threshold": 0.3,
                    "mean_probability_threshold": 0.3,
                    "max_probability_threshold": 0.8,
                },
            }
        ),
        encoding="utf-8",
    )
