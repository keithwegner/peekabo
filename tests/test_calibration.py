import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from peekaboo.calibration import (
    calibration_window_stats,
    evaluate_calibration_candidates,
    recommend_thresholds,
    threshold_candidates,
    write_calibration_report,
    write_recommended_windowing,
)
from peekaboo.capture.synthetic import IPHONE_MAC, LG_TV_MAC, write_synthetic_capture
from peekaboo.cli import app
from peekaboo.config import AppConfig, CalibrationConfig, WindowConfig
from peekaboo.data.readers import read_all_rows, read_json


def prediction(
    timestamp: float,
    *,
    predicted_class: str,
    ground_truth: str,
    target_probability: float,
) -> dict[str, object]:
    return {
        "timestamp": timestamp,
        "packet_index": int(timestamp),
        "predicted_class": predicted_class,
        "confidence": target_probability,
        "ground_truth": ground_truth,
        "probabilities_json": json.dumps(
            {"target": target_probability, "other": 1 - target_probability}
        ),
    }


def test_calibration_config_defaults_and_validation():
    config = AppConfig(output_dir=Path("runs/test"))

    assert config.calibration.output_dir == Path("runs/test/calibration")
    assert config.calibration.objective == "f1"

    with pytest.raises(ValueError, match="objective"):
        CalibrationConfig(objective="accuracy")
    with pytest.raises(ValueError, match="threshold"):
        CalibrationConfig(present_ratio_thresholds=[-0.1])
    with pytest.raises(ValueError, match="window type"):
        CalibrationConfig(window_types=["session"])
    with pytest.raises(ValueError, match="min_truth_frames"):
        CalibrationConfig(min_truth_frames=0)


def test_window_truth_labeling_threshold_sweep_and_recommendation(tmp_path: Path):
    rows = [
        prediction(0, predicted_class="target", ground_truth="target", target_probability=0.9),
        prediction(1, predicted_class="target", ground_truth="target", target_probability=0.8),
        prediction(2, predicted_class="other", ground_truth="other", target_probability=0.1),
        prediction(3, predicted_class="other", ground_truth="other", target_probability=0.2),
    ]
    window_config = WindowConfig(
        frame_count=2,
        time_seconds=10,
        min_frames=1,
        present_ratio_threshold=0.5,
        mean_probability_threshold=0.5,
        max_probability_threshold=0.8,
    )
    calibration_config = CalibrationConfig(
        present_ratio_thresholds=[0.5],
        mean_probability_thresholds=[0.5],
        max_probability_thresholds=[0.8],
        window_types=["frame_count"],
    )

    windows = calibration_window_stats(
        rows,
        target_classes=["target"],
        window_config=window_config,
        window_types=["frame_count"],
        min_truth_frames=1,
    )
    candidates = threshold_candidates(window_config, calibration_config)
    results = evaluate_calibration_candidates(
        windows,
        candidates=candidates,
        base_window_config=window_config,
        objective="f1",
    )
    recommendation = recommend_thresholds(results, objective="f1")

    assert [window["truth_present"] for window in windows] == [True, False]
    assert len(candidates) == 1
    assert recommendation["metrics"]["true_positive"] == 1
    assert recommendation["metrics"]["true_negative"] == 1
    assert recommendation["metrics"]["f1"] == 1.0

    report_path = tmp_path / "calibration_report.md"
    yaml_path = tmp_path / "recommended_windowing.yaml"
    manifest = {
        "status": "completed",
        "config_path": "config.yaml",
        "objective": "f1",
        "target_classes": ["target"],
        "prediction_source": "unit",
        "prediction_rows": len(rows),
        "window_count": len(windows),
        "warnings": [],
        "recommendation": recommendation,
        "artifacts": {"recommended_windowing": str(yaml_path)},
    }
    write_calibration_report(report_path, manifest=manifest, results=results)
    write_recommended_windowing(yaml_path, base=window_config, recommendation=recommendation)

    assert "peekaboo Presence Calibration" in report_path.read_text(encoding="utf-8")
    recommended = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    assert recommended["windowing"]["present_ratio_threshold"] == 0.5


def test_calibrate_presence_cli_regenerates_labeled_predictions(tmp_path: Path):
    capture_path = tmp_path / "synthetic-demo.pcap"
    config_path = tmp_path / "config.yaml"
    targets_path = tmp_path / "targets.yaml"
    output_dir = tmp_path / "runs"
    write_synthetic_capture(capture_path)
    _write_config(config_path, targets_path, capture_path, output_dir)

    runner = CliRunner()
    run_result = runner.invoke(
        app,
        ["run", "--config", str(config_path), "--force", "--quiet"],
    )
    assert run_result.exit_code == 0, run_result.output
    saved_predictions = read_all_rows(output_dir / "predictions.parquet")
    assert all(row.get("ground_truth") is None for row in saved_predictions)

    result = runner.invoke(
        app,
        ["calibrate-presence", "--config", str(config_path), "--force", "--quiet"],
    )

    assert result.exit_code == 0, result.output
    calibration_dir = output_dir / "calibration"
    manifest = read_json(calibration_dir / "calibration_manifest.json")
    rows = read_all_rows(calibration_dir / "calibration_results.csv")
    assert manifest["status"] == "completed"
    assert manifest["prediction_source"].startswith("generated from")
    assert "source_mac" not in manifest["feature_names"]
    assert "destination_mac" not in manifest["feature_names"]
    assert rows
    assert (calibration_dir / "calibration_results.json").exists()
    assert (calibration_dir / "calibration_report.md").exists()
    assert (calibration_dir / "recommended_windowing.yaml").exists()
    assert any((calibration_dir / f"calibration_{metric}.png").exists() for metric in ["f1", "mcc"])


def test_calibrate_presence_cli_all_targets_and_missing_ground_truth_failure(tmp_path: Path):
    capture_path = tmp_path / "synthetic-demo.pcap"
    config_path = tmp_path / "config.yaml"
    targets_path = tmp_path / "targets.yaml"
    output_dir = tmp_path / "runs"
    write_synthetic_capture(capture_path)
    _write_config(config_path, targets_path, capture_path, output_dir, multiclass=True)

    runner = CliRunner()
    for command in ["ingest", "features", "label", "train-online"]:
        result = runner.invoke(app, [command, "--config", str(config_path)])
        assert result.exit_code == 0, result.output

    result = runner.invoke(
        app,
        ["calibrate-presence", "--config", str(config_path), "--all-targets", "--quiet"],
    )

    assert result.exit_code == 0, result.output
    rows = read_all_rows(output_dir / "calibration" / "calibration_results.csv")
    assert {"iphone_5_user1", "lg_tv"} <= {
        row["target_id"] for row in rows if row["scope"] == "target_window"
    }

    unlabeled_path = output_dir / "unlabeled_predictions.jsonl"
    unlabeled_path.write_text(
        json.dumps(
            {
                "timestamp": 0,
                "predicted_class": "target",
                "confidence": 1.0,
                "probabilities_json": json.dumps({"target": 1.0, "other": 0.0}),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    failure = runner.invoke(
        app,
        [
            "calibrate-presence",
            "--config",
            str(config_path),
            "--input-predictions",
            str(unlabeled_path),
            "--output-dir",
            str(output_dir / "explicit-unlabeled"),
        ],
    )

    assert failure.exit_code == 1
    assert "ground_truth" in failure.output


def _write_config(
    config_path: Path,
    targets_path: Path,
    capture_path: Path,
    output_dir: Path,
    *,
    multiclass: bool = False,
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
                    "mode": "multiclass_with_other" if multiclass else "binary_one_vs_rest",
                    "target_id": None if multiclass else "iphone_5_user1",
                    "positive_label": "target",
                    "negative_label": "other",
                },
                "presence": {"target_classes": [], "all_targets": multiclass},
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
                "calibration": {
                    "objective": "f1",
                    "present_ratio_thresholds": [0.2, 0.3, 0.4],
                    "mean_probability_thresholds": [0.2, 0.3, 0.4],
                    "max_probability_thresholds": [0.7, 0.8],
                    "window_types": ["frame_count", "time"],
                    "min_truth_frames": 1,
                    "prepare_predictions_if_missing": True,
                    "output_dir": str(output_dir / "calibration"),
                },
            }
        ),
        encoding="utf-8",
    )
