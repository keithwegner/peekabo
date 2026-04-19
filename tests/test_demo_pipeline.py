import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from peekabo.capture.synthetic import IPHONE_MAC, write_synthetic_capture
from peekabo.cli import app
from peekabo.config import FeatureConfig
from peekabo.data.readers import read_all_rows
from peekabo.features.extract import row_to_model_features

pytest.importorskip("scapy")


def test_synthetic_capture_cli_pipeline(tmp_path: Path):
    capture_path = tmp_path / "synthetic-demo.pcap"
    config_path = tmp_path / "config.yaml"
    targets_path = tmp_path / "targets.yaml"
    output_dir = tmp_path / "runs"
    write_synthetic_capture(capture_path)
    _write_demo_config(config_path, targets_path, capture_path, output_dir)

    runner = CliRunner()
    ingest_result = runner.invoke(app, ["ingest", "--config", str(config_path)])
    assert ingest_result.exit_code == 0, ingest_result.output
    feature_result = runner.invoke(app, ["features", "--config", str(config_path)])
    assert feature_result.exit_code == 0, feature_result.output
    label_result = runner.invoke(app, ["label", "--config", str(config_path)])
    assert label_result.exit_code == 0, label_result.output

    records = read_all_rows(output_dir / "records.csv")
    features = read_all_rows(output_dir / "features.csv")
    labels = read_all_rows(output_dir / "labeled.csv")
    assert len(records) == 5
    assert len(features) == 5
    assert len(labels) == 5
    assert {row["label"] for row in labels} == {"target", "other"}
    assert any(row["source_mac"] == IPHONE_MAC for row in features)

    model_features = row_to_model_features(features[0], FeatureConfig())
    assert "source_mac" not in model_features
    assert "destination_mac" not in model_features


def test_synthetic_capture_ingest_works_in_fresh_process(tmp_path: Path):
    capture_path = tmp_path / "synthetic-demo.pcap"
    config_path = tmp_path / "config.yaml"
    targets_path = tmp_path / "targets.yaml"
    output_dir = tmp_path / "runs"
    write_synthetic_capture(capture_path)
    _write_demo_config(config_path, targets_path, capture_path, output_dir)

    env = os.environ.copy()
    repo_src = Path(__file__).resolve().parents[1] / "src"
    env["PYTHONPATH"] = f"{repo_src}{os.pathsep}{env.get('PYTHONPATH', '')}"
    result = subprocess.run(
        [sys.executable, "-m", "peekabo.cli", "ingest", "--config", str(config_path)],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 0, result.stderr + result.stdout
    records = read_all_rows(output_dir / "records.csv")
    assert len(records) == 5
    assert all(row["parse_ok"] for row in records)


def test_ingest_warns_and_fails_when_no_capture_files(tmp_path: Path):
    config_path = tmp_path / "empty.yaml"
    output_path = tmp_path / "records.csv"
    config_path.write_text(
        yaml.safe_dump(
            {
                "input": {"paths": [str(tmp_path / "missing-captures")], "live_interface": None},
                "output_dir": str(tmp_path / "runs"),
                "data": {"normalized_path": str(output_path)},
            }
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(app, ["ingest", "--config", str(config_path)])

    assert result.exit_code == 1
    assert "No capture files found" in result.output
    assert not output_path.exists()


def _write_demo_config(
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
                    "normalized_path": str(output_dir / "records.csv"),
                    "features_path": str(output_dir / "features.csv"),
                    "labeled_path": str(output_dir / "labeled.csv"),
                    "train_path": str(output_dir / "train.csv"),
                    "test_path": str(output_dir / "test.csv"),
                    "predictions_path": str(output_dir / "predictions.csv"),
                    "rolling_path": str(output_dir / "rolling.csv"),
                    "metrics_path": str(output_dir / "metrics.json"),
                },
            }
        ),
        encoding="utf-8",
    )
