import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from peekaboo.capture.synthetic import IPHONE_MAC, write_synthetic_capture
from peekaboo.cli import app
from peekaboo.config import load_config
from peekaboo.data.readers import read_all_rows, read_json
from peekaboo.labeling.targets import TargetRegistry
from peekaboo.setup import build_setup_config_data, build_target_registry_data

pytest.importorskip("scapy")


def test_setup_config_generation_roots_outputs_under_output_dir(tmp_path: Path):
    output_dir = tmp_path / "runs" / "home"
    config_data = build_setup_config_data(
        input_paths=[tmp_path / "captures"],
        output_dir=output_dir,
        target_registry_path=tmp_path / "configs" / "home-targets.yaml",
        target_id="my_phone",
    )

    assert config_data["features"]["leakage_debug"] is False
    assert config_data["labeling"]["target_id"] == "my_phone"
    for value in config_data["data"].values():
        assert Path(value).is_relative_to(output_dir)
    assert Path(config_data["model"]["checkpoint_path"]).is_relative_to(output_dir)


def test_target_registry_generation_loads_multi_module_match():
    registry_data = build_target_registry_data(
        target_id="my_phone",
        target_mac="AA:BB:CC:DD:EE:FF",
        label="phone",
    )
    registry = TargetRegistry.from_dict(registry_data)

    assert registry.target_id_for_mac("aa:bb:cc:dd:ee:ff") == "my_phone"
    assert registry_data["targets"][0]["label"] == "phone"
    assert registry_data["targets"][0]["mac_addresses"] == ["aa:bb:cc:dd:ee:ff"]


def test_setup_with_target_writes_config_registry_and_candidates(tmp_path: Path):
    capture_path = tmp_path / "synthetic-demo.pcap"
    output_dir = tmp_path / "runs" / "home"
    config_path = tmp_path / "configs" / "home.yaml"
    targets_path = tmp_path / "configs" / "home-targets.yaml"
    write_synthetic_capture(capture_path)

    result = CliRunner().invoke(
        app,
        [
            "setup",
            "--input",
            str(capture_path),
            "--output-dir",
            str(output_dir),
            "--config-output",
            str(config_path),
            "--targets-output",
            str(targets_path),
            "--target-id",
            "my_phone",
            "--target-mac",
            IPHONE_MAC,
            "--label",
            "phone",
            "--quiet",
        ],
    )

    assert result.exit_code == 0, result.output
    assert config_path.exists()
    assert targets_path.exists()
    assert (output_dir / "setup_inspect.json").exists()
    assert (output_dir / "setup_candidates.md").exists()
    config = load_config(config_path)
    registry = TargetRegistry.from_file(targets_path)
    summary = read_json(output_dir / "setup_inspect.json")
    candidates_text = (output_dir / "setup_candidates.md").read_text(encoding="utf-8")

    assert config.output_dir == output_dir
    assert config.features.leakage_debug is False
    assert registry.target_id_for_mac(IPHONE_MAC) == "my_phone"
    assert summary["target_match_total"] > 0
    assert IPHONE_MAC in summary["source_mac_counts"]
    assert IPHONE_MAC in candidates_text


def test_setup_run_executes_full_pipeline_and_preserves_feature_policy(tmp_path: Path):
    capture_path = tmp_path / "synthetic-demo.pcap"
    output_dir = tmp_path / "runs" / "home"
    config_path = tmp_path / "configs" / "home.yaml"
    targets_path = tmp_path / "configs" / "home-targets.yaml"
    write_synthetic_capture(capture_path)

    result = CliRunner().invoke(
        app,
        [
            "setup",
            "--input",
            str(capture_path),
            "--output-dir",
            str(output_dir),
            "--config-output",
            str(config_path),
            "--targets-output",
            str(targets_path),
            "--target-id",
            "my_phone",
            "--target-mac",
            IPHONE_MAC,
            "--run",
            "--quiet",
        ],
    )

    assert result.exit_code == 0, result.output
    assert (output_dir / "run_manifest.json").exists()
    labels = read_all_rows(output_dir / "labeled.parquet")
    predictions = read_all_rows(output_dir / "predictions.parquet")
    manifest = read_json(output_dir / "run_manifest.json")

    assert manifest["status"] == "completed"
    assert any(row["source_mac"] == IPHONE_MAC for row in labels)
    feature_payload = json.loads(predictions[0]["features_json"])
    assert "source_mac" not in feature_payload
    assert "destination_mac" not in feature_payload


def test_setup_missing_capture_fails_with_no_capture_wording(tmp_path: Path):
    result = CliRunner().invoke(
        app,
        [
            "setup",
            "--input",
            str(tmp_path / "missing.pcap"),
            "--output-dir",
            str(tmp_path / "runs" / "home"),
            "--config-output",
            str(tmp_path / "configs" / "home.yaml"),
            "--targets-output",
            str(tmp_path / "configs" / "home-targets.yaml"),
        ],
    )

    assert result.exit_code == 1
    assert "No capture files found" in result.output


def test_setup_existing_config_and_targets_require_force(tmp_path: Path):
    capture_path = tmp_path / "synthetic-demo.pcap"
    output_dir = tmp_path / "runs" / "home"
    config_path = tmp_path / "configs" / "home.yaml"
    targets_path = tmp_path / "configs" / "home-targets.yaml"
    write_synthetic_capture(capture_path)
    config_path.parent.mkdir(parents=True)
    config_path.write_text("existing: true\n", encoding="utf-8")
    targets_path.write_text("existing: true\n", encoding="utf-8")

    result = CliRunner().invoke(
        app,
        [
            "setup",
            "--input",
            str(capture_path),
            "--output-dir",
            str(output_dir),
            "--config-output",
            str(config_path),
            "--targets-output",
            str(targets_path),
            "--target-id",
            "my_phone",
            "--target-mac",
            IPHONE_MAC,
        ],
    )

    assert result.exit_code == 1
    assert "already exists" in result.output

    forced = CliRunner().invoke(
        app,
        [
            "setup",
            "--input",
            str(capture_path),
            "--output-dir",
            str(output_dir),
            "--config-output",
            str(config_path),
            "--targets-output",
            str(targets_path),
            "--target-id",
            "my_phone",
            "--target-mac",
            IPHONE_MAC,
            "--force",
        ],
    )
    assert forced.exit_code == 0, forced.output


def test_setup_without_target_mac_writes_candidates_only(tmp_path: Path):
    capture_path = tmp_path / "synthetic-demo.pcap"
    output_dir = tmp_path / "runs" / "home"
    config_path = tmp_path / "configs" / "home.yaml"
    targets_path = tmp_path / "configs" / "home-targets.yaml"
    write_synthetic_capture(capture_path)

    result = CliRunner().invoke(
        app,
        [
            "setup",
            "--input",
            str(capture_path),
            "--output-dir",
            str(output_dir),
            "--config-output",
            str(config_path),
            "--targets-output",
            str(targets_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Rerun with --target-id and --target-mac" in result.output
    assert (output_dir / "setup_inspect.json").exists()
    assert (output_dir / "setup_candidates.md").exists()
    assert not config_path.exists()
    assert not targets_path.exists()
    assert IPHONE_MAC in (output_dir / "setup_candidates.md").read_text(encoding="utf-8")
