from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from peekaboo.capture.synthetic import IPHONE_MAC, write_synthetic_capture
from peekaboo.cli import app
from peekaboo.comparison import (
    ComparisonError,
    plan_comparisons,
    write_comparison_charts,
    write_comparison_report,
)
from peekaboo.config import AppConfig, ComparisonConfig
from peekaboo.data.readers import read_json

pytest.importorskip("scapy")


def test_comparison_config_defaults_and_validation(tmp_path: Path):
    config = AppConfig(output_dir=tmp_path / "runs")

    assert config.comparison.models == [
        "leveraging_bag",
        "oza_boost",
        "oza_boost_adwin",
        "adaptive_hoeffding_tree",
    ]
    assert config.comparison.train_fractions == [0.01, 0.1, 0.5, 0.9]
    assert config.comparison.output_dir == tmp_path / "runs" / "comparison"

    with pytest.raises(ValueError, match="Unsupported comparison model"):
        ComparisonConfig(models=["not_a_model"])
    with pytest.raises(ValueError, match="0 < fraction < 1"):
        ComparisonConfig(train_fractions=[0.0, 1.0])


def test_plan_comparisons_builds_model_fraction_matrix():
    plan = plan_comparisons(
        models=["leveraging_bag", "adaptive_hoeffding_tree"],
        train_fractions=[0.1, 0.9],
    )

    assert [(item.model_id, item.train_fraction) for item in plan] == [
        ("leveraging_bag", 0.1),
        ("leveraging_bag", 0.9),
        ("adaptive_hoeffding_tree", 0.1),
        ("adaptive_hoeffding_tree", 0.9),
    ]

    with pytest.raises(ComparisonError, match="Unsupported comparison model"):
        plan_comparisons(models=["bad"], train_fractions=[0.5])


def test_comparison_report_and_empty_metric_charts(tmp_path: Path):
    manifest = {
        "status": "completed",
        "config_path": "config.yaml",
        "random_seed": 42,
        "labeled_rows": 10,
        "chronological_split": True,
        "models": ["leveraging_bag"],
        "train_fractions": [0.5],
        "leakage_debug": False,
        "artifacts": {"results_csv": str(tmp_path / "comparison_results.csv")},
    }
    results = [
        {
            "model_id": "leveraging_bag",
            "train_fraction": 0.5,
            "train_rows": 5,
            "test_rows": 5,
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "mcc": None,
            "roc_auc": None,
            "pr_auc": None,
            "model_mapping": "mapping",
            "elapsed_seconds": 0.1,
        }
    ]

    report_path = tmp_path / "comparison_report.md"
    write_comparison_report(report_path, manifest=manifest, results=results)
    chart_paths = write_comparison_charts(tmp_path, results)

    assert "peekaboo Experiment Comparison" in report_path.read_text(encoding="utf-8")
    assert chart_paths == {}


def test_compare_cli_runs_synthetic_model_fraction_matrix(tmp_path: Path):
    capture_path = tmp_path / "synthetic-demo.pcap"
    config_path = tmp_path / "config.yaml"
    targets_path = tmp_path / "targets.yaml"
    output_dir = tmp_path / "runs"
    comparison_dir = output_dir / "comparison"
    write_synthetic_capture(capture_path)
    _write_config(
        config_path,
        targets_path,
        capture_path,
        output_dir,
        models=[
            "leveraging_bag",
            "oza_boost",
            "oza_boost_adwin",
            "adaptive_hoeffding_tree",
        ],
        train_fractions=[0.1, 0.5],
    )

    result = CliRunner().invoke(app, ["compare", "--config", str(config_path), "--quiet"])

    assert result.exit_code == 0, result.output
    manifest = read_json(comparison_dir / "comparison_manifest.json")
    results = read_json(comparison_dir / "comparison_results.json")["results"]
    csv_text = (comparison_dir / "comparison_results.csv").read_text(encoding="utf-8")

    assert manifest["status"] == "completed"
    assert manifest["result_count"] == 8
    assert "source_mac" not in manifest["feature_names"]
    assert "destination_mac" not in manifest["feature_names"]
    assert {row["model_id"] for row in results} == set(manifest["models"])
    assert {row["train_fraction"] for row in results} == {0.1, 0.5}
    assert {"accuracy", "precision", "recall", "f1", "mcc", "roc_auc", "pr_auc"} <= set(
        results[0]
    )
    assert "leveraging_bag" in csv_text
    assert (comparison_dir / "comparison_report.md").exists()
    assert any(path.name.startswith("comparison_") for path in comparison_dir.glob("*.png"))


def test_compare_no_prepare_requires_existing_labeled_dataset(tmp_path: Path):
    capture_path = tmp_path / "synthetic-demo.pcap"
    config_path = tmp_path / "config.yaml"
    targets_path = tmp_path / "targets.yaml"
    output_dir = tmp_path / "runs"
    write_synthetic_capture(capture_path)
    _write_config(config_path, targets_path, capture_path, output_dir)

    result = CliRunner().invoke(app, ["compare", "--config", str(config_path), "--no-prepare"])

    assert result.exit_code == 1
    assert "Labeled dataset" in result.output
    assert "run `peekaboo run --profile prepare`" in result.output


def test_compare_force_overwrites_existing_outputs(tmp_path: Path):
    capture_path = tmp_path / "synthetic-demo.pcap"
    config_path = tmp_path / "config.yaml"
    targets_path = tmp_path / "targets.yaml"
    output_dir = tmp_path / "runs"
    write_synthetic_capture(capture_path)
    _write_config(
        config_path,
        targets_path,
        capture_path,
        output_dir,
        models=["leveraging_bag"],
        train_fractions=[0.5],
    )
    runner = CliRunner()

    first = runner.invoke(
        app,
        [
            "compare",
            "--config",
            str(config_path),
            "--models",
            "leveraging_bag",
            "--train-fractions",
            "0.5",
            "--quiet",
        ],
    )
    assert first.exit_code == 0, first.output
    blocked = runner.invoke(app, ["compare", "--config", str(config_path), "--quiet"])
    assert blocked.exit_code == 1
    assert "Comparison output files already exist" in blocked.output
    forced = runner.invoke(app, ["compare", "--config", str(config_path), "--force", "--quiet"])
    assert forced.exit_code == 0, forced.output


def _write_config(
    config_path: Path,
    targets_path: Path,
    capture_path: Path,
    output_dir: Path,
    *,
    models: list[str] | None = None,
    train_fractions: list[float] | None = None,
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
                "comparison": {
                    "models": models or ["leveraging_bag"],
                    "train_fractions": train_fractions or [0.5],
                    "output_dir": str(output_dir / "comparison"),
                    "prepare_if_missing": True,
                },
            }
        ),
        encoding="utf-8",
    )
