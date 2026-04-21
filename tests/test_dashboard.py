from pathlib import Path

import yaml
from typer.testing import CliRunner

from peekaboo.capture.synthetic import IPHONE_MAC, LG_TV_MAC, write_synthetic_capture
from peekaboo.cli import app
from peekaboo.dashboard import generate_dashboard, summarize_presence_file
from peekaboo.data.writers import write_json, write_rows


def test_dashboard_renders_partial_artifacts_without_external_assets(tmp_path: Path):
    output_dir = tmp_path / "runs"
    config_path = tmp_path / "config.yaml"
    _write_config(config_path, tmp_path / "targets.yaml", tmp_path / "missing.pcap", output_dir)
    write_json(
        output_dir / "run_manifest.json",
        {
            "status": "completed",
            "profile": "minimal",
            "random_seed": 42,
            "stages": [{"name": "eval", "status": "completed", "row_counts": {"rows": 4}}],
            "artifacts": {},
        },
    )
    write_json(
        output_dir / "metrics.json",
        {
            "n_examples": 4,
            "accuracy": 0.75,
            "precision": 0.8,
            "recall": 0.7,
            "f1": 0.74,
            "mcc": 0.5,
            "roc_auc": 0.9,
            "pr_auc": 0.8,
            "per_class": {
                "target": {"precision": 0.8, "recall": 0.7, "f1": 0.74, "support": 2},
                "other": {"precision": 0.7, "recall": 0.8, "f1": 0.74, "support": 2},
            },
            "confusion_matrix": {"labels": ["target", "other"], "matrix": [[1, 1], [0, 2]]},
        },
    )

    result = generate_dashboard(config_path, force=True)
    html = Path(result["output"]).read_text(encoding="utf-8")

    assert result["artifact_count"] >= 2
    assert "Run Overview" in html
    assert "Evaluation" in html
    assert "Artifact Index" in html
    assert "Missing optional artifacts" in html
    assert "run_manifest.json" in html
    assert "metrics.json" in html
    assert "http://" not in html
    assert "https://" not in html


def test_presence_summary_groups_target_windows(tmp_path: Path):
    path = tmp_path / "presence.jsonl"
    write_rows(
        path,
        [
            {"target_id": "phone", "window_type": "frame_count", "state": "present"},
            {"target_id": "phone", "window_type": "frame_count", "state": "absent"},
            {"target_id": "tv", "window_type": "time", "state": "uncertain"},
        ],
    )

    summary = summarize_presence_file(path)

    assert summary["row_count"] == 3
    assert {(group["target_id"], group["window_type"]) for group in summary["groups"]} == {
        ("phone", "frame_count"),
        ("tv", "time"),
    }
    phone = next(group for group in summary["groups"] if group["target_id"] == "phone")
    assert phone["states"] == {"present": 1, "absent": 1}


def test_dashboard_cli_full_synthetic_run_and_existing_output_guard(tmp_path: Path):
    capture_path = tmp_path / "synthetic-demo.pcap"
    config_path = tmp_path / "config.yaml"
    targets_path = tmp_path / "targets.yaml"
    output_dir = tmp_path / "runs"
    write_synthetic_capture(capture_path)
    _write_config(config_path, targets_path, capture_path, output_dir)

    runner = CliRunner()
    run_result = runner.invoke(app, ["run", "--config", str(config_path), "--force", "--quiet"])
    assert run_result.exit_code == 0, run_result.output

    dashboard_result = runner.invoke(app, ["dashboard", "--config", str(config_path)])
    assert dashboard_result.exit_code == 0, dashboard_result.output

    dashboard_path = output_dir / "dashboard" / "index.html"
    html = dashboard_path.read_text(encoding="utf-8")
    assert "peekaboo Run Dashboard" in html
    assert "Capture Readiness" in html
    assert "Target Presence" in html
    assert "target" in html
    assert "data:image/png;base64" in html
    assert "source_mac" not in html
    assert "destination_mac" not in html

    second_result = runner.invoke(app, ["dashboard", "--config", str(config_path)])
    assert second_result.exit_code == 1
    assert "use --force" in second_result.output


def test_dashboard_cli_multitarget_presence_sections(tmp_path: Path):
    capture_path = tmp_path / "synthetic-demo.pcap"
    config_path = tmp_path / "config.yaml"
    targets_path = tmp_path / "targets.yaml"
    output_dir = tmp_path / "runs"
    write_synthetic_capture(capture_path)
    _write_config(config_path, targets_path, capture_path, output_dir, multiclass=True)

    runner = CliRunner()
    run_result = runner.invoke(app, ["run", "--config", str(config_path), "--force", "--quiet"])
    assert run_result.exit_code == 0, run_result.output
    dashboard_result = runner.invoke(app, ["dashboard", "--config", str(config_path)])
    assert dashboard_result.exit_code == 0, dashboard_result.output

    html = (output_dir / "dashboard" / "index.html").read_text(encoding="utf-8")
    assert "iphone_5_user1" in html
    assert "lg_tv" in html
    assert "frame_count" in html
    assert "time" in html


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
            }
        ),
        encoding="utf-8",
    )
