"""Local capture onboarding helpers."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import yaml

from peekaboo.capture.inspect import inspect_capture_paths
from peekaboo.capture.sources import expand_input_paths
from peekaboo.config import load_config
from peekaboo.data.writers import write_json
from peekaboo.labeling.targets import TargetRegistry
from peekaboo.parsing.dot11 import normalize_mac
from peekaboo.runner import run_experiment

ProgressCallback = Callable[[str], None]


class SetupError(RuntimeError):
    """Raised when local capture setup cannot complete."""


def run_setup(
    *,
    input_paths: Iterable[str | Path],
    output_dir: str | Path,
    config_output: str | Path,
    targets_output: str | Path,
    target_id: str | None = None,
    target_mac: str | None = None,
    label: str | None = None,
    run_pipeline: bool = False,
    force: bool = False,
    quiet: bool = False,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    config_path = Path(config_output)
    targets_path = Path(targets_output)
    raw_input_paths = [Path(path) for path in input_paths]
    capture_paths = expand_input_paths(raw_input_paths)
    if not capture_paths:
        path_list = ", ".join(str(path) for path in raw_input_paths) or "<none>"
        raise SetupError(
            "No capture files found for input path(s): "
            f"{path_list}. Expected .pcap, .pcapng, or .cap files."
        )

    normalized_target_mac = _normalize_target_mac(target_mac)
    resolved_target_id = target_id or "target"
    resolved_label = label or "device"
    registry_data = (
        build_target_registry_data(
            target_id=resolved_target_id,
            target_mac=normalized_target_mac,
            label=resolved_label,
        )
        if normalized_target_mac is not None
        else None
    )
    registry = TargetRegistry.from_dict(registry_data) if registry_data is not None else None

    output_path.mkdir(parents=True, exist_ok=True)
    summary = inspect_capture_paths(capture_paths, registry=registry)
    inspect_path = output_path / "setup_inspect.json"
    candidates_path = output_path / "setup_candidates.md"
    write_json(inspect_path, summary)
    write_candidate_report(candidates_path, summary)

    result: dict[str, Any] = {
        "status": "candidates_only",
        "capture_files": [str(path) for path in capture_paths],
        "setup_inspect_path": str(inspect_path),
        "candidate_report_path": str(candidates_path),
        "config_path": None,
        "targets_path": None,
        "run_manifest_path": None,
    }

    if normalized_target_mac is None:
        if progress is not None and not quiet:
            progress(
                "No --target-mac supplied. Wrote candidate source MACs; rerun setup with "
                "--target-id and --target-mac to generate a runnable config."
            )
        return result

    _ensure_can_write(config_path, force=force)
    _ensure_can_write(targets_path, force=force)
    config_data = build_setup_config_data(
        input_paths=raw_input_paths,
        output_dir=output_path,
        target_registry_path=targets_path,
        target_id=resolved_target_id,
    )
    _write_yaml(targets_path, registry_data or {})
    _write_yaml(config_path, config_data)
    load_config(config_path)
    result.update(
        {
            "status": "configured",
            "config_path": str(config_path),
            "targets_path": str(targets_path),
        }
    )

    if progress is not None and not quiet:
        progress(f"Wrote config to {config_path}")
        progress(f"Wrote target registry to {targets_path}")

    if run_pipeline:
        manifest = run_experiment(
            config_path,
            force=force,
            quiet=quiet,
            progress=progress,
        )
        result["status"] = "ran"
        result["run_manifest_path"] = str(manifest["artifacts"]["run_manifest"])
    return result


def build_target_registry_data(
    *,
    target_id: str,
    target_mac: str,
    label: str,
) -> dict[str, Any]:
    normalized = _normalize_target_mac(target_mac)
    if normalized is None:
        raise SetupError(f"Invalid target MAC address: {target_mac}")
    return {
        "targets": [
            {
                "target_id": target_id,
                "label": label,
                "mac_addresses": [normalized],
                "enabled": True,
            }
        ]
    }


def build_setup_config_data(
    *,
    input_paths: Iterable[str | Path],
    output_dir: str | Path,
    target_registry_path: str | Path,
    target_id: str,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    return {
        "input": {
            "paths": [str(path) for path in input_paths],
            "live_interface": None,
        },
        "target_registry_path": str(target_registry_path),
        "output_dir": str(output_path),
        "random_seed": 42,
        "features": {
            "data_size_mode": "dot11_frame_len",
            "leakage_debug": False,
            "impute_numeric": -1.0,
            "impute_categorical": "missing",
        },
        "labeling": {
            "mode": "binary_one_vs_rest",
            "target_id": target_id,
            "positive_label": "target",
            "negative_label": "other",
        },
        "filters": {
            "known_targets_only": False,
            "include_ap_originated": True,
            "include_management": True,
            "include_control": True,
            "include_data": True,
            "channel_frequency": None,
            "start_time": None,
            "end_time": None,
            "rssi_min": None,
            "rssi_max": None,
            "min_frame_size": None,
            "ap_macs": [],
        },
        "model": {
            "model_id": "leveraging_bag",
            "checkpoint_path": str(output_path / "model.pkl"),
            "params": {"n_models": 5, "base_model": {"grace_period": 5}},
        },
        "data": {
            "normalized_path": str(output_path / "records.parquet"),
            "features_path": str(output_path / "features.parquet"),
            "labeled_path": str(output_path / "labeled.parquet"),
            "train_path": str(output_path / "train.parquet"),
            "test_path": str(output_path / "test.parquet"),
            "predictions_path": str(output_path / "predictions.parquet"),
            "rolling_path": str(output_path / "rolling.parquet"),
            "metrics_path": str(output_path / "metrics.json"),
        },
        "sampling": {
            "percentage": 100,
            "balance_strategy": "none",
            "class_weight_column": "sample_weight",
        },
        "split": {
            "train_fraction": 0.75,
            "chronological": True,
        },
        "windowing": {
            "frame_count": 20,
            "time_seconds": 20,
            "min_frames": 5,
            "present_ratio_threshold": 0.3,
            "mean_probability_threshold": 0.3,
            "max_probability_threshold": 0.8,
        },
    }


def write_candidate_report(path: str | Path, summary: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    channels = ", ".join(map(str, summary.get("channel_frequencies") or [])) or "none"
    lines = [
        "# peekaboo Setup Candidates",
        "",
        "## Capture Summary",
        "",
        f"- Capture files: `{summary.get('capture_file_count', 0)}`",
        f"- Packets scanned: `{summary.get('packets_scanned', 0)}`",
        f"- Dot11 frames: `{summary.get('dot11_frame_count', 0)}`",
        f"- Protected frames: `{summary.get('protected_frame_count', 0)}`",
        f"- Channel frequencies: `{channels}`",
        "",
        "## Source MAC Candidates",
        "",
        "| Rank | Source MAC | Frames |",
        "| ---: | --- | ---: |",
    ]
    source_counts = summary.get("source_mac_counts") or {}
    if source_counts:
        for index, (mac, count) in enumerate(source_counts.items(), start=1):
            lines.append(f"| {index} | `{mac}` | {count} |")
    else:
        lines.append("| - | No source MACs observed | 0 |")

    lines.extend(
        [
            "",
            "## Next Step",
            "",
            (
                "Choose a MAC address for a device you are authorized to identify, then rerun "
                "`peekaboo setup` with `--target-id` and `--target-mac`."
            ),
            "",
            "## Warnings",
            "",
        ]
    )
    warnings = summary.get("warnings") or []
    if warnings:
        lines.extend(f"- {warning}" for warning in warnings)
    else:
        lines.append("- None")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _ensure_can_write(path: Path, *, force: bool) -> None:
    if path.exists() and not force:
        raise SetupError(f"{path} already exists; use --force to overwrite it.")


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def _normalize_target_mac(mac: str | None) -> str | None:
    if mac is None:
        return None
    normalized = normalize_mac(mac)
    if normalized is None:
        raise SetupError(f"Invalid target MAC address: {mac}")
    return normalized
