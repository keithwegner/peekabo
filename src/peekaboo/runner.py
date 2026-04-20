"""Config-driven experiment runner."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from peekaboo import __version__
from peekaboo.capture.inspect import inspect_capture_paths, write_inspection_markdown
from peekaboo.capture.sources import expand_input_paths, iter_packet_records
from peekaboo.config import AppConfig, load_config, write_run_config
from peekaboo.data.readers import iter_rows, read_json
from peekaboo.data.splits import split_file
from peekaboo.data.writers import JsonlRowWriter, write_json, write_rows
from peekaboo.evaluation.holdout import evaluate_holdout_rows, train_online_rows
from peekaboo.evaluation.plots import (
    plot_binary_curves,
    plot_confusion_matrix,
    write_confusion_matrix_csv,
)
from peekaboo.evaluation.reports import write_markdown_report
from peekaboo.features.extract import iter_feature_rows, model_feature_names
from peekaboo.inference.aggregate import rolling_aggregates
from peekaboo.inference.classify import classify_row, classify_rows
from peekaboo.inference.presence import StreamingPresenceEngine
from peekaboo.labeling.filters import passes_filters
from peekaboo.labeling.labelers import iter_labeled_rows
from peekaboo.labeling.targets import TargetRegistry
from peekaboo.models.base import load_checkpoint
from peekaboo.models.registry import create_model

StageName = str
ProgressCallback = Callable[[str], None]

STAGE_ORDER: tuple[StageName, ...] = (
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
)

PROFILES: dict[str, tuple[StageName, ...]] = {
    "full": STAGE_ORDER,
    "prepare": ("inspect", "ingest", "features", "label", "split"),
    "train-eval": ("train", "eval", "classify", "report"),
    "presence-replay": ("train", "presence"),
}


class ExperimentRunError(RuntimeError):
    """Raised when an experiment runner stage fails."""


@dataclass(frozen=True)
class RunOptions:
    profile: str = "full"
    from_stage: str | None = None
    to_stage: str | None = None
    force: bool = False
    skip_existing: bool = False
    dry_run: bool = False
    quiet: bool = False


def plan_stages(
    *,
    profile: str = "full",
    from_stage: str | None = None,
    to_stage: str | None = None,
) -> list[StageName]:
    if profile not in PROFILES:
        raise ValueError(f"Unknown run profile {profile!r}; expected one of {sorted(PROFILES)}")
    stages = list(PROFILES[profile])
    if from_stage is not None:
        _require_stage_in_profile(from_stage, stages, profile)
        stages = stages[stages.index(from_stage) :]
    if to_stage is not None:
        _require_stage_in_profile(to_stage, stages, profile)
        stages = stages[: stages.index(to_stage) + 1]
    if not stages:
        raise ValueError("No stages selected")
    return stages


def stage_outputs(config: AppConfig, stage: StageName) -> list[Path]:
    output_dir = config.output_dir
    paths: dict[StageName, list[Path]] = {
        "inspect": [output_dir / "inspect.json", output_dir / "inspect.md"],
        "ingest": [config.data.normalized_path],
        "features": [config.data.features_path],
        "label": [config.data.labeled_path],
        "split": [config.data.train_path, config.data.test_path],
        "train": [config.model.checkpoint_path],
        "eval": [config.data.metrics_path],
        "classify": [config.data.predictions_path, config.data.rolling_path],
        "presence": [
            output_dir / "replay_predictions.jsonl",
            output_dir / "replay_presence.jsonl",
        ],
        "report": [output_dir / "report.md"],
    }
    return paths[stage]


def stage_outputs_exist(config: AppConfig, stage: StageName) -> bool:
    outputs = stage_outputs(config, stage)
    return bool(outputs) and all(path.exists() for path in outputs)


def run_experiment(
    config_path: str | Path,
    *,
    profile: str = "full",
    from_stage: str | None = None,
    to_stage: str | None = None,
    force: bool = False,
    skip_existing: bool = False,
    dry_run: bool = False,
    quiet: bool = False,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    if force and skip_existing:
        raise ValueError("Use either --force or --skip-existing, not both")

    config = load_config(config_path)
    options = RunOptions(
        profile=profile,
        from_stage=from_stage,
        to_stage=to_stage,
        force=force,
        skip_existing=skip_existing,
        dry_run=dry_run,
        quiet=quiet,
    )
    stages = plan_stages(profile=profile, from_stage=from_stage, to_stage=to_stage)
    manifest = _new_manifest(config, Path(config_path), options, stages)

    if dry_run:
        manifest["status"] = "dry_run"
        manifest["ended_at"] = _now()
        if progress is not None:
            progress("Planned stages: " + " -> ".join(stages))
        return manifest

    conflicts = _preexisting_outputs(config, stages)
    if conflicts and not force and not skip_existing:
        error = (
            "Output files already exist; use --force to overwrite or --skip-existing "
            f"to reuse them: {', '.join(str(path) for path in conflicts)}"
        )
        manifest["stages"].append(_stage_result("preflight", "failed", error=error))
        manifest["status"] = "failed"
        manifest["ended_at"] = _now()
        _write_run_outputs(config, manifest)
        raise ExperimentRunError(error)

    for stage in stages:
        if skip_existing and stage_outputs_exist(config, stage):
            result = _stage_result(
                stage,
                "skipped",
                outputs=stage_outputs(config, stage),
                message="All required outputs already exist.",
            )
            manifest["stages"].append(result)
            if progress is not None and not quiet:
                progress(f"Skipped {stage}; outputs already exist")
            continue

        if force:
            _remove_outputs(stage_outputs(config, stage))

        if progress is not None and not quiet:
            progress(f"Running {stage}")
        result = _stage_result(stage, "running", outputs=stage_outputs(config, stage))
        try:
            counts = _run_stage(config, stage, quiet=quiet)
        except Exception as exc:
            result["status"] = "failed"
            result["error"] = str(exc)
            result["ended_at"] = _now()
            manifest["stages"].append(result)
            manifest["status"] = "failed"
            manifest["ended_at"] = _now()
            _write_run_outputs(config, manifest)
            raise ExperimentRunError(f"Stage {stage} failed: {exc}") from exc

        result["status"] = "completed"
        result["row_counts"] = counts
        result["ended_at"] = _now()
        manifest["stages"].append(result)

    manifest["status"] = "completed"
    manifest["ended_at"] = _now()
    _write_run_outputs(config, manifest)
    return manifest


def write_run_summary(path: str | Path, manifest: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# peekaboo Run Summary",
        "",
        f"- Status: `{manifest.get('status')}`",
        f"- Config: `{manifest.get('config_path')}`",
        f"- Profile: `{manifest.get('profile')}`",
        f"- Random seed: `{manifest.get('random_seed')}`",
        f"- Started: `{manifest.get('started_at')}`",
        f"- Ended: `{manifest.get('ended_at')}`",
        "",
        "## Stages",
        "",
        "| Stage | Status | Counts |",
        "| --- | --- | --- |",
    ]
    for stage in manifest.get("stages", []):
        counts = stage.get("row_counts") or {}
        count_text = ", ".join(f"{key}={value}" for key, value in counts.items()) or "-"
        lines.append(f"| `{stage.get('name')}` | `{stage.get('status')}` | {count_text} |")

    lines.extend(
        [
            "",
            "## Key Artifacts",
            "",
        ]
    )
    for name, path_value in (manifest.get("artifacts") or {}).items():
        lines.append(f"- `{name}`: `{path_value}`")

    failures = [stage for stage in manifest.get("stages", []) if stage.get("status") == "failed"]
    if failures:
        lines.extend(["", "## Failures", ""])
        for failure in failures:
            lines.append(f"- `{failure.get('name')}`: {failure.get('error')}")

    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_stage(config: AppConfig, stage: StageName, *, quiet: bool) -> dict[str, int]:
    stage_handlers = {
        "inspect": _run_inspect,
        "ingest": _run_ingest,
        "features": _run_features,
        "label": _run_label,
        "split": _run_split,
        "train": _run_train,
        "eval": _run_eval_holdout,
        "classify": _run_classify_file,
        "presence": _run_presence_replay,
        "report": _run_report,
    }
    return stage_handlers[stage](config, quiet=quiet)


def _run_inspect(config: AppConfig, *, quiet: bool) -> dict[str, int]:
    del quiet
    capture_paths = _capture_paths_or_raise(config)
    registry = _target_registry_optional(config)
    summary = inspect_capture_paths(capture_paths, registry=registry)
    write_json(config.output_dir / "inspect.json", summary)
    write_inspection_markdown(config.output_dir / "inspect.md", summary)
    return {
        "capture_files": int(summary["capture_file_count"]),
        "packets_scanned": int(summary["packets_scanned"]),
        "dot11_frames": int(summary["dot11_frame_count"]),
    }


def _run_ingest(config: AppConfig, *, quiet: bool) -> dict[str, int]:
    del quiet
    capture_paths = _capture_paths_or_raise(config)
    write_run_config(config, config.output_dir)
    count = write_rows(
        config.data.normalized_path,
        (record.to_dict() for record in iter_packet_records(capture_paths)),
    )
    return {"records": count}


def _run_features(config: AppConfig, *, quiet: bool) -> dict[str, int]:
    del quiet
    rows = (row for row in iter_rows(config.data.normalized_path) if row.get("parse_ok", True))
    count = write_rows(config.data.features_path, iter_feature_rows(rows, config.features))
    return {"features": count}


def _run_label(config: AppConfig, *, quiet: bool) -> dict[str, int]:
    del quiet
    registry = _target_registry(config)

    def filtered_rows():
        for row in iter_rows(config.data.features_path):
            if passes_filters(row, config.filters, registry=registry):
                yield row

    count = write_rows(
        config.data.labeled_path,
        iter_labeled_rows(filtered_rows(), registry, config.labeling),
    )
    return {"labels": count}


def _run_split(config: AppConfig, *, quiet: bool) -> dict[str, int]:
    del quiet
    train_count, test_count = split_file(
        config.data.labeled_path,
        config.data.train_path,
        config.data.test_path,
        train_fraction=config.split.train_fraction,
        chronological=config.split.chronological,
        seed=config.random_seed,
    )
    return {"train": train_count, "test": test_count}


def _run_train(config: AppConfig, *, quiet: bool) -> dict[str, int]:
    del quiet
    model = create_model(
        config.model.model_id,
        feature_names=model_feature_names(config.features),
        seed=config.random_seed,
        params=config.model.params,
    )
    count = train_online_rows(
        iter_rows(config.data.labeled_path),
        model,
        config.features,
        weight_column=config.sampling.class_weight_column,
    )
    model.save(config.model.checkpoint_path)
    return {"train_examples": count}


def _run_eval_holdout(config: AppConfig, *, quiet: bool) -> dict[str, int]:
    del quiet
    model = create_model(
        config.model.model_id,
        feature_names=model_feature_names(config.features),
        seed=config.random_seed,
        params=config.model.params,
    )
    metrics, predictions = evaluate_holdout_rows(
        iter_rows(config.data.train_path),
        iter_rows(config.data.test_path),
        model,
        config.features,
        positive_label=config.labeling.positive_label,
        weight_column=config.sampling.class_weight_column,
    )
    _write_eval_outputs(config, metrics, predictions)
    return {
        "train_examples": int(metrics.get("train_examples") or 0),
        "eval_examples": int(metrics.get("n_examples") or 0),
        "predictions": len(predictions),
    }


def _run_classify_file(config: AppConfig, *, quiet: bool) -> dict[str, int]:
    del quiet
    model = load_checkpoint(config.model.checkpoint_path)
    predictions = classify_rows(
        iter_rows(config.data.features_path),
        model,
        config.features,
        label_mode=config.labeling.mode,
        positive_label=config.labeling.positive_label,
    )
    write_rows(config.data.predictions_path, predictions)
    target = _default_rolling_target_class(config)
    rolling = rolling_aggregates(predictions, target_class=target, config=config.windowing)
    write_rows(config.data.rolling_path, rolling)
    return {"predictions": len(predictions), "rolling": len(rolling)}


def _run_presence_replay(config: AppConfig, *, quiet: bool) -> dict[str, int]:
    model = load_checkpoint(config.model.checkpoint_path)
    prediction_count, event_count = _write_presence_stream(
        iter_rows(config.data.features_path),
        model,
        config,
        target_class=_default_rolling_target_class(config),
        predictions_output=config.output_dir / "replay_predictions.jsonl",
        presence_output=config.output_dir / "replay_presence.jsonl",
        quiet=quiet,
    )
    return {"predictions": prediction_count, "presence_events": event_count}


def _run_report(config: AppConfig, *, quiet: bool) -> dict[str, int]:
    del quiet
    metrics = read_json(config.data.metrics_path)
    write_markdown_report(config.output_dir / "report.md", config=config, metrics=metrics)
    return {"reports": 1}


def _write_presence_stream(
    rows: Any,
    model: Any,
    config: AppConfig,
    *,
    target_class: str,
    predictions_output: Path,
    presence_output: Path,
    quiet: bool,
) -> tuple[int, int]:
    del quiet
    engine = StreamingPresenceEngine(target_class=target_class, config=config.windowing)
    prediction_count = 0
    event_count = 0
    with JsonlRowWriter(predictions_output) as predictions, JsonlRowWriter(
        presence_output
    ) as presence:
        for row in rows:
            prediction = classify_row(
                row,
                model,
                config.features,
                label_mode=config.labeling.mode,
                packet_index=prediction_count,
            )
            predictions.write(prediction)
            prediction_count += 1
            for event in engine.process(prediction):
                presence.write(event)
                event_count += 1

        for event in engine.flush():
            presence.write(event)
            event_count += 1
    return prediction_count, event_count


def _write_eval_outputs(
    config: AppConfig,
    metrics: dict[str, Any],
    predictions: list[dict[str, Any]],
) -> None:
    write_json(config.data.metrics_path, metrics)
    write_rows(config.data.predictions_path, predictions)
    write_confusion_matrix_csv(config.output_dir / "confusion_matrix.csv", metrics)
    try:
        plot_confusion_matrix(config.output_dir / "confusion_matrix.png", metrics)
    except RuntimeError:
        pass
    try:
        plot_binary_curves(
            config.output_dir / "roc_curve.png",
            config.output_dir / "pr_curve.png",
            predictions,
            positive_label=config.labeling.positive_label,
        )
    except RuntimeError:
        pass
    write_markdown_report(config.output_dir / "report.md", config=config, metrics=metrics)


def _new_manifest(
    config: AppConfig,
    config_path: Path,
    options: RunOptions,
    stages: list[StageName],
) -> dict[str, Any]:
    return {
        "status": "running",
        "package_version": __version__,
        "config_path": str(config_path),
        "profile": options.profile,
        "from_stage": options.from_stage,
        "to_stage": options.to_stage,
        "force": options.force,
        "skip_existing": options.skip_existing,
        "dry_run": options.dry_run,
        "random_seed": config.random_seed,
        "started_at": _now(),
        "ended_at": None,
        "stage_order": stages,
        "stages": [],
        "artifacts": _artifact_paths(config),
    }


def _stage_result(
    name: str,
    status: str,
    *,
    outputs: list[Path] | None = None,
    row_counts: dict[str, int] | None = None,
    message: str | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "name": name,
        "status": status,
        "started_at": _now(),
        "ended_at": _now() if status in {"completed", "skipped", "failed"} else None,
        "outputs": [str(path) for path in outputs or []],
        "row_counts": row_counts or {},
    }
    if message is not None:
        result["message"] = message
    if error is not None:
        result["error"] = error
    return result


def _write_run_outputs(config: AppConfig, manifest: dict[str, Any]) -> None:
    write_json(config.output_dir / "run_manifest.json", manifest)
    write_run_summary(config.output_dir / "run_summary.md", manifest)


def _artifact_paths(config: AppConfig) -> dict[str, str]:
    return {
        "run_manifest": str(config.output_dir / "run_manifest.json"),
        "run_summary": str(config.output_dir / "run_summary.md"),
        "inspection_json": str(config.output_dir / "inspect.json"),
        "inspection_markdown": str(config.output_dir / "inspect.md"),
        "records": str(config.data.normalized_path),
        "features": str(config.data.features_path),
        "labels": str(config.data.labeled_path),
        "train": str(config.data.train_path),
        "test": str(config.data.test_path),
        "model": str(config.model.checkpoint_path),
        "metrics": str(config.data.metrics_path),
        "predictions": str(config.data.predictions_path),
        "rolling": str(config.data.rolling_path),
        "replay_predictions": str(config.output_dir / "replay_predictions.jsonl"),
        "replay_presence": str(config.output_dir / "replay_presence.jsonl"),
        "report": str(config.output_dir / "report.md"),
    }


def _preexisting_outputs(config: AppConfig, stages: list[StageName]) -> list[Path]:
    paths: list[Path] = []
    for stage in stages:
        for path in stage_outputs(config, stage):
            if path.exists() and path not in paths:
                paths.append(path)
    return paths


def _remove_outputs(paths: list[Path]) -> None:
    for path in paths:
        if path.exists() and path.is_file():
            path.unlink()


def _capture_paths_or_raise(config: AppConfig) -> list[Path]:
    capture_paths = expand_input_paths(config.input.paths)
    if capture_paths:
        return capture_paths
    path_list = ", ".join(str(path) for path in config.input.paths) or "<none>"
    raise ExperimentRunError(
        "No capture files found for input path(s): "
        f"{path_list}. Expected .pcap, .pcapng, or .cap files."
    )


def _target_registry(config: AppConfig) -> TargetRegistry:
    if config.target_registry_path is None:
        raise ExperimentRunError("target_registry_path is required")
    return TargetRegistry.from_file(config.target_registry_path)


def _target_registry_optional(config: AppConfig) -> TargetRegistry | None:
    if config.target_registry_path is None:
        return None
    return TargetRegistry.from_file(config.target_registry_path)


def _default_rolling_target_class(config: AppConfig) -> str:
    if config.labeling.mode in {"binary_one_vs_rest", "per_target_binary"}:
        return config.labeling.positive_label
    return config.labeling.target_id or config.labeling.positive_label


def _require_stage_in_profile(stage: str, stages: list[StageName], profile: str) -> None:
    if stage not in stages:
        raise ValueError(f"Stage {stage!r} is not part of profile {profile!r}")


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()
