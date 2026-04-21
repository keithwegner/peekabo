"""Presence threshold calibration over labeled predictions."""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import product
from pathlib import Path
from statistics import mean
from typing import Any

import yaml

from peekaboo import __version__
from peekaboo.config import AppConfig, CalibrationConfig, WindowConfig, load_config
from peekaboo.data.readers import iter_rows, read_all_rows
from peekaboo.data.writers import write_json
from peekaboo.features.extract import model_feature_names
from peekaboo.inference.aggregate import presence_state_from_stats
from peekaboo.inference.classify import classify_rows
from peekaboo.inference.presence import resolve_presence_target_classes
from peekaboo.models.base import load_checkpoint

ProgressCallback = Callable[[str], None]

OBJECTIVES = {"f1", "mcc", "precision", "recall"}
WINDOW_TYPES = {"frame_count", "time"}
CHART_METRICS = ("objective_score", "precision", "recall", "f1", "mcc")
RESULT_FIELDS = [
    "scope",
    "target_id",
    "window_type",
    "candidate_index",
    "present_ratio_threshold",
    "mean_probability_threshold",
    "max_probability_threshold",
    "window_count",
    "support",
    "positive_support",
    "truth_absent_windows",
    "true_positive",
    "false_positive",
    "true_negative",
    "false_negative",
    "precision",
    "recall",
    "f1",
    "mcc",
    "objective",
    "objective_score",
]


class CalibrationError(RuntimeError):
    """Raised when presence calibration cannot complete."""


@dataclass(frozen=True)
class ThresholdCandidate:
    candidate_index: int
    present_ratio_threshold: float
    mean_probability_threshold: float
    max_probability_threshold: float

    def window_config(self, base: WindowConfig) -> WindowConfig:
        return WindowConfig(
            frame_count=base.frame_count,
            time_seconds=base.time_seconds,
            min_frames=base.min_frames,
            present_ratio_threshold=self.present_ratio_threshold,
            mean_probability_threshold=self.mean_probability_threshold,
            max_probability_threshold=self.max_probability_threshold,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_index": self.candidate_index,
            "present_ratio_threshold": self.present_ratio_threshold,
            "mean_probability_threshold": self.mean_probability_threshold,
            "max_probability_threshold": self.max_probability_threshold,
        }


def calibration_artifact_paths(output_dir: str | Path) -> dict[str, Path | dict[str, Path]]:
    output_path = Path(output_dir)
    return {
        "manifest": output_path / "calibration_manifest.json",
        "results_csv": output_path / "calibration_results.csv",
        "results_json": output_path / "calibration_results.json",
        "report": output_path / "calibration_report.md",
        "recommended_windowing": output_path / "recommended_windowing.yaml",
        "charts": {
            metric: output_path / f"calibration_{metric}.png" for metric in CHART_METRICS
        },
    }


def run_presence_calibration(
    config_path: str | Path,
    *,
    input_predictions: str | Path | None = None,
    input_labeled: str | Path | None = None,
    output_dir: str | Path | None = None,
    target_classes: Iterable[str] | None = None,
    all_targets: bool | None = None,
    objective: str | None = None,
    force: bool = False,
    quiet: bool = False,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    config = load_config(config_path)
    selected_objective = _validate_objective(objective or config.calibration.objective)
    selected_output_dir = Path(output_dir or config.calibration.output_dir or "calibration")
    selected_targets = resolve_presence_target_classes(
        config,
        target_classes=target_classes,
        all_targets=all_targets,
    )
    artifacts = calibration_artifact_paths(selected_output_dir)
    output_paths = _flatten_artifact_paths(artifacts)

    if force:
        _remove_outputs(output_paths)
    else:
        conflicts = [path for path in output_paths if path.exists()]
        if conflicts:
            raise CalibrationError(
                "Calibration output files already exist; use --force to overwrite them: "
                + ", ".join(str(path) for path in conflicts)
            )

    manifest = _new_manifest(
        config=config,
        config_path=Path(config_path),
        output_dir=selected_output_dir,
        objective=selected_objective,
        target_classes=selected_targets,
    )

    try:
        if progress is not None and not quiet:
            progress("Loading calibration predictions")
        predictions, prediction_source = _load_calibration_predictions(
            config,
            input_predictions=Path(input_predictions) if input_predictions else None,
            input_labeled=Path(input_labeled) if input_labeled else None,
        )
        _require_ground_truth(predictions)
        if not predictions:
            raise CalibrationError("Calibration predictions are empty")

        candidates = threshold_candidates(config.windowing, config.calibration)
        window_stats = calibration_window_stats(
            predictions,
            target_classes=selected_targets,
            window_config=config.windowing,
            window_types=config.calibration.window_types,
            min_truth_frames=config.calibration.min_truth_frames,
        )
        if not window_stats:
            raise CalibrationError("No calibration windows could be created from the predictions")

        if progress is not None and not quiet:
            progress(
                f"Evaluating {len(candidates)} threshold candidate(s) "
                f"across {len(window_stats)} window(s)"
            )
        results = evaluate_calibration_candidates(
            window_stats,
            candidates=candidates,
            base_window_config=config.windowing,
            objective=selected_objective,
        )
        recommendation = recommend_thresholds(results, objective=selected_objective)
        warnings = calibration_warnings(
            config=config,
            predictions=predictions,
            window_stats=window_stats,
            recommendation=recommendation,
        )
        chart_paths = write_calibration_charts(selected_output_dir, results)

        manifest.update(
            {
                "status": "completed",
                "ended_at": _now(),
                "prediction_source": prediction_source,
                "prediction_rows": len(predictions),
                "window_count": len(window_stats),
                "candidate_count": len(candidates),
                "result_count": len(results),
                "warnings": warnings,
                "recommendation": recommendation,
                "artifacts": _manifest_artifacts(artifacts, chart_paths),
            }
        )
        _write_results_csv(Path(artifacts["results_csv"]), results)
        write_json(Path(artifacts["results_json"]), {"results": results})
        write_recommended_windowing(
            Path(artifacts["recommended_windowing"]),
            base=config.windowing,
            recommendation=recommendation,
        )
        write_calibration_report(
            Path(artifacts["report"]),
            manifest=manifest,
            results=results,
        )
        write_json(Path(artifacts["manifest"]), manifest)
        return manifest
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["ended_at"] = _now()
        manifest["error"] = str(exc)
        manifest["artifacts"] = _manifest_artifacts(artifacts, {})
        Path(artifacts["manifest"]).parent.mkdir(parents=True, exist_ok=True)
        write_json(Path(artifacts["manifest"]), manifest)
        if isinstance(exc, CalibrationError):
            raise
        raise CalibrationError(str(exc)) from exc


def threshold_candidates(
    window_config: WindowConfig,
    calibration_config: CalibrationConfig,
) -> list[ThresholdCandidate]:
    ratio_values = _with_current_value(
        calibration_config.present_ratio_thresholds,
        window_config.present_ratio_threshold,
    )
    mean_values = _with_current_value(
        calibration_config.mean_probability_thresholds,
        window_config.mean_probability_threshold,
    )
    max_values = _with_current_value(
        calibration_config.max_probability_thresholds,
        window_config.max_probability_threshold,
    )
    return [
        ThresholdCandidate(
            candidate_index=index,
            present_ratio_threshold=ratio,
            mean_probability_threshold=mean_threshold,
            max_probability_threshold=max_threshold,
        )
        for index, (ratio, mean_threshold, max_threshold) in enumerate(
            product(ratio_values, mean_values, max_values)
        )
    ]


def calibration_window_stats(
    predictions: list[dict[str, Any]],
    *,
    target_classes: Iterable[str],
    window_config: WindowConfig,
    window_types: Iterable[str],
    min_truth_frames: int,
) -> list[dict[str, Any]]:
    selected_window_types = list(dict.fromkeys(window_types))
    unknown = sorted(set(selected_window_types) - WINDOW_TYPES)
    if unknown:
        raise CalibrationError(f"Unsupported calibration window type(s): {', '.join(unknown)}")

    output: list[dict[str, Any]] = []
    for target_class in target_classes:
        if "frame_count" in selected_window_types:
            output.extend(
                _frame_count_window_stats(
                    predictions,
                    target_class=target_class,
                    config=window_config,
                    min_truth_frames=min_truth_frames,
                )
            )
        if "time" in selected_window_types:
            output.extend(
                _time_window_stats(
                    predictions,
                    target_class=target_class,
                    config=window_config,
                    min_truth_frames=min_truth_frames,
                )
            )
    return output


def evaluate_calibration_candidates(
    window_stats: list[dict[str, Any]],
    *,
    candidates: list[ThresholdCandidate],
    base_window_config: WindowConfig,
    objective: str,
) -> list[dict[str, Any]]:
    selected_objective = _validate_objective(objective)
    results: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for stat in window_stats:
        grouped.setdefault((str(stat["target_id"]), str(stat["window_type"])), []).append(stat)

    for candidate in candidates:
        config = candidate.window_config(base_window_config)
        candidate_rows = [
            _metric_row(
                windows,
                candidate=candidate,
                config=config,
                objective=selected_objective,
                target_id=target_id,
                window_type=window_type,
                scope="target_window",
            )
            for (target_id, window_type), windows in sorted(grouped.items())
        ]
        results.extend(candidate_rows)
        results.append(
            _metric_row(
                window_stats,
                candidate=candidate,
                config=config,
                objective=selected_objective,
                target_id="__all__",
                window_type="__all__",
                scope="global",
            )
        )
    return results


def recommend_thresholds(results: list[dict[str, Any]], *, objective: str) -> dict[str, Any]:
    selected_objective = _validate_objective(objective)
    global_rows = [row for row in results if row.get("scope") == "global"]
    if not global_rows:
        raise CalibrationError("No global calibration rows were produced")

    best = max(
        global_rows,
        key=lambda row: (
            _score(row.get("objective_score")),
            _score(row.get("mcc")),
            -int(row.get("false_positive") or 0),
            _score(row.get("recall")),
            -int(row.get("candidate_index") or 0),
        ),
    )
    return {
        "objective": selected_objective,
        "candidate_index": best["candidate_index"],
        "present_ratio_threshold": best["present_ratio_threshold"],
        "mean_probability_threshold": best["mean_probability_threshold"],
        "max_probability_threshold": best["max_probability_threshold"],
        "metrics": {field: best.get(field) for field in _metric_fields()},
    }


def write_recommended_windowing(
    path: str | Path,
    *,
    base: WindowConfig,
    recommendation: dict[str, Any],
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "windowing": {
            "frame_count": base.frame_count,
            "time_seconds": base.time_seconds,
            "min_frames": base.min_frames,
            "present_ratio_threshold": recommendation["present_ratio_threshold"],
            "mean_probability_threshold": recommendation["mean_probability_threshold"],
            "max_probability_threshold": recommendation["max_probability_threshold"],
        }
    }
    output.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def write_calibration_report(
    path: str | Path,
    *,
    manifest: dict[str, Any],
    results: list[dict[str, Any]],
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    recommendation = manifest.get("recommendation") or {}
    metrics = recommendation.get("metrics") or {}
    lines = [
        "# peekaboo Presence Calibration",
        "",
        "## Summary",
        "",
        f"- Status: `{manifest.get('status')}`",
        f"- Config: `{manifest.get('config_path')}`",
        f"- Objective: `{manifest.get('objective')}`",
        f"- Target classes: `{', '.join(manifest.get('target_classes', []))}`",
        f"- Prediction source: `{manifest.get('prediction_source')}`",
        f"- Prediction rows: `{manifest.get('prediction_rows', 0)}`",
        f"- Calibration windows: `{manifest.get('window_count', 0)}`",
        "",
        "Calibration uses existing metadata-only predictions and labels. It does not decrypt "
        "traffic, inspect payloads, inject frames, probe networks, configure adapters, or "
        "channel hop.",
        "",
        "## Recommendation",
        "",
        f"- Candidate: `{recommendation.get('candidate_index')}`",
        f"- Present ratio threshold: `{_fmt(recommendation.get('present_ratio_threshold'))}`",
        f"- Mean probability threshold: `{_fmt(recommendation.get('mean_probability_threshold'))}`",
        f"- Max probability threshold: `{_fmt(recommendation.get('max_probability_threshold'))}`",
        f"- Precision: `{_fmt(metrics.get('precision'))}`",
        f"- Recall: `{_fmt(metrics.get('recall'))}`",
        f"- F1: `{_fmt(metrics.get('f1'))}`",
        f"- MCC: `{_fmt(metrics.get('mcc'))}`",
        "",
    ]
    warnings = manifest.get("warnings") or []
    if warnings:
        lines.extend(["## Warnings", ""])
        lines.extend(f"- {warning}" for warning in warnings)
        lines.append("")

    lines.extend(
        [
            "## Top Global Candidates",
            "",
            (
                "| Candidate | Ratio | Mean Probability | Max Probability | Precision | "
                "Recall | F1 | MCC | FP | FN |"
            ),
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    global_rows = [row for row in results if row.get("scope") == "global"]
    sorted_rows = sorted(
        global_rows,
        key=lambda row: (
            _score(row.get("objective_score")),
            _score(row.get("mcc")),
            -int(row.get("false_positive") or 0),
            _score(row.get("recall")),
        ),
        reverse=True,
    )
    for row in sorted_rows[:10]:
        lines.append(
            f"| {row['candidate_index']} | {_fmt(row.get('present_ratio_threshold'))} | "
            f"{_fmt(row.get('mean_probability_threshold'))} | "
            f"{_fmt(row.get('max_probability_threshold'))} | "
            f"{_fmt(row.get('precision'))} | {_fmt(row.get('recall'))} | "
            f"{_fmt(row.get('f1'))} | {_fmt(row.get('mcc'))} | "
            f"{row.get('false_positive')} | {row.get('false_negative')} |"
        )

    artifacts = manifest.get("artifacts") or {}
    if artifacts:
        lines.extend(["", "## Artifacts", ""])
        for name, value in artifacts.items():
            if isinstance(value, dict):
                for nested_name, nested_value in value.items():
                    lines.append(f"- `{name}.{nested_name}`: `{nested_value}`")
            else:
                lines.append(f"- `{name}`: `{value}`")

    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_calibration_charts(
    output_dir: str | Path,
    results: list[dict[str, Any]],
) -> dict[str, Path]:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return {}

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    global_rows = sorted(
        [row for row in results if row.get("scope") == "global"],
        key=lambda row: int(row["candidate_index"]),
    )
    chart_paths: dict[str, Path] = {}
    for metric in CHART_METRICS:
        points = [
            (int(row["candidate_index"]), row.get(metric))
            for row in global_rows
            if isinstance(row.get(metric), int | float)
        ]
        if not points:
            continue
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot([point[0] for point in points], [float(point[1]) for point in points])
        ax.set_xlabel("Threshold candidate")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(f"Calibration {metric.replace('_', ' ').title()}")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        chart_path = output_path / f"calibration_{metric}.png"
        fig.savefig(chart_path)
        plt.close(fig)
        chart_paths[metric] = chart_path
    return chart_paths


def calibration_warnings(
    *,
    config: AppConfig,
    predictions: list[dict[str, Any]],
    window_stats: list[dict[str, Any]],
    recommendation: dict[str, Any],
) -> list[str]:
    warnings: list[str] = []
    if config.features.leakage_debug:
        warnings.append(
            "This run enabled leakage/debug features, so calibrated thresholds are not faithful "
            "to the default MAC-dropping pipeline."
        )
    if _looks_synthetic(config, predictions):
        warnings.append(
            "This appears to use synthetic data; treat calibration output as workflow smoke "
            "evidence, not real-world performance evidence."
        )
    truth_present = sum(1 for stat in window_stats if stat["truth_present"])
    total_windows = len(window_stats)
    if total_windows < 5:
        warnings.append("Very few windows were available; threshold recommendations may be sparse.")
    if total_windows and min(truth_present, total_windows - truth_present) / total_windows < 0.1:
        warnings.append(
            "Window truth labels are highly skewed; precision, recall, and MCC may be unstable."
        )
    metrics = recommendation.get("metrics") or {}
    if (metrics.get("false_positive") or 0) > 0 or (metrics.get("false_negative") or 0) > 0:
        warnings.append(
            "The recommended threshold still has calibration errors; inspect false positives and "
            "false negatives before using it for live alerts."
        )
    return warnings


def _frame_count_window_stats(
    rows: list[dict[str, Any]],
    *,
    target_class: str,
    config: WindowConfig,
    min_truth_frames: int,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for start in range(0, len(rows), config.frame_count):
        chunk = rows[start : start + config.frame_count]
        if chunk:
            output.append(
                _summarize_calibration_window(
                    chunk,
                    target_class=target_class,
                    window_type="frame_count",
                    window_start=start,
                    window_end=start + len(chunk) - 1,
                    min_truth_frames=min_truth_frames,
                )
            )
    return output


def _time_window_stats(
    rows: list[dict[str, Any]],
    *,
    target_class: str,
    config: WindowConfig,
    min_truth_frames: int,
) -> list[dict[str, Any]]:
    buckets: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        timestamp = row.get("timestamp")
        if timestamp is None:
            continue
        bucket = int(float(timestamp) // config.time_seconds)
        buckets.setdefault(bucket, []).append(row)

    output: list[dict[str, Any]] = []
    for bucket, chunk in sorted(buckets.items()):
        output.append(
            _summarize_calibration_window(
                chunk,
                target_class=target_class,
                window_type="time",
                window_start=bucket * config.time_seconds,
                window_end=(bucket + 1) * config.time_seconds,
                min_truth_frames=min_truth_frames,
            )
        )
    return output


def _summarize_calibration_window(
    rows: list[dict[str, Any]],
    *,
    target_class: str,
    window_type: str,
    window_start: float | int,
    window_end: float | int,
    min_truth_frames: int,
) -> dict[str, Any]:
    probabilities = [_target_probability(row, target_class) for row in rows]
    positive_predictions = [
        row for row in rows if str(row.get("predicted_class")) == target_class
    ]
    truth_count = sum(1 for row in rows if str(row.get("ground_truth")) == target_class)
    return {
        "target_id": target_class,
        "window_type": window_type,
        "window_start": window_start,
        "window_end": window_end,
        "frame_count": len(rows),
        "positive_prediction_ratio": len(positive_predictions) / len(rows) if rows else None,
        "mean_probability": mean(probabilities) if probabilities else None,
        "max_probability": max(probabilities) if probabilities else None,
        "truth_count": truth_count,
        "truth_present": truth_count >= min_truth_frames,
    }


def _metric_row(
    windows: list[dict[str, Any]],
    *,
    candidate: ThresholdCandidate,
    config: WindowConfig,
    objective: str,
    target_id: str,
    window_type: str,
    scope: str,
) -> dict[str, Any]:
    counts = _confusion_counts(windows, config)
    metrics = _binary_metrics(counts)
    row: dict[str, Any] = {
        "scope": scope,
        "target_id": target_id,
        "window_type": window_type,
        **candidate.to_dict(),
        **counts,
        **metrics,
        "objective": objective,
        "objective_score": metrics.get(objective),
    }
    return row


def _confusion_counts(
    windows: list[dict[str, Any]],
    config: WindowConfig,
) -> dict[str, int]:
    tp = fp = tn = fn = 0
    for stat in windows:
        state = presence_state_from_stats(
            int(stat["frame_count"]),
            stat.get("positive_prediction_ratio"),
            stat.get("mean_probability"),
            stat.get("max_probability"),
            config,
        )
        predicted_present = state == "present"
        truth_present = bool(stat["truth_present"])
        if predicted_present and truth_present:
            tp += 1
        elif predicted_present and not truth_present:
            fp += 1
        elif not predicted_present and truth_present:
            fn += 1
        else:
            tn += 1
    positive_support = tp + fn
    return {
        "window_count": len(windows),
        "support": len(windows),
        "positive_support": positive_support,
        "truth_absent_windows": tn + fp,
        "true_positive": tp,
        "false_positive": fp,
        "true_negative": tn,
        "false_negative": fn,
    }


def _binary_metrics(counts: dict[str, int]) -> dict[str, float | None]:
    tp = counts["true_positive"]
    fp = counts["false_positive"]
    tn = counts["true_negative"]
    fn = counts["false_negative"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn) - (fp * fn)) / denominator if denominator else None
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mcc": mcc,
    }


def _load_calibration_predictions(
    config: AppConfig,
    *,
    input_predictions: Path | None,
    input_labeled: Path | None,
) -> tuple[list[dict[str, Any]], str]:
    explicit_predictions = input_predictions is not None
    prediction_path = input_predictions or config.data.predictions_path
    if prediction_path.exists():
        rows = read_all_rows(prediction_path)
        if _has_ground_truth(rows):
            return rows, str(prediction_path)
        if explicit_predictions:
            raise CalibrationError(
                f"Prediction rows in {prediction_path} must include ground_truth for calibration"
            )
        if not config.calibration.prepare_predictions_if_missing:
            raise CalibrationError(
                f"Prediction rows in {prediction_path} lack ground_truth; enable "
                "calibration.prepare_predictions_if_missing or provide --input-labeled"
            )
    elif explicit_predictions:
        raise CalibrationError(f"Prediction input not found: {prediction_path}")
    elif not config.calibration.prepare_predictions_if_missing:
        raise CalibrationError(
            f"Prediction input not found: {prediction_path}; enable "
            "calibration.prepare_predictions_if_missing or provide --input-labeled"
        )

    labeled_path = input_labeled or config.data.labeled_path
    if not labeled_path.exists():
        raise CalibrationError(
            f"Labeled input not found for prediction preparation: {labeled_path}"
        )
    if not config.model.checkpoint_path.exists():
        raise CalibrationError(
            f"Model checkpoint not found for prediction preparation: {config.model.checkpoint_path}"
        )
    model = load_checkpoint(config.model.checkpoint_path)
    rows = classify_rows(
        iter_rows(labeled_path),
        model,
        config.features,
        label_mode=config.labeling.mode,
    )
    return rows, f"generated from {labeled_path}"


def _has_ground_truth(rows: list[dict[str, Any]]) -> bool:
    return bool(rows) and all(row.get("ground_truth") is not None for row in rows)


def _require_ground_truth(rows: list[dict[str, Any]]) -> None:
    if not _has_ground_truth(rows):
        raise CalibrationError("Calibration requires prediction rows with ground_truth labels")


def _target_probability(row: dict[str, Any], target_class: str) -> float:
    raw = row.get("probabilities_json")
    if raw:
        try:
            probabilities = json.loads(str(raw))
            if target_class in probabilities:
                return float(probabilities[target_class])
        except Exception:
            pass
    if str(row.get("predicted_class")) == target_class:
        return float(row.get("confidence") or 1.0)
    return 0.0


def _validate_objective(value: str) -> str:
    if value not in OBJECTIVES:
        raise CalibrationError(f"objective must be one of {', '.join(sorted(OBJECTIVES))}")
    return value


def _with_current_value(values: Iterable[float], current: float) -> list[float]:
    return sorted({round(float(value), 10) for value in [*values, current]})


def _score(value: Any) -> float:
    return float(value) if isinstance(value, int | float) else -1.0


def _metric_fields() -> list[str]:
    return [
        "window_count",
        "support",
        "positive_support",
        "truth_absent_windows",
        "true_positive",
        "false_positive",
        "true_negative",
        "false_negative",
        "precision",
        "recall",
        "f1",
        "mcc",
        "objective_score",
    ]


def _flatten_artifact_paths(artifacts: dict[str, Path | dict[str, Path]]) -> list[Path]:
    paths: list[Path] = []
    for value in artifacts.values():
        if isinstance(value, dict):
            paths.extend(value.values())
        else:
            paths.append(value)
    return paths


def _remove_outputs(paths: Iterable[Path]) -> None:
    for path in paths:
        if path.exists():
            path.unlink()


def _manifest_artifacts(
    artifacts: dict[str, Path | dict[str, Path]],
    chart_paths: dict[str, Path],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for name, value in artifacts.items():
        if name == "charts":
            output[name] = {metric: str(path) for metric, path in chart_paths.items()}
        elif isinstance(value, dict):
            output[name] = {nested: str(path) for nested, path in value.items()}
        else:
            output[name] = str(value)
    return output


def _new_manifest(
    *,
    config: AppConfig,
    config_path: Path,
    output_dir: Path,
    objective: str,
    target_classes: list[str],
) -> dict[str, Any]:
    return {
        "status": "running",
        "created_at": _now(),
        "ended_at": None,
        "config_path": str(config_path),
        "output_dir": str(output_dir),
        "package_version": __version__,
        "random_seed": config.random_seed,
        "objective": objective,
        "target_classes": target_classes,
        "window_types": list(config.calibration.window_types),
        "min_truth_frames": config.calibration.min_truth_frames,
        "feature_names": model_feature_names(config.features),
        "leakage_debug": config.features.leakage_debug,
        "label_mode": config.labeling.mode,
        "windowing": {
            "frame_count": config.windowing.frame_count,
            "time_seconds": config.windowing.time_seconds,
            "min_frames": config.windowing.min_frames,
            "present_ratio_threshold": config.windowing.present_ratio_threshold,
            "mean_probability_threshold": config.windowing.mean_probability_threshold,
            "max_probability_threshold": config.windowing.max_probability_threshold,
        },
    }


def _write_results_csv(path: Path, results: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)


def _looks_synthetic(config: AppConfig, predictions: list[dict[str, Any]]) -> bool:
    text_parts = [
        str(config.output_dir),
        *(str(path) for path in config.input.paths),
        *(str(row.get("source_file") or "") for row in predictions[:20]),
    ]
    return any("synthetic" in value.lower() for value in text_parts)


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()
