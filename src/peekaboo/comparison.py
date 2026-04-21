"""Paper-style aggregate experiment comparison."""

from __future__ import annotations

import csv
import random
import time
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from peekaboo import __version__
from peekaboo.config import AppConfig, FeatureConfig, load_config
from peekaboo.data.readers import iter_rows, read_all_rows
from peekaboo.data.writers import write_json
from peekaboo.evaluation.holdout import train_online_rows
from peekaboo.evaluation.metrics import classification_metrics
from peekaboo.features.extract import model_feature_names, row_to_model_features
from peekaboo.models.base import OnlineModel
from peekaboo.models.registry import MODEL_MAPPINGS, create_model
from peekaboo.runner import ExperimentRunError, run_experiment

ProgressCallback = Callable[[str], None]

METRIC_CHARTS = ("accuracy", "f1", "mcc", "roc_auc", "pr_auc")
RESULT_FIELDS = [
    "model_id",
    "train_fraction",
    "train_rows",
    "test_rows",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "mcc",
    "roc_auc",
    "pr_auc",
    "model_mapping",
    "elapsed_seconds",
]


class ComparisonError(RuntimeError):
    """Raised when an experiment comparison cannot complete."""


@dataclass(frozen=True)
class ComparisonRunPlan:
    model_id: str
    train_fraction: float


def plan_comparisons(
    *,
    models: Iterable[str],
    train_fractions: Iterable[float],
) -> list[ComparisonRunPlan]:
    selected_models = _validate_models(list(models))
    selected_fractions = _validate_train_fractions(list(train_fractions))
    return [
        ComparisonRunPlan(model_id=model_id, train_fraction=train_fraction)
        for model_id in selected_models
        for train_fraction in selected_fractions
    ]


def comparison_artifact_paths(output_dir: str | Path) -> dict[str, Path | dict[str, Path]]:
    output_path = Path(output_dir)
    return {
        "manifest": output_path / "comparison_manifest.json",
        "results_csv": output_path / "comparison_results.csv",
        "results_json": output_path / "comparison_results.json",
        "report": output_path / "comparison_report.md",
        "charts": {
            metric: output_path / f"comparison_{metric}.png" for metric in METRIC_CHARTS
        },
    }


def run_comparison(
    config_path: str | Path,
    *,
    models: Iterable[str] | None = None,
    train_fractions: Iterable[float] | None = None,
    output_dir: str | Path | None = None,
    prepare_if_missing: bool | None = None,
    force: bool = False,
    quiet: bool = False,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    config = load_config(config_path)
    selected_models = _validate_models(list(models or config.comparison.models))
    selected_fractions = _validate_train_fractions(
        list(train_fractions or config.comparison.train_fractions)
    )
    selected_output_dir = Path(output_dir or config.comparison.output_dir or "comparison")
    prepare = (
        config.comparison.prepare_if_missing if prepare_if_missing is None else prepare_if_missing
    )
    artifacts = comparison_artifact_paths(selected_output_dir)
    output_paths = _flatten_artifact_paths(artifacts)

    if force:
        _remove_outputs(output_paths)
    else:
        conflicts = [path for path in output_paths if path.exists()]
        if conflicts:
            raise ComparisonError(
                "Comparison output files already exist; use --force to overwrite them: "
                + ", ".join(str(path) for path in conflicts)
            )

    manifest = _new_manifest(
        config=config,
        config_path=Path(config_path),
        output_dir=selected_output_dir,
        models=selected_models,
        train_fractions=selected_fractions,
        prepare_if_missing=prepare,
    )

    try:
        _ensure_labeled_dataset(
            config_path=Path(config_path),
            config=config,
            prepare_if_missing=prepare,
            quiet=quiet,
            progress=progress,
        )
        total_rows = sum(1 for _ in iter_rows(config.data.labeled_path))
        if total_rows == 0:
            raise ComparisonError(f"Labeled dataset {config.data.labeled_path} has no rows")

        results: list[dict[str, Any]] = []
        feature_names = model_feature_names(config.features)
        manifest["feature_names"] = feature_names
        manifest["labeled_rows"] = total_rows
        for item in plan_comparisons(models=selected_models, train_fractions=selected_fractions):
            if progress is not None and not quiet:
                progress(
                    f"Comparing {item.model_id} at train_fraction={item.train_fraction:g}"
                )
            row = _run_one_comparison(
                config,
                item,
                total_rows=total_rows,
                feature_names=feature_names,
            )
            results.append(row)

        chart_paths = write_comparison_charts(selected_output_dir, results)
        manifest["status"] = "completed"
        manifest["ended_at"] = _now()
        manifest["result_count"] = len(results)
        manifest["artifacts"] = _manifest_artifacts(artifacts, chart_paths)

        _write_results_csv(Path(artifacts["results_csv"]), results)
        write_json(Path(artifacts["results_json"]), {"results": results})
        write_comparison_report(Path(artifacts["report"]), manifest=manifest, results=results)
        write_json(Path(artifacts["manifest"]), manifest)
        return manifest
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["ended_at"] = _now()
        manifest["error"] = str(exc)
        manifest["artifacts"] = _manifest_artifacts(artifacts, {})
        write_json(Path(artifacts["manifest"]), manifest)
        if isinstance(exc, ComparisonError):
            raise
        raise ComparisonError(str(exc)) from exc


def write_comparison_charts(
    output_dir: str | Path,
    results: list[dict[str, Any]],
) -> dict[str, Path]:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return {}

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    chart_paths: dict[str, Path] = {}
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        by_model[str(row["model_id"])].append(row)

    for metric in METRIC_CHARTS:
        plotted = False
        fig, ax = plt.subplots(figsize=(6, 4))
        for model_id, rows in sorted(by_model.items()):
            points = [
                (float(row["train_fraction"]), row.get(metric))
                for row in sorted(rows, key=lambda item: float(item["train_fraction"]))
                if isinstance(row.get(metric), int | float)
            ]
            if not points:
                continue
            x_values = [point[0] for point in points]
            y_values = [float(point[1]) for point in points]
            ax.plot(x_values, y_values, marker="o", label=model_id)
            plotted = True
        if not plotted:
            plt.close(fig)
            continue
        ax.set_xlabel("Training fraction")
        ax.set_ylabel(metric.replace("_", " ").upper())
        ax.set_title(f"{metric.replace('_', ' ').title()} by Training Fraction")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        chart_path = output_path / f"comparison_{metric}.png"
        fig.savefig(chart_path)
        plt.close(fig)
        chart_paths[metric] = chart_path
    return chart_paths


def write_comparison_report(
    path: str | Path,
    *,
    manifest: dict[str, Any],
    results: list[dict[str, Any]],
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# peekaboo Experiment Comparison",
        "",
        "## Summary",
        "",
        f"- Status: `{manifest.get('status')}`",
        f"- Config: `{manifest.get('config_path')}`",
        f"- Random seed: `{manifest.get('random_seed')}`",
        f"- Labeled rows: `{manifest.get('labeled_rows', 0)}`",
        f"- Split mode: `{'chronological' if manifest.get('chronological_split') else 'shuffled'}`",
        f"- Models: `{', '.join(manifest.get('models', []))}`",
        (
            "- Training fractions: "
            f"`{', '.join(str(item) for item in manifest.get('train_fractions', []))}`"
        ),
        "",
        "Synthetic-demo comparison results are smoke evidence only; they are not real-world "
        "wireless identification performance evidence.",
        "",
    ]
    if manifest.get("leakage_debug"):
        lines.extend(
            [
                "## Leakage Debug Warning",
                "",
                (
                    "This comparison included MAC-derived model features and should not be treated "
                    "as a faithful reproduction run."
                ),
                "",
            ]
        )

    lines.extend(
        [
            "## Results",
            "",
            (
                "| Model | Train Fraction | Train Rows | Test Rows | Accuracy | Precision | "
                "Recall | F1 | MCC | ROC AUC | PR AUC | Seconds |"
            ),
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in results:
        lines.append(
            f"| `{row['model_id']}` | {row['train_fraction']:.2f} | "
            f"{row['train_rows']} | {row['test_rows']} | {_fmt(row.get('accuracy'))} | "
            f"{_fmt(row.get('precision'))} | {_fmt(row.get('recall'))} | "
            f"{_fmt(row.get('f1'))} | {_fmt(row.get('mcc'))} | "
            f"{_fmt(row.get('roc_auc'))} | {_fmt(row.get('pr_auc'))} | "
            f"{_fmt(row.get('elapsed_seconds'))} |"
        )

    lines.extend(["", "## Model Mapping", ""])
    for model_id in manifest.get("models", []):
        lines.append(f"- `{model_id}`: `{MODEL_MAPPINGS[model_id]}`")

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


def _run_one_comparison(
    config: AppConfig,
    item: ComparisonRunPlan,
    *,
    total_rows: int,
    feature_names: list[str],
) -> dict[str, Any]:
    model = create_model(
        item.model_id,
        feature_names=feature_names,
        seed=config.random_seed,
        params=config.model.params,
    )
    train_rows, test_rows, expected_train, expected_test = _split_rows_for_fraction(
        config,
        total_rows=total_rows,
        train_fraction=item.train_fraction,
    )
    started = time.perf_counter()
    metrics = _evaluate_holdout_metrics(
        train_rows,
        test_rows,
        model,
        config.features,
        positive_label=config.labeling.positive_label,
        weight_column=config.sampling.class_weight_column,
    )
    elapsed = time.perf_counter() - started
    train_examples = metrics.get("train_examples")
    test_examples = metrics.get("n_examples")
    return {
        "model_id": item.model_id,
        "train_fraction": item.train_fraction,
        "train_rows": int(expected_train if train_examples is None else train_examples),
        "test_rows": int(expected_test if test_examples is None else test_examples),
        "accuracy": metrics.get("accuracy"),
        "precision": metrics.get("precision"),
        "recall": metrics.get("recall"),
        "f1": metrics.get("f1"),
        "mcc": metrics.get("mcc"),
        "roc_auc": metrics.get("roc_auc"),
        "pr_auc": metrics.get("pr_auc"),
        "model_mapping": MODEL_MAPPINGS[item.model_id],
        "elapsed_seconds": round(elapsed, 6),
    }


def _evaluate_holdout_metrics(
    train_rows: Iterable[dict[str, Any]],
    test_rows: Iterable[dict[str, Any]],
    model: OnlineModel,
    feature_config: FeatureConfig,
    *,
    positive_label: str,
    weight_column: str | None,
) -> dict[str, Any]:
    train_count = train_online_rows(
        train_rows,
        model,
        feature_config,
        weight_column=weight_column,
    )
    y_true: list[str] = []
    y_pred: list[str] = []
    y_score: list[float] = []
    for row in test_rows:
        label = row.get("label")
        if label is None:
            continue
        features = row_to_model_features(row, feature_config)
        probabilities = model.predict_proba_one(features)
        prediction = model.predict_one(features)
        if prediction is None and probabilities:
            prediction = max(probabilities, key=probabilities.get)
        prediction = prediction or "__none__"
        y_true.append(str(label))
        y_pred.append(str(prediction))
        y_score.append(float(probabilities.get(positive_label, 0.0)))

    metrics = classification_metrics(y_true, y_pred, y_score=y_score, positive_label=positive_label)
    metrics["train_examples"] = train_count
    return metrics


def _split_rows_for_fraction(
    config: AppConfig,
    *,
    total_rows: int,
    train_fraction: float,
) -> tuple[Iterable[dict[str, Any]], Iterable[dict[str, Any]], int, int]:
    train_count = int(total_rows * train_fraction)
    if config.split.chronological:

        def train_rows() -> Iterator[dict[str, Any]]:
            for index, row in enumerate(iter_rows(config.data.labeled_path)):
                if index < train_count:
                    yield row

        def test_rows() -> Iterator[dict[str, Any]]:
            for index, row in enumerate(iter_rows(config.data.labeled_path)):
                if index >= train_count:
                    yield row

        return train_rows(), test_rows(), train_count, total_rows - train_count

    rows = read_all_rows(config.data.labeled_path)
    rng = random.Random(config.random_seed)
    rng.shuffle(rows)
    return rows[:train_count], rows[train_count:], train_count, total_rows - train_count


def _ensure_labeled_dataset(
    *,
    config_path: Path,
    config: AppConfig,
    prepare_if_missing: bool,
    quiet: bool,
    progress: ProgressCallback | None,
) -> None:
    if config.data.labeled_path.exists():
        return
    if not prepare_if_missing:
        raise ComparisonError(
            f"Labeled dataset {config.data.labeled_path} does not exist; "
            "run `peekaboo run --profile prepare` first or pass --prepare."
        )
    if progress is not None and not quiet:
        progress("Preparing labeled dataset through label stage")
    try:
        run_experiment(
            config_path,
            profile="full",
            to_stage="label",
            skip_existing=True,
            quiet=quiet,
            progress=progress,
        )
    except ExperimentRunError as exc:
        raise ComparisonError(f"Preparing labeled dataset failed: {exc}") from exc
    if not config.data.labeled_path.exists():
        raise ComparisonError(
            f"Preparing labeled dataset did not create {config.data.labeled_path}"
        )


def _write_results_csv(path: Path, results: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in results:
            writer.writerow({field: _csv_value(row.get(field)) for field in RESULT_FIELDS})


def _csv_value(value: Any) -> Any:
    return "" if value is None else value


def _validate_models(value: list[str]) -> list[str]:
    unknown = sorted(set(value) - set(MODEL_MAPPINGS))
    if unknown:
        raise ComparisonError(f"Unsupported comparison model id(s): {', '.join(unknown)}")
    if not value:
        raise ComparisonError("At least one comparison model is required")
    return value


def _validate_train_fractions(value: list[float]) -> list[float]:
    if not value:
        raise ComparisonError("At least one training fraction is required")
    invalid = [fraction for fraction in value if fraction <= 0 or fraction >= 1]
    if invalid:
        raise ComparisonError("Training fractions must satisfy 0 < fraction < 1")
    return value


def _flatten_artifact_paths(artifacts: dict[str, Path | dict[str, Path]]) -> list[Path]:
    paths: list[Path] = []
    for value in artifacts.values():
        if isinstance(value, dict):
            paths.extend(value.values())
        else:
            paths.append(value)
    return paths


def _manifest_artifacts(
    artifacts: dict[str, Path | dict[str, Path]],
    generated_charts: dict[str, Path],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, value in artifacts.items():
        if name == "charts":
            result[name] = {metric: str(path) for metric, path in generated_charts.items()}
        elif isinstance(value, Path):
            result[name] = str(value)
    return result


def _remove_outputs(paths: Iterable[Path]) -> None:
    for path in paths:
        if path.exists() and path.is_file():
            path.unlink()


def _new_manifest(
    *,
    config: AppConfig,
    config_path: Path,
    output_dir: Path,
    models: list[str],
    train_fractions: list[float],
    prepare_if_missing: bool,
) -> dict[str, Any]:
    return {
        "status": "running",
        "package_version": __version__,
        "config_path": str(config_path),
        "random_seed": config.random_seed,
        "labeled_path": str(config.data.labeled_path),
        "output_dir": str(output_dir),
        "models": models,
        "train_fractions": train_fractions,
        "chronological_split": config.split.chronological,
        "prepare_if_missing": prepare_if_missing,
        "leakage_debug": config.features.leakage_debug,
        "started_at": _now(),
        "ended_at": None,
        "feature_names": [],
        "labeled_rows": 0,
        "result_count": 0,
        "artifacts": {},
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()
