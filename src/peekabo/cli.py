"""Typer CLI for peekabo."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from peekabo.capture.live import iter_live_records
from peekabo.capture.sources import expand_input_paths, iter_packet_records
from peekabo.config import AppConfig, load_config, write_run_config
from peekabo.data.readers import iter_rows
from peekabo.data.sampling import balance_file, random_sample_file
from peekabo.data.splits import split_file
from peekabo.data.writers import write_json, write_rows
from peekabo.evaluation.holdout import evaluate_holdout_rows, train_online_rows
from peekabo.evaluation.plots import (
    plot_binary_curves,
    plot_confusion_matrix,
    write_confusion_matrix_csv,
)
from peekabo.evaluation.prequential import evaluate_prequential_rows
from peekabo.evaluation.reports import write_markdown_report
from peekabo.features.extract import iter_feature_rows, model_feature_names
from peekabo.features.selection import rank_feature_file
from peekabo.inference.aggregate import rolling_aggregates
from peekabo.inference.classify import classify_rows
from peekabo.labeling.filters import passes_filters
from peekabo.labeling.labelers import iter_labeled_rows
from peekabo.labeling.targets import TargetRegistry
from peekabo.models.base import load_checkpoint
from peekabo.models.registry import create_model

app = typer.Typer(no_args_is_help=True, help="Passive 802.11 device identification.")

ConfigOption = Annotated[
    Path, typer.Option("--config", "-c", exists=True, help="YAML config path.")
]


@app.command()
def ingest(
    config: ConfigOption,
    input_path: Annotated[list[Path] | None, typer.Option("--input", "-i")] = None,
    output: Annotated[Path | None, typer.Option("--output", "-o")] = None,
    allow_empty: Annotated[
        bool,
        typer.Option(
            "--allow-empty",
            help="Write an empty dataset when no capture files are found.",
        ),
    ] = False,
) -> None:
    cfg = load_config(config)
    paths = input_path or cfg.input.paths
    capture_paths = expand_input_paths(paths)
    if not capture_paths:
        path_list = ", ".join(str(path) for path in paths) or "<none>"
        message = (
            "No capture files found for input path(s): "
            f"{path_list}. Expected .pcap, .pcapng, or .cap files."
        )
        if not allow_empty:
            typer.secho(f"{message} Use --allow-empty to write an empty dataset.", err=True)
            raise typer.Exit(1)
        typer.secho(f"{message} Writing empty dataset because --allow-empty was set.", err=True)

    output_path = output or cfg.data.normalized_path
    _write_run_config(cfg)
    count = write_rows(
        output_path,
        (record.to_dict() for record in iter_packet_records(capture_paths)),
    )
    typer.echo(f"Wrote {count} normalized packet records to {output_path}")


@app.command("features")
def features_command(
    config: ConfigOption,
    input_path: Annotated[Path | None, typer.Option("--input", "-i")] = None,
    output: Annotated[Path | None, typer.Option("--output", "-o")] = None,
) -> None:
    cfg = load_config(config)
    source = input_path or cfg.data.normalized_path
    output_path = output or cfg.data.features_path
    rows = (row for row in iter_rows(source) if row.get("parse_ok", True))
    count = write_rows(output_path, iter_feature_rows(rows, cfg.features))
    typer.echo(f"Wrote {count} feature rows to {output_path}")


@app.command()
def label(
    config: ConfigOption,
    input_path: Annotated[Path | None, typer.Option("--input", "-i")] = None,
    output: Annotated[Path | None, typer.Option("--output", "-o")] = None,
) -> None:
    cfg = load_config(config)
    registry = _target_registry(cfg)
    source = input_path or cfg.data.features_path
    output_path = output or cfg.data.labeled_path

    def filtered_rows():
        for row in iter_rows(source):
            if passes_filters(row, cfg.filters, registry=registry):
                yield row

    count = write_rows(output_path, iter_labeled_rows(filtered_rows(), registry, cfg.labeling))
    typer.echo(f"Wrote {count} labeled rows to {output_path}")


@app.command()
def sample(
    config: ConfigOption,
    input_path: Annotated[Path | None, typer.Option("--input", "-i")] = None,
    output: Annotated[Path | None, typer.Option("--output", "-o")] = None,
    percentage: Annotated[float | None, typer.Option("--percentage")] = None,
    balance_strategy: Annotated[str | None, typer.Option("--balance-strategy")] = None,
) -> None:
    cfg = load_config(config)
    source = input_path or cfg.data.labeled_path
    output_path = output or cfg.data.labeled_path.with_name(
        "sampled" + cfg.data.labeled_path.suffix
    )
    strategy = balance_strategy or cfg.sampling.balance_strategy
    if strategy and strategy.lower() not in {"none", "off"}:
        count = balance_file(
            source,
            output_path,
            strategy=strategy,
            seed=cfg.random_seed,
            class_weight_column=cfg.sampling.class_weight_column,
        )
    else:
        count = random_sample_file(
            source,
            output_path,
            percentage=percentage if percentage is not None else cfg.sampling.percentage,
            seed=cfg.random_seed,
        )
    typer.echo(f"Wrote {count} sampled rows to {output_path}")


@app.command()
def split(
    config: ConfigOption,
    input_path: Annotated[Path | None, typer.Option("--input", "-i")] = None,
    train_output: Annotated[Path | None, typer.Option("--train-output")] = None,
    test_output: Annotated[Path | None, typer.Option("--test-output")] = None,
) -> None:
    cfg = load_config(config)
    train_count, test_count = split_file(
        input_path or cfg.data.labeled_path,
        train_output or cfg.data.train_path,
        test_output or cfg.data.test_path,
        train_fraction=cfg.split.train_fraction,
        chronological=cfg.split.chronological,
        seed=cfg.random_seed,
    )
    typer.echo(f"Wrote {train_count} train rows and {test_count} test rows")


@app.command("feature-rank")
def feature_rank(
    config: ConfigOption,
    input_path: Annotated[Path | None, typer.Option("--input", "-i")] = None,
    output_json: Annotated[Path | None, typer.Option("--output-json")] = None,
    output_markdown: Annotated[Path | None, typer.Option("--output-markdown")] = None,
    sample_size: Annotated[int | None, typer.Option("--sample-size")] = None,
) -> None:
    cfg = load_config(config)
    out_json = output_json or cfg.output_dir / "feature_ranking.json"
    out_md = output_markdown or cfg.output_dir / "feature_ranking.md"
    ranked = rank_feature_file(
        input_path or cfg.data.labeled_path,
        features=model_feature_names(cfg.features),
        sample_size=sample_size,
        seed=cfg.random_seed,
        output_json=out_json,
        output_markdown=out_md,
    )
    typer.echo(f"Ranked {len(ranked)} features")


@app.command("train-online")
def train_online(
    config: ConfigOption,
    input_path: Annotated[Path | None, typer.Option("--input", "-i")] = None,
    checkpoint: Annotated[Path | None, typer.Option("--checkpoint")] = None,
    model_id: Annotated[str | None, typer.Option("--model-id")] = None,
) -> None:
    cfg = load_config(config)
    model = create_model(
        model_id or cfg.model.model_id,
        feature_names=model_feature_names(cfg.features),
        seed=cfg.random_seed,
        params=cfg.model.params,
    )
    count = train_online_rows(
        iter_rows(input_path or cfg.data.labeled_path),
        model,
        cfg.features,
        weight_column=cfg.sampling.class_weight_column,
    )
    model.save(checkpoint or cfg.model.checkpoint_path)
    typer.echo(f"Trained {model.model_id} on {count} rows")


@app.command("eval-prequential")
def eval_prequential(
    config: ConfigOption,
    input_path: Annotated[Path | None, typer.Option("--input", "-i")] = None,
    predictions_output: Annotated[Path | None, typer.Option("--predictions-output")] = None,
    metrics_output: Annotated[Path | None, typer.Option("--metrics-output")] = None,
    model_id: Annotated[str | None, typer.Option("--model-id")] = None,
) -> None:
    cfg = load_config(config)
    model = create_model(
        model_id or cfg.model.model_id,
        feature_names=model_feature_names(cfg.features),
        seed=cfg.random_seed,
        params=cfg.model.params,
    )
    metrics, predictions = evaluate_prequential_rows(
        iter_rows(input_path or cfg.data.labeled_path),
        model,
        cfg.features,
        positive_label=cfg.labeling.positive_label,
        weight_column=cfg.sampling.class_weight_column,
    )
    _write_eval_outputs(cfg, metrics, predictions, metrics_output, predictions_output)
    typer.echo(f"Evaluated {metrics['n_examples']} rows prequentially")


@app.command("eval-holdout")
def eval_holdout(
    config: ConfigOption,
    train_input: Annotated[Path | None, typer.Option("--train-input")] = None,
    test_input: Annotated[Path | None, typer.Option("--test-input")] = None,
    predictions_output: Annotated[Path | None, typer.Option("--predictions-output")] = None,
    metrics_output: Annotated[Path | None, typer.Option("--metrics-output")] = None,
    model_id: Annotated[str | None, typer.Option("--model-id")] = None,
) -> None:
    cfg = load_config(config)
    model = create_model(
        model_id or cfg.model.model_id,
        feature_names=model_feature_names(cfg.features),
        seed=cfg.random_seed,
        params=cfg.model.params,
    )
    metrics, predictions = evaluate_holdout_rows(
        iter_rows(train_input or cfg.data.train_path),
        iter_rows(test_input or cfg.data.test_path),
        model,
        cfg.features,
        positive_label=cfg.labeling.positive_label,
        weight_column=cfg.sampling.class_weight_column,
    )
    _write_eval_outputs(cfg, metrics, predictions, metrics_output, predictions_output)
    typer.echo(f"Evaluated {metrics['n_examples']} holdout rows")


@app.command("classify-file")
def classify_file(
    config: ConfigOption,
    input_path: Annotated[Path | None, typer.Option("--input", "-i")] = None,
    checkpoint: Annotated[Path | None, typer.Option("--checkpoint")] = None,
    predictions_output: Annotated[Path | None, typer.Option("--predictions-output")] = None,
    rolling_output: Annotated[Path | None, typer.Option("--rolling-output")] = None,
    target_class: Annotated[str | None, typer.Option("--target-class")] = None,
) -> None:
    cfg = load_config(config)
    model = load_checkpoint(checkpoint or cfg.model.checkpoint_path)
    predictions = classify_rows(
        iter_rows(input_path or cfg.data.features_path),
        model,
        cfg.features,
        label_mode=cfg.labeling.mode,
        positive_label=cfg.labeling.positive_label,
    )
    pred_path = predictions_output or cfg.data.predictions_path
    roll_path = rolling_output or cfg.data.rolling_path
    write_rows(pred_path, predictions)
    target = target_class or cfg.labeling.target_id or cfg.labeling.positive_label
    rolling = rolling_aggregates(predictions, target_class=target, config=cfg.windowing)
    write_rows(roll_path, rolling)
    typer.echo(f"Wrote {len(predictions)} predictions and {len(rolling)} rolling rows")


@app.command("classify-live")
def classify_live(
    config: ConfigOption,
    interface: Annotated[str | None, typer.Option("--interface")] = None,
    checkpoint: Annotated[Path | None, typer.Option("--checkpoint")] = None,
    output: Annotated[Path | None, typer.Option("--output", "-o")] = None,
    max_packets: Annotated[int | None, typer.Option("--max-packets")] = None,
    timeout_seconds: Annotated[float | None, typer.Option("--timeout-seconds")] = None,
) -> None:
    cfg = load_config(config)
    iface = interface or cfg.input.live_interface
    if not iface:
        raise typer.BadParameter("A monitor-mode interface is required")
    model = load_checkpoint(checkpoint or cfg.model.checkpoint_path)
    feature_rows = iter_feature_rows(
        iter_live_records(iface, timeout_seconds=timeout_seconds, max_packets=max_packets),
        cfg.features,
    )
    predictions = classify_rows(feature_rows, model, cfg.features, label_mode=cfg.labeling.mode)
    count = write_rows(output or cfg.data.predictions_path, predictions)
    typer.echo(f"Wrote {count} live predictions")


@app.command()
def report(
    config: ConfigOption,
    metrics_input: Annotated[Path | None, typer.Option("--metrics-input")] = None,
    output: Annotated[Path | None, typer.Option("--output", "-o")] = None,
) -> None:
    import json

    cfg = load_config(config)
    metrics_path = metrics_input or cfg.data.metrics_path
    with Path(metrics_path).open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    report_path = output or cfg.output_dir / "report.md"
    write_markdown_report(report_path, config=cfg, metrics=metrics)
    typer.echo(f"Wrote report to {report_path}")


def _target_registry(cfg: AppConfig) -> TargetRegistry:
    if cfg.target_registry_path is None:
        raise typer.BadParameter("target_registry_path is required")
    return TargetRegistry.from_file(cfg.target_registry_path)


def _write_run_config(cfg: AppConfig) -> None:
    write_run_config(cfg, cfg.output_dir)


def _write_eval_outputs(
    cfg: AppConfig,
    metrics: dict,
    predictions: list[dict],
    metrics_output: Path | None,
    predictions_output: Path | None,
) -> None:
    metrics_path = metrics_output or cfg.data.metrics_path
    predictions_path = predictions_output or cfg.data.predictions_path
    write_json(metrics_path, metrics)
    write_rows(predictions_path, predictions)
    write_confusion_matrix_csv(cfg.output_dir / "confusion_matrix.csv", metrics)
    try:
        plot_confusion_matrix(cfg.output_dir / "confusion_matrix.png", metrics)
    except RuntimeError:
        pass
    try:
        plot_binary_curves(
            cfg.output_dir / "roc_curve.png",
            cfg.output_dir / "pr_curve.png",
            predictions,
            positive_label=cfg.labeling.positive_label,
        )
    except RuntimeError:
        pass
    write_markdown_report(cfg.output_dir / "report.md", config=cfg, metrics=metrics)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
