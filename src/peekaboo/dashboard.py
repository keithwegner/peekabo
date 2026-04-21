"""Static HTML dashboard generation for completed runs."""

from __future__ import annotations

import base64
import html
import os
from collections import Counter
from pathlib import Path
from typing import Any

from peekaboo.config import AppConfig, load_config
from peekaboo.data.readers import iter_rows, read_json

DASHBOARD_TITLE = "peekaboo Run Dashboard"
ARTIFACT_SUFFIXES = {".json", ".md", ".csv", ".parquet", ".jsonl", ".png", ".yaml", ".yml"}
STATE_COLORS = {
    "present": "#159447",
    "absent": "#7a8699",
    "uncertain": "#d48b00",
}


class DashboardError(RuntimeError):
    """Raised when dashboard generation cannot complete."""


def generate_dashboard(
    config_path: str | Path,
    *,
    output: str | Path | None = None,
    force: bool = False,
    quiet: bool = False,
) -> dict[str, Any]:
    del quiet
    config = load_config(config_path)
    output_path = Path(output or config.output_dir / "dashboard" / "index.html")
    if output_path.exists() and not force:
        raise DashboardError(f"Dashboard output already exists; use --force: {output_path}")

    data = collect_dashboard_data(config, config_path=Path(config_path), output_path=output_path)
    dashboard_html = render_dashboard(data)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(dashboard_html, encoding="utf-8")
    return {
        "status": "completed",
        "output": str(output_path),
        "warnings": data["warnings"],
        "artifact_count": len(data["artifacts"]),
    }


def collect_dashboard_data(
    config: AppConfig,
    *,
    config_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    output_dir = config.output_dir
    warnings: list[str] = []
    run_manifest = _read_json_optional(output_dir / "run_manifest.json", warnings, "run manifest")
    inspect_summary = _read_json_optional(output_dir / "inspect.json", warnings, "inspect summary")
    metrics = _read_json_optional(config.data.metrics_path, warnings, "metrics")
    calibration_manifest = _read_json_optional(
        (config.calibration.output_dir or output_dir / "calibration") / "calibration_manifest.json",
        warnings,
        "calibration manifest",
    )
    comparison_manifest = _read_json_optional(
        (config.comparison.output_dir or output_dir / "comparison") / "comparison_manifest.json",
        warnings,
        "comparison manifest",
    )

    missing = _missing_expected_artifacts(config)
    if missing:
        warnings.append(
            "Missing optional artifacts: "
            + ", ".join(_display_path(path, output_dir) for path in missing)
        )
    warnings.extend(
        _dashboard_warnings(
            config,
            run_manifest,
            inspect_summary,
            metrics,
            calibration_manifest,
        )
    )

    artifacts = discover_dashboard_artifacts(config, output_path=output_path)
    charts = _chart_artifacts(config)
    return {
        "title": DASHBOARD_TITLE,
        "config": config,
        "config_path": config_path,
        "output_path": output_path,
        "run_manifest": run_manifest,
        "inspect": inspect_summary,
        "metrics": metrics,
        "calibration_manifest": calibration_manifest,
        "comparison_manifest": comparison_manifest,
        "calibration_yaml": _read_text_optional(
            (config.calibration.output_dir or output_dir / "calibration")
            / "recommended_windowing.yaml"
        ),
        "comparison_rows": _read_rows_optional(
            (config.comparison.output_dir or output_dir / "comparison") / "comparison_results.csv",
            limit=200,
        ),
        "rolling_presence": summarize_presence_file(config.data.rolling_path),
        "replay_presence": summarize_presence_file(output_dir / "replay_presence.jsonl"),
        "live_presence": summarize_presence_file(output_dir / "live_presence.jsonl"),
        "artifacts": artifacts,
        "charts": charts,
        "warnings": warnings,
    }


def discover_dashboard_artifacts(
    config: AppConfig,
    *,
    output_path: Path,
) -> list[dict[str, Any]]:
    run_dir = config.output_dir
    expected = _expected_artifacts(config)
    discovered = _discovered_artifacts(run_dir)
    paths = sorted({*expected, *discovered})
    rows: list[dict[str, Any]] = []
    for path in paths:
        if not path.exists() or path == output_path:
            continue
        rows.append(
            {
                "name": _display_path(path, run_dir),
                "path": path,
                "link": _relative_link(path, output_path.parent),
                "kind": path.suffix.lstrip(".") or "file",
                "size": path.stat().st_size,
            }
        )
    return rows


def summarize_presence_file(path: str | Path) -> dict[str, Any]:
    presence_path = Path(path)
    if not presence_path.exists():
        return {"path": str(presence_path), "exists": False, "row_count": 0, "groups": []}

    groups: dict[tuple[str, str], dict[str, Any]] = {}
    row_count = 0
    for row in iter_rows(presence_path):
        row_count += 1
        target_id = str(row.get("target_id") or "unknown")
        window_type = str(row.get("window_type") or "unknown")
        key = (target_id, window_type)
        group = groups.setdefault(
            key,
            {
                "target_id": target_id,
                "window_type": window_type,
                "row_count": 0,
                "states": Counter(),
                "timeline": [],
            },
        )
        state = str(row.get("state") or "unknown")
        group["row_count"] += 1
        group["states"][state] += 1
        if len(group["timeline"]) < 120:
            group["timeline"].append(state)

    output_groups = []
    for group in sorted(groups.values(), key=lambda item: (item["target_id"], item["window_type"])):
        output_groups.append(
            {
                "target_id": group["target_id"],
                "window_type": group["window_type"],
                "row_count": group["row_count"],
                "states": dict(group["states"]),
                "timeline": group["timeline"],
            }
        )
    return {
        "path": str(presence_path),
        "exists": True,
        "row_count": row_count,
        "groups": output_groups,
    }


def render_dashboard(data: dict[str, Any]) -> str:
    config: AppConfig = data["config"]
    title = data["title"]
    cards = _summary_cards(data)
    sections = [
        ("Overview", _overview_section(data, cards)),
        ("Capture", _capture_section(data)),
        ("Evaluation", _evaluation_section(data)),
        ("Presence", _presence_section(data)),
        ("Calibration", _calibration_section(data)),
        ("Comparison", _comparison_section(data)),
        ("Artifacts", _artifact_section(data)),
    ]
    tabs = "\n".join(
        (
            f'<button class="tab{" active" if index == 0 else ""}" '
            f'data-tab="{_slug(name)}">{_e(name)}</button>'
        )
        for index, (name, _) in enumerate(sections)
    )
    panels = "\n".join(
        (
            f'<section class="panel{" active" if index == 0 else ""}" id="{_slug(name)}">'
            f"{content}</section>"
        )
        for index, (name, content) in enumerate(sections)
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{_e(title)}</title>
  <style>{_css()}</style>
</head>
<body>
  <header class="hero">
    <div>
      <p class="eyebrow">Passive 802.11 Metadata Dashboard</p>
      <h1>{_e(title)}</h1>
      <p class="summary">{_e(config.output_dir)}</p>
    </div>
    <div class="status-pill">{_e(_run_status(data))}</div>
  </header>
  <main>
    <section class="notice">
      <strong>Passive-only safety:</strong> this dashboard reads local metadata artifacts only. It
      does not decrypt traffic, inspect payloads, inject frames, probe networks, configure adapters,
      or channel hop.
    </section>
    {_warnings_html(data["warnings"])}
    <nav class="tabs" aria-label="Dashboard sections">{tabs}</nav>
    {panels}
  </main>
  <script>{_js()}</script>
</body>
</html>
"""


def _overview_section(data: dict[str, Any], cards: list[tuple[str, Any]]) -> str:
    run_manifest = data["run_manifest"] or {}
    config: AppConfig = data["config"]
    return (
        "<h2>Run Overview</h2>"
        + _card_grid(cards)
        + _definition_table(
            [
                ("Config", data["config_path"]),
                ("Output directory", config.output_dir),
                ("Model", config.model.model_id),
                ("Label mode", config.labeling.mode),
                ("Random seed", config.random_seed),
                ("Run profile", run_manifest.get("profile", "n/a")),
                ("Started", run_manifest.get("started_at", "n/a")),
                ("Ended", run_manifest.get("ended_at", "n/a")),
            ]
        )
        + _stage_table(run_manifest)
    )


def _capture_section(data: dict[str, Any]) -> str:
    inspect_summary = data["inspect"]
    if not inspect_summary:
        return "<h2>Capture Readiness</h2><p class=\"muted\">No inspect summary found.</p>"
    coverage = inspect_summary.get("radiotap_coverage") or {}
    coverage_rows = [
        (name, f"{item.get('present', 0)} / {item.get('total', 0)}")
        for name, item in coverage.items()
        if isinstance(item, dict)
    ]
    warnings = inspect_summary.get("warnings") or []
    return (
        "<h2>Capture Readiness</h2>"
        + _card_grid(
            [
                ("Capture files", inspect_summary.get("capture_file_count")),
                ("Packets scanned", inspect_summary.get("packets_scanned")),
                ("Dot11 frames", inspect_summary.get("dot11_frame_count")),
                ("Protected frames", inspect_summary.get("protected_frame_count")),
                ("Target matches", inspect_summary.get("target_match_total")),
            ]
        )
        + "<h3>Radiotap Coverage</h3>"
        + _table(["Field", "Present / Total"], coverage_rows)
        + _inline_list("Inspect Warnings", warnings)
    )


def _evaluation_section(data: dict[str, Any]) -> str:
    metrics = data["metrics"]
    if not metrics:
        return "<h2>Evaluation</h2><p class=\"muted\">No metrics file found.</p>"
    chart_html = _chart_gallery(
        [
            ("Confusion Matrix", data["charts"].get("confusion_matrix")),
            ("ROC Curve", data["charts"].get("roc_curve")),
            ("PR Curve", data["charts"].get("pr_curve")),
        ],
        data["output_path"],
    )
    per_class = metrics.get("per_class") or {}
    per_class_rows = [
        (
            label,
            _fmt(item.get("precision")),
            _fmt(item.get("recall")),
            _fmt(item.get("f1")),
            int(item.get("support", 0)),
        )
        for label, item in per_class.items()
    ]
    return (
        "<h2>Evaluation</h2>"
        + _card_grid(
            [
                ("Examples", metrics.get("n_examples")),
                ("Accuracy", _fmt(metrics.get("accuracy"))),
                ("Precision", _fmt(metrics.get("precision"))),
                ("Recall", _fmt(metrics.get("recall"))),
                ("F1", _fmt(metrics.get("f1"))),
                ("MCC", _fmt(metrics.get("mcc"))),
                ("ROC AUC", _fmt(metrics.get("roc_auc"))),
                ("PR AUC", _fmt(metrics.get("pr_auc"))),
            ]
        )
        + chart_html
        + "<h3>Per-Class Metrics</h3>"
        + _table(["Class", "Precision", "Recall", "F1", "Support"], per_class_rows)
        + "<h3>Confusion Matrix</h3>"
        + _confusion_matrix_table(metrics)
    )


def _presence_section(data: dict[str, Any]) -> str:
    summaries = [
        ("Rolling Presence", data["rolling_presence"]),
        ("Replay Presence", data["replay_presence"]),
        ("Live Presence", data["live_presence"]),
    ]
    chunks = ["<h2>Target Presence</h2>"]
    for title, summary in summaries:
        chunks.append(f"<h3>{_e(title)}</h3>")
        if not summary["exists"]:
            chunks.append("<p class=\"muted\">No presence artifact found.</p>")
            continue
        chunks.append(f"<p class=\"muted\">Rows: {_e(summary['row_count'])}</p>")
        rows = []
        for group in summary["groups"]:
            rows.append(
                (
                    group["target_id"],
                    group["window_type"],
                    group["row_count"],
                    _state_counts(group["states"]),
                    _timeline_svg(group["timeline"]),
                )
            )
        chunks.append(
            _table(
                ["Target", "Window", "Rows", "States", "Timeline"],
                rows,
                raw_columns={4},
            )
        )
    return "".join(chunks)


def _calibration_section(data: dict[str, Any]) -> str:
    manifest = data["calibration_manifest"]
    if not manifest:
        return "<h2>Calibration</h2><p class=\"muted\">No calibration output found.</p>"
    recommendation = manifest.get("recommendation") or {}
    metrics = recommendation.get("metrics") or {}
    charts = _chart_gallery(
        [
            ("Objective Score", data["charts"].get("calibration_objective_score")),
            ("F1", data["charts"].get("calibration_f1")),
            ("Precision", data["charts"].get("calibration_precision")),
            ("Recall", data["charts"].get("calibration_recall")),
            ("MCC", data["charts"].get("calibration_mcc")),
        ],
        data["output_path"],
    )
    yaml_block = ""
    if data.get("calibration_yaml"):
        yaml_block = f"<h3>Recommended Windowing YAML</h3><pre>{_e(data['calibration_yaml'])}</pre>"
    return (
        "<h2>Calibration</h2>"
        + _card_grid(
            [
                ("Status", manifest.get("status")),
                ("Objective", recommendation.get("objective") or manifest.get("objective")),
                ("Candidate", recommendation.get("candidate_index")),
                ("Precision", _fmt(metrics.get("precision"))),
                ("Recall", _fmt(metrics.get("recall"))),
                ("F1", _fmt(metrics.get("f1"))),
                ("False positives", metrics.get("false_positive")),
                ("False negatives", metrics.get("false_negative")),
            ]
        )
        + _definition_table(
            [
                ("Present ratio threshold", recommendation.get("present_ratio_threshold")),
                ("Mean probability threshold", recommendation.get("mean_probability_threshold")),
                ("Max probability threshold", recommendation.get("max_probability_threshold")),
            ]
        )
        + _inline_list("Calibration Warnings", manifest.get("warnings") or [])
        + charts
        + yaml_block
    )


def _comparison_section(data: dict[str, Any]) -> str:
    manifest = data["comparison_manifest"]
    rows = data["comparison_rows"]
    if not manifest and not rows:
        return "<h2>Comparison</h2><p class=\"muted\">No comparison output found.</p>"
    chart_html = _chart_gallery(
        [
            ("Accuracy", data["charts"].get("comparison_accuracy")),
            ("F1", data["charts"].get("comparison_f1")),
            ("MCC", data["charts"].get("comparison_mcc")),
            ("ROC AUC", data["charts"].get("comparison_roc_auc")),
            ("PR AUC", data["charts"].get("comparison_pr_auc")),
        ],
        data["output_path"],
    )
    top_rows = sorted(rows, key=lambda row: float(row.get("f1") or 0), reverse=True)[:10]
    table_rows = [
        (
            row.get("model_id"),
            row.get("train_fraction"),
            row.get("train_rows"),
            row.get("test_rows"),
            _fmt(row.get("accuracy")),
            _fmt(row.get("f1")),
            _fmt(row.get("mcc")),
        )
        for row in top_rows
    ]
    return (
        "<h2>Model Comparison</h2>"
        + _card_grid(
            [
                ("Status", (manifest or {}).get("status", "n/a")),
                ("Result rows", len(rows)),
                ("Models", ", ".join((manifest or {}).get("models", [])) or "n/a"),
            ]
        )
        + chart_html
        + _table(
            ["Model", "Train Fraction", "Train Rows", "Test Rows", "Accuracy", "F1", "MCC"],
            table_rows,
        )
    )


def _artifact_section(data: dict[str, Any]) -> str:
    rows = [
        (
            f'<a href="{_e(item["link"])}">{_e(item["name"])}</a>',
            item["kind"],
            _format_bytes(item["size"]),
        )
        for item in data["artifacts"]
    ]
    return (
        "<h2>Artifact Index</h2>"
        "<p class=\"muted\">Links are relative to this dashboard file.</p>"
        + _table(["Artifact", "Kind", "Size"], rows, raw_columns={0})
    )


def _summary_cards(data: dict[str, Any]) -> list[tuple[str, Any]]:
    config: AppConfig = data["config"]
    metrics = data["metrics"] or {}
    inspect_summary = data["inspect"] or {}
    return [
        ("Run status", _run_status(data)),
        ("Model", config.model.model_id),
        ("Label mode", config.labeling.mode),
        ("Dot11 frames", inspect_summary.get("dot11_frame_count", "n/a")),
        ("Examples", metrics.get("n_examples", "n/a")),
        ("F1", _fmt(metrics.get("f1"))),
        ("Rolling rows", data["rolling_presence"]["row_count"]),
        ("Artifacts", len(data["artifacts"])),
    ]


def _dashboard_warnings(
    config: AppConfig,
    run_manifest: dict[str, Any] | None,
    inspect_summary: dict[str, Any] | None,
    metrics: dict[str, Any] | None,
    calibration_manifest: dict[str, Any] | None,
) -> list[str]:
    warnings: list[str] = []
    if config.features.leakage_debug:
        warnings.append("Leakage/debug mode is enabled; MAC-derived features may be present.")
    if _looks_synthetic(config):
        warnings.append(
            "This appears to be synthetic data; do not treat results as real-world performance."
        )
    if run_manifest:
        failed = [
            stage for stage in run_manifest.get("stages", []) if stage.get("status") == "failed"
        ]
        if failed:
            warnings.append("One or more run stages failed.")
    if inspect_summary:
        warnings.extend(str(warning) for warning in inspect_summary.get("warnings") or [])
    if metrics and _severe_skew(metrics):
        warnings.append("Class distribution is severely skewed; accuracy may be misleading.")
    if calibration_manifest:
        warnings.extend(str(warning) for warning in calibration_manifest.get("warnings") or [])
        if calibration_manifest.get("status") == "failed":
            warnings.append("Calibration failed; inspect calibration_manifest.json.")
    return list(dict.fromkeys(warnings))


def _expected_artifacts(config: AppConfig) -> set[Path]:
    output_dir = config.output_dir
    calibration_dir = config.calibration.output_dir or output_dir / "calibration"
    comparison_dir = config.comparison.output_dir or output_dir / "comparison"
    return {
        output_dir / "run_manifest.json",
        output_dir / "run_summary.md",
        output_dir / "inspect.json",
        output_dir / "inspect.md",
        output_dir / "run_config.json",
        output_dir / "report.md",
        output_dir / "confusion_matrix.csv",
        output_dir / "confusion_matrix.png",
        output_dir / "roc_curve.png",
        output_dir / "pr_curve.png",
        output_dir / "replay_predictions.jsonl",
        output_dir / "replay_presence.jsonl",
        output_dir / "live_predictions.jsonl",
        output_dir / "live_presence.jsonl",
        config.data.normalized_path,
        config.data.features_path,
        config.data.labeled_path,
        config.data.train_path,
        config.data.test_path,
        config.data.predictions_path,
        config.data.rolling_path,
        config.data.metrics_path,
        config.model.checkpoint_path,
        calibration_dir / "calibration_manifest.json",
        calibration_dir / "calibration_results.csv",
        calibration_dir / "calibration_results.json",
        calibration_dir / "calibration_report.md",
        calibration_dir / "recommended_windowing.yaml",
        comparison_dir / "comparison_manifest.json",
        comparison_dir / "comparison_results.csv",
        comparison_dir / "comparison_results.json",
        comparison_dir / "comparison_report.md",
    }


def _missing_expected_artifacts(config: AppConfig) -> list[Path]:
    optional = _expected_artifacts(config)
    return sorted(path for path in optional if not path.exists())


def _discovered_artifacts(output_dir: Path) -> set[Path]:
    if not output_dir.exists():
        return set()
    return {
        path
        for path in output_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in ARTIFACT_SUFFIXES
    }


def _chart_artifacts(config: AppConfig) -> dict[str, Path]:
    output_dir = config.output_dir
    calibration_dir = config.calibration.output_dir or output_dir / "calibration"
    comparison_dir = config.comparison.output_dir or output_dir / "comparison"
    names = {
        "confusion_matrix": output_dir / "confusion_matrix.png",
        "roc_curve": output_dir / "roc_curve.png",
        "pr_curve": output_dir / "pr_curve.png",
        "calibration_objective_score": calibration_dir / "calibration_objective_score.png",
        "calibration_precision": calibration_dir / "calibration_precision.png",
        "calibration_recall": calibration_dir / "calibration_recall.png",
        "calibration_f1": calibration_dir / "calibration_f1.png",
        "calibration_mcc": calibration_dir / "calibration_mcc.png",
        "comparison_accuracy": comparison_dir / "comparison_accuracy.png",
        "comparison_f1": comparison_dir / "comparison_f1.png",
        "comparison_mcc": comparison_dir / "comparison_mcc.png",
        "comparison_roc_auc": comparison_dir / "comparison_roc_auc.png",
        "comparison_pr_auc": comparison_dir / "comparison_pr_auc.png",
    }
    return {name: path for name, path in names.items() if path.exists()}


def _read_json_optional(
    path: Path,
    warnings: list[str],
    name: str,
) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return read_json(path)
    except Exception as exc:
        warnings.append(f"Could not read {name} at {_display_path(path, path.parent)}: {exc}")
        return None


def _read_text_optional(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return None


def _read_rows_optional(path: Path, *, limit: int) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        for row in iter_rows(path):
            rows.append(row)
            if len(rows) >= limit:
                break
    except Exception:
        return []
    return rows


def _run_status(data: dict[str, Any]) -> str:
    run_manifest = data["run_manifest"] or {}
    return str(run_manifest.get("status") or "unknown")


def _stage_table(run_manifest: dict[str, Any]) -> str:
    stages = run_manifest.get("stages") or []
    if not stages:
        return "<h3>Stages</h3><p class=\"muted\">No run stage data found.</p>"
    rows = []
    for stage in stages:
        counts = stage.get("row_counts") or {}
        rows.append(
            (
                stage.get("name"),
                stage.get("status"),
                ", ".join(f"{key}={value}" for key, value in counts.items()) or "-",
            )
        )
    return "<h3>Stages</h3>" + _table(["Stage", "Status", "Counts"], rows)


def _confusion_matrix_table(metrics: dict[str, Any]) -> str:
    confusion = metrics.get("confusion_matrix") or {}
    labels = confusion.get("labels") or []
    matrix = confusion.get("matrix") or []
    if not labels or not matrix:
        return "<p class=\"muted\">No confusion matrix data found.</p>"
    rows = []
    for label, values in zip(labels, matrix, strict=False):
        rows.append((label, *values))
    return _table(["Actual \\ Predicted", *labels], rows)


def _card_grid(cards: list[tuple[str, Any]]) -> str:
    return "<div class=\"cards\">" + "".join(
        f"<article class=\"card\"><span>{_e(label)}</span><strong>{_e(value)}</strong></article>"
        for label, value in cards
    ) + "</div>"


def _definition_table(items: list[tuple[str, Any]]) -> str:
    return _table(["Field", "Value"], items)


def _table(
    headers: list[str],
    rows: list[tuple[Any, ...]],
    *,
    raw_columns: set[int] | None = None,
) -> str:
    raw = raw_columns or set()
    if not rows:
        return "<p class=\"muted\">No rows available.</p>"
    header_html = "".join(f"<th>{_e(header)}</th>" for header in headers)
    row_html = []
    for row in rows:
        cells = []
        for index, value in enumerate(row):
            text = str(value)
            cells.append(f"<td>{text if index in raw else _e(text)}</td>")
        row_html.append("<tr>" + "".join(cells) + "</tr>")
    return (
        '<div class="table-wrap"><table><thead><tr>'
        f"{header_html}</tr></thead><tbody>{''.join(row_html)}</tbody></table></div>"
    )


def _inline_list(title: str, items: list[Any]) -> str:
    if not items:
        return ""
    return (
        f"<h3>{_e(title)}</h3><ul class=\"warning-list\">"
        + "".join(f"<li>{_e(item)}</li>" for item in items)
        + "</ul>"
    )


def _warnings_html(warnings: list[str]) -> str:
    if not warnings:
        return ""
    return (
        "<section class=\"warnings\"><h2>Warnings</h2><ul>"
        + "".join(f"<li>{_e(warning)}</li>" for warning in warnings)
        + "</ul></section>"
    )


def _chart_gallery(charts: list[tuple[str, Path | None]], output_path: Path) -> str:
    items = []
    for title, path in charts:
        if path is None or not path.exists():
            continue
        items.append(
            "<figure>"
            f"<img alt=\"{_e(title)}\" src=\"{_image_data_uri(path)}\">"
            f"<figcaption>{_e(title)} · "
            f"<a href=\"{_e(_relative_link(path, output_path.parent))}\">artifact</a>"
            "</figcaption>"
            "</figure>"
        )
    if not items:
        return ""
    return "<div class=\"chart-grid\">" + "".join(items) + "</div>"


def _timeline_svg(states: list[str]) -> str:
    if not states:
        return "<span class=\"muted\">n/a</span>"
    width = max(120, len(states) * 6)
    height = 20
    rect_width = width / len(states)
    rects = []
    for index, state in enumerate(states):
        color = STATE_COLORS.get(state, "#394150")
        x = index * rect_width
        width = rect_width + 0.4
        rects.append(
            f'<rect x="{x:.2f}" y="0" width="{width:.2f}" '
            f'height="{height}" fill="{color}"><title>{_e(state)}</title></rect>'
        )
    return (
        f'<svg class="timeline" role="img" viewBox="0 0 {width} {height}">'
        f'{"".join(rects)}</svg>'
    )


def _state_counts(states: dict[str, int]) -> str:
    return ", ".join(f"{state}={count}" for state, count in sorted(states.items())) or "-"


def _image_data_uri(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _relative_link(path: Path, start: Path) -> str:
    return os.path.relpath(path, start=start).replace(os.sep, "/")


def _display_path(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _format_bytes(size: int) -> str:
    if size < 1024:
        return f"{size} B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"
    return f"{size / (1024 * 1024):.1f} MB"


def _severe_skew(metrics: dict[str, Any]) -> bool:
    supports = [item.get("support", 0) for item in (metrics.get("per_class") or {}).values()]
    if len(supports) < 2:
        return False
    smallest = min(supports)
    largest = max(supports)
    return bool(smallest and largest / smallest >= 10)


def _looks_synthetic(config: AppConfig) -> bool:
    text = " ".join([str(config.output_dir), *(str(path) for path in config.input.paths)])
    return "synthetic" in text.lower()


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _slug(value: str) -> str:
    return value.lower().replace(" ", "-")


def _e(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _css() -> str:
    return """
:root {
  color-scheme: light;
  --ink:#1f2933;
  --muted:#65758b;
  --line:#d8dee8;
  --panel:#ffffff;
  --band:#f4f7fb;
  --accent:#1660a8;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  background: var(--band);
  color: var(--ink);
  line-height: 1.45;
}
.hero {
  display:flex;
  justify-content:space-between;
  gap:24px;
  align-items:flex-end;
  padding:32px;
  background:#172033;
  color:#fff;
}
.eyebrow {
  margin:0 0 6px;
  color:#b8c7d9;
  font-size:13px;
  text-transform:uppercase;
  letter-spacing:.08em;
}
h1 { margin:0; font-size:34px; }
h2 { margin:0 0 18px; font-size:24px; }
h3 { margin:24px 0 10px; font-size:17px; }
.summary { margin:8px 0 0; color:#c9d5e4; }
.status-pill {
  padding:8px 12px;
  border:1px solid #7f93ad;
  border-radius:6px;
  min-width:110px;
  text-align:center;
}
main { max-width:1180px; margin:0 auto; padding:24px; }
.notice, .warnings, .panel {
  background:var(--panel);
  border:1px solid var(--line);
  border-radius:8px;
  padding:18px;
  margin:0 0 18px;
}
.warnings { border-color:#f0c56a; background:#fff8e8; }
.warnings h2 { font-size:18px; margin-bottom:8px; }
.tabs { display:flex; flex-wrap:wrap; gap:8px; margin:22px 0; }
.tab {
  border:1px solid var(--line);
  background:#fff;
  color:var(--ink);
  border-radius:6px;
  padding:9px 12px;
  cursor:pointer;
}
.tab.active { background:var(--accent); color:#fff; border-color:var(--accent); }
.panel { display:none; }
.panel.active { display:block; }
.cards {
  display:grid;
  grid-template-columns:repeat(auto-fit, minmax(150px, 1fr));
  gap:12px;
  margin-bottom:20px;
}
.card { border:1px solid var(--line); border-radius:8px; background:#fff; padding:14px; }
.card span { display:block; color:var(--muted); font-size:13px; margin-bottom:6px; }
.card strong { font-size:20px; overflow-wrap:anywhere; }
.table-wrap {
  overflow-x:auto;
  margin:10px 0 18px;
  border:1px solid var(--line);
  border-radius:8px;
}
table { width:100%; border-collapse:collapse; background:#fff; }
th, td {
  padding:10px 12px;
  border-bottom:1px solid var(--line);
  text-align:left;
  vertical-align:top;
}
th { background:#eef3f8; font-size:13px; color:#394150; }
tr:last-child td { border-bottom:0; }
.chart-grid {
  display:grid;
  grid-template-columns:repeat(auto-fit, minmax(280px, 1fr));
  gap:14px;
  margin:14px 0 20px;
}
figure { margin:0; border:1px solid var(--line); border-radius:8px; padding:10px; background:#fff; }
figure img { width:100%; height:auto; display:block; }
figcaption { color:var(--muted); font-size:13px; margin-top:8px; }
.timeline {
  width:160px;
  max-width:100%;
  height:20px;
  border-radius:4px;
  overflow:hidden;
  background:#eef3f8;
}
.muted { color:var(--muted); }
.warning-list { margin-top:8px; }
pre {
  white-space:pre-wrap;
  background:#172033;
  color:#f2f6fb;
  padding:14px;
  border-radius:8px;
  overflow:auto;
}
a { color:var(--accent); }
"""


def _js() -> str:
    return """
document.querySelectorAll('[data-tab]').forEach((button) => {
  button.addEventListener('click', () => {
    document.querySelectorAll('[data-tab]').forEach((item) => item.classList.remove('active'));
    document.querySelectorAll('.panel').forEach((panel) => panel.classList.remove('active'));
    button.classList.add('active');
    const panel = document.getElementById(button.dataset.tab);
    if (panel) panel.classList.add('active');
  });
});
"""
