"""Human-readable experiment reports."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from peekabo.config import AppConfig
from peekabo.models.registry import MODEL_MAPPINGS


def write_markdown_report(
    path: str | Path,
    *,
    config: AppConfig,
    metrics: dict[str, Any],
    title: str = "peekabo Experiment Report",
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# {title}",
        "",
        "## Summary",
        "",
        f"- Model: `{config.model.model_id}`",
        f"- Label mode: `{config.labeling.mode}`",
        f"- Examples: `{metrics.get('n_examples', 0)}`",
        f"- Accuracy: `{_fmt(metrics.get('accuracy'))}`",
        f"- Precision: `{_fmt(metrics.get('precision'))}`",
        f"- Recall: `{_fmt(metrics.get('recall'))}`",
        f"- F1: `{_fmt(metrics.get('f1'))}`",
        f"- MCC: `{_fmt(metrics.get('mcc'))}`",
        f"- ROC AUC: `{_fmt(metrics.get('roc_auc'))}`",
        f"- PR AUC: `{_fmt(metrics.get('pr_auc'))}`",
        "",
        "## Model Mapping",
        "",
        f"`{config.model.model_id}` maps to `{MODEL_MAPPINGS.get(config.model.model_id, 'unknown')}`.",
    ]
    if config.features.leakage_debug:
        lines.extend(
            [
                "",
                "## Leakage Debug Warning",
                "",
                "This run included MAC-derived model features and should not be treated as a faithful reproduction run.",
            ]
        )
    if _severe_skew(metrics):
        lines.extend(
            [
                "",
                "## Imbalance Warning",
                "",
                "The class distribution is severely skewed; accuracy is less informative than precision, recall, F1, MCC, ROC AUC, and PR AUC.",
            ]
        )
    lines.extend(["", "## Per-Class Metrics", "", "| Class | Precision | Recall | F1 | Support |", "| --- | --- | --- | --- | --- |"])
    for label, item in (metrics.get("per_class") or {}).items():
        lines.append(
            f"| `{label}` | {_fmt(item.get('precision'))} | {_fmt(item.get('recall'))} | {_fmt(item.get('f1'))} | {int(item.get('support', 0))} |"
        )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _severe_skew(metrics: dict[str, Any]) -> bool:
    supports = [item.get("support", 0) for item in (metrics.get("per_class") or {}).values()]
    if len(supports) < 2:
        return False
    smallest = min(supports)
    largest = max(supports)
    return bool(smallest and largest / smallest >= 10)

