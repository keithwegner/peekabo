"""Plot helpers for generated reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_confusion_matrix_csv(path: str | Path, metrics: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    labels = metrics["confusion_matrix"]["labels"]
    matrix = metrics["confusion_matrix"]["matrix"]
    lines = ["," + ",".join(labels)]
    for label, row in zip(labels, matrix):
        lines.append(",".join([label, *[str(value) for value in row]]))
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_confusion_matrix(path: str | Path, metrics: dict[str, Any]) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency availability
        raise RuntimeError("matplotlib is required for PNG plots") from exc

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    labels = metrics["confusion_matrix"]["labels"]
    matrix = metrics["confusion_matrix"]["matrix"]
    fig, ax = plt.subplots(figsize=(max(4, len(labels)), max(4, len(labels))))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels=labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for row_index, row in enumerate(matrix):
        for col_index, value in enumerate(row):
            ax.text(col_index, row_index, str(value), ha="center", va="center")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def plot_binary_curves(
    roc_path: str | Path,
    pr_path: str | Path,
    predictions: list[dict[str, Any]],
    *,
    positive_label: str = "target",
) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
        from sklearn.metrics import precision_recall_curve, roc_curve  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency availability
        raise RuntimeError("matplotlib and scikit-learn are required for ROC/PR plots") from exc

    y_true: list[int] = []
    y_score: list[float] = []
    for row in predictions:
        truth = row.get("ground_truth")
        if truth is None:
            continue
        probabilities = _probabilities(row)
        y_true.append(1 if str(truth) == positive_label else 0)
        y_score.append(float(probabilities.get(positive_label, 0.0)))
    if len(set(y_true)) < 2:
        return

    fpr, tpr, _ = roc_curve(y_true, y_score)
    precision, recall, _ = precision_recall_curve(y_true, y_score)

    _line_plot(roc_path, fpr, tpr, xlabel="False positive rate", ylabel="True positive rate", title="ROC Curve")
    _line_plot(pr_path, recall, precision, xlabel="Recall", ylabel="Precision", title="PR Curve")


def _line_plot(path: str | Path, x: Any, y: Any, *, xlabel: str, ylabel: str, title: str) -> None:
    import matplotlib.pyplot as plt  # type: ignore

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(x, y)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def _probabilities(row: dict[str, Any]) -> dict[str, float]:
    raw = row.get("probabilities_json")
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except Exception:
        return {}
    return {str(key): float(value) for key, value in data.items()}
