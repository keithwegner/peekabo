"""Classification metrics with sklearn-backed AUCs when available."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any


def classification_metrics(
    y_true: list[str],
    y_pred: list[str],
    *,
    y_score: list[float] | None = None,
    positive_label: str = "target",
) -> dict[str, Any]:
    labels = sorted(set(y_true) | set(y_pred))
    matrix = confusion_matrix(y_true, y_pred, labels)
    total = len(y_true)
    correct = sum(1 for truth, pred in zip(y_true, y_pred) if truth == pred)
    accuracy = correct / total if total else 0.0

    per_class = per_class_metrics(matrix, labels)
    precision, recall, f1 = _primary_precision_recall_f1(per_class, positive_label)
    mcc = matthews_corrcoef(matrix, labels)

    metrics: dict[str, Any] = {
        "n_examples": total,
        "labels": labels,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mcc": mcc,
        "per_class": per_class,
        "confusion_matrix": {
            "labels": labels,
            "matrix": matrix,
        },
    }

    aucs = binary_auc_metrics(y_true, y_score, positive_label=positive_label)
    metrics.update(aucs)
    return metrics


def confusion_matrix(y_true: list[str], y_pred: list[str], labels: list[str]) -> list[list[int]]:
    index = {label: idx for idx, label in enumerate(labels)}
    matrix = [[0 for _ in labels] for _ in labels]
    for truth, pred in zip(y_true, y_pred):
        matrix[index[truth]][index[pred]] += 1
    return matrix


def per_class_metrics(matrix: list[list[int]], labels: list[str]) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    supports = [sum(row) for row in matrix]
    columns = [sum(matrix[row][col] for row in range(len(labels))) for col in range(len(labels))]
    for idx, label in enumerate(labels):
        tp = matrix[idx][idx]
        fp = columns[idx] - tp
        fn = supports[idx] - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        result[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": float(supports[idx]),
        }
    return result


def matthews_corrcoef(matrix: list[list[int]], labels: list[str]) -> float | None:
    if not labels:
        return None
    n = sum(sum(row) for row in matrix)
    if n == 0:
        return None
    trace = sum(matrix[i][i] for i in range(len(labels)))
    row_sums = [sum(row) for row in matrix]
    col_sums = [sum(matrix[row][col] for row in range(len(labels))) for col in range(len(labels))]
    cov_ytyp = trace * n - sum(row_sums[i] * col_sums[i] for i in range(len(labels)))
    cov_ypyp = n * n - sum(value * value for value in col_sums)
    cov_ytyt = n * n - sum(value * value for value in row_sums)
    denominator = math.sqrt(cov_ytyt * cov_ypyp)
    if denominator == 0:
        return None
    return cov_ytyp / denominator


def binary_auc_metrics(
    y_true: list[str],
    y_score: list[float] | None,
    *,
    positive_label: str,
) -> dict[str, float | None]:
    if not y_score or len(set(y_true)) < 2:
        return {"roc_auc": None, "pr_auc": None}
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score  # type: ignore
    except Exception:
        return {"roc_auc": None, "pr_auc": None}
    truth = [1 if label == positive_label else 0 for label in y_true]
    try:
        return {
            "roc_auc": float(roc_auc_score(truth, y_score)),
            "pr_auc": float(average_precision_score(truth, y_score)),
        }
    except Exception:
        return {"roc_auc": None, "pr_auc": None}


def class_distribution(rows: list[dict[str, Any]], label_column: str = "label") -> dict[str, int]:
    return dict(Counter(str(row[label_column]) for row in rows if row.get(label_column) is not None))


def _primary_precision_recall_f1(
    per_class: dict[str, dict[str, float]],
    positive_label: str,
) -> tuple[float, float, float]:
    if positive_label in per_class:
        item = per_class[positive_label]
        return item["precision"], item["recall"], item["f1"]

    total_support = sum(item["support"] for item in per_class.values())
    if total_support == 0:
        return 0.0, 0.0, 0.0
    precision = sum(item["precision"] * item["support"] for item in per_class.values()) / total_support
    recall = sum(item["recall"] * item["support"] for item in per_class.values()) / total_support
    f1 = sum(item["f1"] * item["support"] for item in per_class.values()) / total_support
    return precision, recall, f1

