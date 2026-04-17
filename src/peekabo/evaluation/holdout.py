"""Classic train/test holdout evaluation."""

from __future__ import annotations

import json
from typing import Any, Iterable

from peekabo.config import FeatureConfig
from peekabo.evaluation.metrics import classification_metrics
from peekabo.features.extract import row_to_model_features
from peekabo.models.base import OnlineModel
from peekabo.parsing.records import PredictionRow


def train_online_rows(
    rows: Iterable[dict[str, Any]],
    model: OnlineModel,
    feature_config: FeatureConfig,
    *,
    label_column: str = "label",
    weight_column: str | None = None,
) -> int:
    count = 0
    for row in rows:
        label = row.get(label_column)
        if label is None:
            continue
        features = row_to_model_features(row, feature_config)
        weight = None if weight_column is None else row.get(weight_column)
        model.learn_one(features, str(label), None if weight is None else float(weight))
        count += 1
    return count


def evaluate_holdout_rows(
    train_rows: Iterable[dict[str, Any]],
    test_rows: Iterable[dict[str, Any]],
    model: OnlineModel,
    feature_config: FeatureConfig,
    *,
    label_column: str = "label",
    positive_label: str = "target",
    weight_column: str | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    train_count = train_online_rows(
        train_rows,
        model,
        feature_config,
        label_column=label_column,
        weight_column=weight_column,
    )
    y_true: list[str] = []
    y_pred: list[str] = []
    y_score: list[float] = []
    predictions: list[dict[str, Any]] = []

    for row in test_rows:
        label = row.get(label_column)
        if label is None:
            continue
        features = row_to_model_features(row, feature_config)
        probabilities = model.predict_proba_one(features)
        prediction = model.predict_one(features)
        if prediction is None and probabilities:
            prediction = max(probabilities, key=probabilities.get)
        prediction = prediction or "__none__"
        confidence = probabilities.get(prediction)
        if confidence is None and probabilities:
            confidence = max(probabilities.values())
        y_true.append(str(label))
        y_pred.append(str(prediction))
        y_score.append(float(probabilities.get(positive_label, 0.0)))
        predictions.append(
            PredictionRow(
                timestamp=row.get("timestamp"),
                packet_index=int(row.get("packet_index") or len(predictions)),
                label_mode=str(row.get("label_mode") or "unknown"),
                predicted_class=str(prediction),
                confidence=confidence,
                ground_truth=str(label),
                source_mac=row.get("source_mac"),
                destination_mac=row.get("destination_mac"),
                features_json=json.dumps(features, sort_keys=True),
                probabilities_json=json.dumps(probabilities, sort_keys=True),
            ).to_dict()
        )

    metrics = classification_metrics(y_true, y_pred, y_score=y_score, positive_label=positive_label)
    metrics["train_examples"] = train_count
    return metrics, predictions
