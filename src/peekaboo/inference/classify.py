"""Offline row classification."""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

from peekaboo.config import FeatureConfig
from peekaboo.features.extract import row_to_model_features
from peekaboo.models.base import OnlineModel
from peekaboo.parsing.records import PredictionRow


def classify_rows(
    rows: Iterable[dict[str, Any]],
    model: OnlineModel,
    feature_config: FeatureConfig,
    *,
    label_mode: str = "unknown",
    label_column: str = "label",
    positive_label: str = "target",
) -> list[dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    for row in rows:
        features = row_to_model_features(row, feature_config)
        probabilities = model.predict_proba_one(features)
        prediction = model.predict_one(features)
        if prediction is None and probabilities:
            prediction = max(probabilities, key=probabilities.get)
        prediction = prediction or "__none__"
        confidence = probabilities.get(prediction)
        if confidence is None and probabilities:
            confidence = max(probabilities.values())
        predictions.append(
            PredictionRow(
                timestamp=row.get("timestamp"),
                packet_index=int(row.get("packet_index") or len(predictions)),
                label_mode=str(row.get("label_mode") or label_mode),
                predicted_class=str(prediction),
                confidence=confidence,
                ground_truth=None if row.get(label_column) is None else str(row.get(label_column)),
                source_mac=row.get("source_mac"),
                destination_mac=row.get("destination_mac"),
                features_json=json.dumps(features, sort_keys=True),
                probabilities_json=json.dumps(probabilities, sort_keys=True),
            ).to_dict()
        )
    return predictions
