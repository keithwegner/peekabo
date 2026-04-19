"""Common online classifier wrapper and checkpoint format."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CHECKPOINT_VERSION = 1


@dataclass
class OnlineModel:
    model_id: str
    estimator: Any
    feature_names: list[str]
    metadata: dict[str, Any]

    def predict_one(self, features: dict[str, Any]) -> str | None:
        prediction = self.estimator.predict_one(features)
        return None if prediction is None else str(prediction)

    def predict_proba_one(self, features: dict[str, Any]) -> dict[str, float]:
        if not hasattr(self.estimator, "predict_proba_one"):
            return {}
        probabilities = self.estimator.predict_proba_one(features) or {}
        return {str(label): float(value) for label, value in probabilities.items()}

    def learn_one(self, features: dict[str, Any], label: str, weight: float | None = None) -> None:
        if weight is None:
            self.estimator.learn_one(features, label)
            return
        try:
            self.estimator.learn_one(features, label, w=weight)
        except TypeError:
            self.estimator.learn_one(features, label)

    def save(self, path: str | Path) -> None:
        save_checkpoint(path, self)


def save_checkpoint(path: str | Path, model: OnlineModel) -> None:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": CHECKPOINT_VERSION,
        "model_id": model.model_id,
        "feature_names": model.feature_names,
        "metadata": model.metadata,
        "estimator": model.estimator,
    }
    with checkpoint_path.open("wb") as handle:
        pickle.dump(payload, handle)


def load_checkpoint(path: str | Path) -> OnlineModel:
    with Path(path).open("rb") as handle:
        payload = pickle.load(handle)
    version = payload.get("version")
    if version != CHECKPOINT_VERSION:
        raise ValueError(f"Unsupported checkpoint version: {version}")
    return OnlineModel(
        model_id=payload["model_id"],
        estimator=payload["estimator"],
        feature_names=list(payload["feature_names"]),
        metadata=dict(payload.get("metadata") or {}),
    )
