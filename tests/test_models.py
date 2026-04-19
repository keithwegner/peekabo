from pathlib import Path

import pytest

from peekaboo.models.base import load_checkpoint
from peekaboo.models.registry import MODEL_MAPPINGS, create_model

pytest.importorskip("river")


def test_all_model_ids_instantiate_and_checkpoint(tmp_path: Path):
    for model_id in MODEL_MAPPINGS:
        model = create_model(model_id, feature_names=["x"], seed=1, params={"n_models": 2})
        model.learn_one({"x": 1}, "target")
        assert model.predict_one({"x": 1}) is not None
        checkpoint = tmp_path / f"{model_id}.pkl"
        model.save(checkpoint)
        loaded = load_checkpoint(checkpoint)
        assert loaded.model_id == model_id
