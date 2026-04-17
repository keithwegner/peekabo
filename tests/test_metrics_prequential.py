from wifi_id.config import FeatureConfig
from wifi_id.evaluation.metrics import classification_metrics
from wifi_id.evaluation.prequential import evaluate_prequential_rows
from wifi_id.models.base import OnlineModel


class RecordingEstimator:
    def __init__(self):
        self.events = []
        self.learned = 0

    def predict_proba_one(self, x):
        self.events.append(("proba", self.learned))
        return {"target": 0.9, "other": 0.1} if self.learned else {}

    def predict_one(self, x):
        self.events.append(("predict", self.learned))
        return "target" if self.learned else None

    def learn_one(self, x, y):
        self.events.append(("learn", self.learned))
        self.learned += 1


def test_prequential_predicts_before_learning_current_row():
    estimator = RecordingEstimator()
    model = OnlineModel("fake", estimator, [], {})
    metrics, predictions = evaluate_prequential_rows(
        [
            {"label": "target", "hour": 1, "data_rate": 1, "ssi": 1, "frame_type": 2, "frame_subtype": 0, "data_size": 10},
            {"label": "target", "hour": 1, "data_rate": 1, "ssi": 1, "frame_type": 2, "frame_subtype": 0, "data_size": 10},
        ],
        model,
        FeatureConfig(),
    )
    assert estimator.events[:3] == [("proba", 0), ("predict", 0), ("learn", 0)]
    assert metrics["n_examples"] == 2
    assert len(predictions) == 2


def test_metrics_include_imbalance_metrics():
    metrics = classification_metrics(
        ["target", "target", "other", "other"],
        ["target", "other", "other", "other"],
        y_score=[0.9, 0.4, 0.2, 0.1],
    )
    assert metrics["accuracy"] == 0.75
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 0.5
    assert metrics["f1"] > 0
    assert metrics["mcc"] is not None

