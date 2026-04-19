"""Application model IDs mapped to native Python streaming estimators."""

from __future__ import annotations

from typing import Any

from peekaboo.models.base import OnlineModel

MODEL_MAPPINGS = {
    "leveraging_bag": "river.ensemble.LeveragingBaggingClassifier(HoeffdingTreeClassifier)",
    "oza_boost": "river.ensemble.AdaBoostClassifier(HoeffdingTreeClassifier)",
    "oza_boost_adwin": (
        "river.ensemble.ADWINBoostingClassifier(HoeffdingTreeClassifier), "
        "falling back to ADWINBaggingClassifier when unavailable"
    ),
    "adaptive_hoeffding_tree": "river.tree.HoeffdingAdaptiveTreeClassifier",
}


def create_model(
    model_id: str,
    *,
    feature_names: list[str],
    seed: int = 42,
    params: dict[str, Any] | None = None,
) -> OnlineModel:
    try:
        from river import ensemble, tree  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency availability
        raise RuntimeError("river is required for online model training") from exc

    params = dict(params or {})
    base_params = dict(params.pop("base_model", {}))
    n_models = int(params.pop("n_models", 10))

    if model_id == "adaptive_hoeffding_tree":
        estimator = tree.HoeffdingAdaptiveTreeClassifier(seed=seed, **params)
    elif model_id == "leveraging_bag":
        base = tree.HoeffdingTreeClassifier(**base_params)
        estimator = ensemble.LeveragingBaggingClassifier(
            model=base,
            n_models=n_models,
            seed=seed,
            **params,
        )
    elif model_id == "oza_boost":
        base = tree.HoeffdingTreeClassifier(**base_params)
        estimator = ensemble.AdaBoostClassifier(model=base, n_models=n_models, seed=seed, **params)
    elif model_id == "oza_boost_adwin":
        base = tree.HoeffdingTreeClassifier(**base_params)
        if hasattr(ensemble, "ADWINBoostingClassifier"):
            estimator = ensemble.ADWINBoostingClassifier(
                model=base,
                n_models=n_models,
                seed=seed,
                **params,
            )
        elif hasattr(ensemble, "ADWINBaggingClassifier"):
            estimator = ensemble.ADWINBaggingClassifier(
                model=base,
                n_models=n_models,
                seed=seed,
                **params,
            )
        else:
            raise RuntimeError("The installed river version provides no ADWIN ensemble classifier")
    else:
        raise ValueError(f"Unsupported model_id: {model_id}")

    return OnlineModel(
        model_id=model_id,
        estimator=estimator,
        feature_names=feature_names,
        metadata={"mapping": MODEL_MAPPINGS[model_id], "seed": seed},
    )
