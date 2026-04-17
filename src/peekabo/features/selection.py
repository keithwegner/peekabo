"""Information-gain-like feature ranking."""

from __future__ import annotations

import math
import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from peekabo.data.readers import iter_rows
from peekabo.data.writers import write_json
from peekabo.features.extract import DEFAULT_MODEL_FEATURES


def rank_feature_file(
    input_path: str | Path,
    *,
    label_column: str = "label",
    features: list[str] | None = None,
    sample_size: int | None = None,
    seed: int = 42,
    output_json: str | Path | None = None,
    output_markdown: str | Path | None = None,
) -> list[dict[str, Any]]:
    rows = list(_sample_rows(iter_rows(input_path), sample_size=sample_size, seed=seed))
    feature_names = features or [name for name in DEFAULT_MODEL_FEATURES if rows and name in rows[0]]
    ranked = [
        {"feature": feature, "score": mutual_information(rows, feature, label_column)}
        for feature in feature_names
    ]
    ranked.sort(key=lambda item: item["score"], reverse=True)
    if output_json is not None:
        write_json(output_json, {"features": ranked})
    if output_markdown is not None:
        write_feature_ranking_markdown(output_markdown, ranked)
    return ranked


def mutual_information(rows: list[dict[str, Any]], feature: str, label_column: str) -> float:
    pair_counts: Counter[tuple[str, str]] = Counter()
    feature_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()

    for row in rows:
        label = row.get(label_column)
        if label is None:
            continue
        value = _bucket_value(row.get(feature))
        label_text = str(label)
        pair_counts[(value, label_text)] += 1
        feature_counts[value] += 1
        label_counts[label_text] += 1

    total = sum(pair_counts.values())
    if total == 0:
        return 0.0

    score = 0.0
    for (value, label), pair_count in pair_counts.items():
        p_xy = pair_count / total
        p_x = feature_counts[value] / total
        p_y = label_counts[label] / total
        score += p_xy * math.log(p_xy / (p_x * p_y), 2)
    return score


def write_feature_ranking_markdown(path: str | Path, ranked: list[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Feature Ranking", "", "| Rank | Feature | Score |", "| --- | --- | --- |"]
    for index, item in enumerate(ranked, start=1):
        lines.append(f"| {index} | `{item['feature']}` | {item['score']:.6f} |")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _sample_rows(
    rows: Iterable[dict[str, Any]],
    *,
    sample_size: int | None,
    seed: int,
) -> list[dict[str, Any]]:
    if sample_size is None:
        return list(rows)
    rng = random.Random(seed)
    reservoir: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        if len(reservoir) < sample_size:
            reservoir.append(row)
            continue
        replace_index = rng.randrange(index)
        if replace_index < sample_size:
            reservoir[replace_index] = row
    return reservoir


def _bucket_value(value: Any) -> str:
    if value is None:
        return "missing"
    if isinstance(value, float):
        if math.isnan(value):
            return "missing"
        return f"{value:.3g}"
    return str(value)
