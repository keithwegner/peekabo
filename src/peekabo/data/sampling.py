"""Deterministic sampling and balancing utilities."""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from peekabo.data.readers import iter_rows, read_all_rows
from peekabo.data.writers import write_rows


def random_sample_rows(
    rows: Iterable[dict[str, Any]],
    *,
    percentage: float,
    seed: int,
) -> Iterator[dict[str, Any]]:
    if percentage >= 100:
        yield from rows
        return
    if percentage <= 0:
        return
    threshold = percentage / 100.0
    rng = random.Random(seed)
    for row in rows:
        if rng.random() <= threshold:
            yield row


def random_sample_file(
    input_path: str | Path,
    output_path: str | Path,
    *,
    percentage: float,
    seed: int,
) -> int:
    return write_rows(
        output_path, random_sample_rows(iter_rows(input_path), percentage=percentage, seed=seed)
    )


def balance_rows(
    rows: Iterable[dict[str, Any]],
    *,
    strategy: str,
    label_column: str = "label",
    seed: int = 42,
    class_weight_column: str = "sample_weight",
) -> Iterator[dict[str, Any]]:
    strategy = strategy.lower()
    if strategy in {"none", "off", ""}:
        yield from rows
        return

    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        label = row.get(label_column)
        if label is not None:
            buckets[str(label)].append(row)

    if not buckets:
        return

    rng = random.Random(seed)
    counts = {label: len(label_rows) for label, label_rows in buckets.items()}

    if strategy == "downsample":
        target_count = min(counts.values())
        output: list[dict[str, Any]] = []
        for label_rows in buckets.values():
            output.extend(rng.sample(label_rows, target_count))
        rng.shuffle(output)
        yield from output
        return

    if strategy == "upsample":
        target_count = max(counts.values())
        output = []
        for label_rows in buckets.values():
            output.extend(label_rows)
            for _ in range(target_count - len(label_rows)):
                output.append(dict(rng.choice(label_rows)))
        rng.shuffle(output)
        yield from output
        return

    if strategy in {"class_weight", "weights"}:
        total = sum(counts.values())
        n_classes = len(counts)
        for label_rows in buckets.values():
            label = str(label_rows[0][label_column])
            weight = total / (n_classes * counts[label])
            for row in label_rows:
                output_row = dict(row)
                output_row[class_weight_column] = weight
                yield output_row
        return

    raise ValueError(f"Unsupported balance strategy: {strategy}")


def balance_file(
    input_path: str | Path,
    output_path: str | Path,
    *,
    strategy: str,
    label_column: str = "label",
    seed: int = 42,
    class_weight_column: str = "sample_weight",
) -> int:
    return write_rows(
        output_path,
        balance_rows(
            read_all_rows(input_path),
            strategy=strategy,
            label_column=label_column,
            seed=seed,
            class_weight_column=class_weight_column,
        ),
    )
