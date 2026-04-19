"""Train/test split utilities."""

from __future__ import annotations

import random
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from peekaboo.data.readers import iter_rows, read_all_rows
from peekaboo.data.writers import write_rows


def chronological_split_file(
    input_path: str | Path,
    train_path: str | Path,
    test_path: str | Path,
    *,
    train_fraction: float,
) -> tuple[int, int]:
    total = sum(1 for _ in iter_rows(input_path))
    train_count = int(total * train_fraction)

    def train_rows() -> Iterator[dict[str, Any]]:
        for index, row in enumerate(iter_rows(input_path)):
            if index < train_count:
                yield row

    def test_rows() -> Iterator[dict[str, Any]]:
        for index, row in enumerate(iter_rows(input_path)):
            if index >= train_count:
                yield row

    written_train = write_rows(train_path, train_rows())
    written_test = write_rows(test_path, test_rows())
    return written_train, written_test


def holdout_split_file(
    input_path: str | Path,
    train_path: str | Path,
    test_path: str | Path,
    *,
    train_fraction: float,
    seed: int,
) -> tuple[int, int]:
    rows = read_all_rows(input_path)
    rng = random.Random(seed)
    rng.shuffle(rows)
    train_count = int(len(rows) * train_fraction)
    written_train = write_rows(train_path, rows[:train_count])
    written_test = write_rows(test_path, rows[train_count:])
    return written_train, written_test


def split_file(
    input_path: str | Path,
    train_path: str | Path,
    test_path: str | Path,
    *,
    train_fraction: float,
    chronological: bool,
    seed: int,
) -> tuple[int, int]:
    if chronological:
        return chronological_split_file(
            input_path,
            train_path,
            test_path,
            train_fraction=train_fraction,
        )
    return holdout_split_file(
        input_path,
        train_path,
        test_path,
        train_fraction=train_fraction,
        seed=seed,
    )
