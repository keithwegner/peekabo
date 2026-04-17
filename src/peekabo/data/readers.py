"""Chunk-friendly dataset readers."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterator


def iter_rows(path: str | Path, *, batch_size: int = 65_536) -> Iterator[dict[str, Any]]:
    dataset_path = Path(path)
    suffix = dataset_path.suffix.lower()
    if suffix == ".parquet":
        yield from _iter_parquet_rows(dataset_path, batch_size=batch_size)
    elif suffix == ".jsonl":
        with dataset_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)
    else:
        yield from _iter_csv_rows(dataset_path)


def read_all_rows(path: str | Path) -> list[dict[str, Any]]:
    return list(iter_rows(path))


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _iter_csv_rows(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            yield {key: _coerce_csv_value(value) for key, value in row.items()}


def _coerce_csv_value(value: str | None) -> Any:
    if value in {None, ""}:
        return None
    text = str(value)
    lowered = text.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        if "." not in text:
            return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def _iter_parquet_rows(path: Path, *, batch_size: int) -> Iterator[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency availability
        raise RuntimeError("pyarrow is required to read Parquet datasets") from exc

    parquet_file = pq.ParquetFile(path)
    for batch in parquet_file.iter_batches(batch_size=batch_size):
        for row in batch.to_pylist():
            yield row

