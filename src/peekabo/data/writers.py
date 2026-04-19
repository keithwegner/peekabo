"""Dataset writers."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Iterator
from itertools import islice
from pathlib import Path
from typing import Any


def write_rows(
    path: str | Path, rows: Iterable[dict[str, Any]], *, chunk_size: int = 10_000
) -> int:
    dataset_path = Path(path)
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = dataset_path.suffix.lower()
    if suffix == ".parquet":
        return _write_parquet_rows(dataset_path, rows, chunk_size=chunk_size)
    if suffix == ".jsonl":
        return _write_jsonl_rows(dataset_path, rows)
    if suffix == ".arff":
        return _write_arff_rows(dataset_path, rows)
    return _write_csv_rows(dataset_path, rows)


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)


def _chunks(iterator: Iterator[dict[str, Any]], size: int) -> Iterator[list[dict[str, Any]]]:
    while True:
        chunk = list(islice(iterator, size))
        if not chunk:
            break
        yield chunk


def _write_csv_rows(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    iterator = iter(rows)
    first = next(iterator, None)
    if first is None:
        path.write_text("", encoding="utf-8")
        return 0
    fieldnames = list(first.keys())
    count = 0
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerow(first)
        count += 1
        for row in iterator:
            for key in row.keys():
                if key not in fieldnames:
                    fieldnames.append(key)
                    raise ValueError(
                        f"CSV schema changed while writing {path}; new field {key!r} appeared"
                    )
            writer.writerow(row)
            count += 1
    return count


def _write_jsonl_rows(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            count += 1
    return count


def _write_arff_rows(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    buffered = list(rows)
    if not buffered:
        path.write_text("@RELATION peekabo\n\n@DATA\n", encoding="utf-8")
        return 0
    fieldnames = list(buffered[0].keys())
    lines = ["@RELATION peekabo", ""]
    for field in fieldnames:
        values = [row.get(field) for row in buffered if row.get(field) is not None]
        if values and all(isinstance(value, (int, float, bool)) for value in values):
            kind = "NUMERIC"
        else:
            kind = "STRING"
        lines.append(f"@ATTRIBUTE {field} {kind}")
    lines.extend(["", "@DATA"])
    for row in buffered:
        values = [_arff_value(row.get(field)) for field in fieldnames]
        lines.append(",".join(values))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(buffered)


def _arff_value(value: Any) -> str:
    if value is None:
        return "?"
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (int, float)):
        return str(value)
    return "'" + str(value).replace("\\", "\\\\").replace("'", "\\'") + "'"


def _write_parquet_rows(path: Path, rows: Iterable[dict[str, Any]], *, chunk_size: int) -> int:
    try:
        import pyarrow as pa  # type: ignore
        import pyarrow.parquet as pq  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency availability
        raise RuntimeError("pyarrow is required to write Parquet datasets") from exc

    iterator = iter(rows)
    writer = None
    count = 0
    try:
        for chunk in _chunks(iterator, chunk_size):
            table = pa.Table.from_pylist(chunk)
            if writer is None:
                writer = pq.ParquetWriter(path, table.schema)
            writer.write_table(table)
            count += len(chunk)
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        empty = pa.Table.from_pylist([])
        pq.write_table(empty, path)
    return count
