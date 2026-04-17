from pathlib import Path

from peekabo.data.readers import read_all_rows
from peekabo.data.sampling import balance_file, random_sample_file
from peekabo.data.splits import chronological_split_file, holdout_split_file
from peekabo.data.writers import write_rows


def rows():
    return [
        {"timestamp": 1, "label": "target", "value": 1},
        {"timestamp": 2, "label": "target", "value": 2},
        {"timestamp": 3, "label": "other", "value": 3},
        {"timestamp": 4, "label": "other", "value": 4},
        {"timestamp": 5, "label": "other", "value": 5},
    ]


def test_random_sample_is_deterministic(tmp_path: Path):
    source = tmp_path / "source.csv"
    out1 = tmp_path / "out1.csv"
    out2 = tmp_path / "out2.csv"
    write_rows(source, rows())
    random_sample_file(source, out1, percentage=50, seed=123)
    random_sample_file(source, out2, percentage=50, seed=123)
    assert read_all_rows(out1) == read_all_rows(out2)


def test_downsample_balance(tmp_path: Path):
    source = tmp_path / "source.csv"
    output = tmp_path / "balanced.csv"
    write_rows(source, rows())
    balance_file(source, output, strategy="downsample", seed=1)
    balanced = read_all_rows(output)
    assert sum(1 for row in balanced if row["label"] == "target") == 2
    assert sum(1 for row in balanced if row["label"] == "other") == 2


def test_chronological_and_holdout_splits(tmp_path: Path):
    source = tmp_path / "source.csv"
    train = tmp_path / "train.csv"
    test = tmp_path / "test.csv"
    write_rows(source, rows())
    chronological_split_file(source, train, test, train_fraction=0.6)
    assert [row["timestamp"] for row in read_all_rows(train)] == [1, 2, 3]
    assert [row["timestamp"] for row in read_all_rows(test)] == [4, 5]

    holdout_split_file(source, train, test, train_fraction=0.6, seed=99)
    assert len(read_all_rows(train)) == 3
    assert len(read_all_rows(test)) == 2

