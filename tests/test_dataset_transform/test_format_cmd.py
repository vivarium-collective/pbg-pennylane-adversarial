"""Tests for the dataset transform pipeline (reader + transform + writer)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import polars as pl
import pytest

from pbg_pennylane_adversarial.dataset_transform.reader import (
    read_dataset,
    supported_extensions,
)
from pbg_pennylane_adversarial.dataset_transform.transform import (
    resolve_target_column,
    validate_classification_target,
    encode_labels,
    transform,
)
from pbg_pennylane_adversarial.dataset_transform.writer import write

# ── helpers ──────────────────────────────────────────────────────────────────


def _csv_file(df: pl.DataFrame, name: str = "data.csv") -> Path:
    p = Path(tempfile.mkdtemp()) / name
    df.write_csv(p)
    return p


def _parquet_file(df: pl.DataFrame, name: str = "data.parquet") -> Path:
    p = Path(tempfile.mkdtemp()) / name
    df.write_parquet(p)
    return p


# ── resolve_target_column ────────────────────────────────────────────────────


class TestResolveTargetColumn:
    def test_explicit(self):
        df = pl.DataFrame({"a": [1], "b": [2], "c": [3]})
        assert resolve_target_column(df, "b") == "b"

    def test_name_target(self):
        df = pl.DataFrame({"a": [1], "target": [0], "b": [2]})
        assert resolve_target_column(df, None) == "target"

    def test_no_target_name_falls_back_to_last(self):
        df = pl.DataFrame({"a": [1], "b": [0]})
        assert resolve_target_column(df, None) == "b"

    def test_explicit_overrides_target_name(self):
        df = pl.DataFrame({"a": [1], "target": [0], "b": [2]})
        assert resolve_target_column(df, "b") == "b"


# ── validate_classification_target ───────────────────────────────────────────


class TestValidateClassificationTarget:
    def test_ok_binary(self):
        s = pl.Series("y", [0, 1, 0, 1])
        validate_classification_target(s)

    def test_ok_multiclass(self):
        s = pl.Series("y", [0, 1, 2, 3])
        validate_classification_target(s)

    def test_too_few_labels(self):
        s = pl.Series("y", [0, 0, 0])
        with pytest.raises(ValueError, match="only 1 unique"):
            validate_classification_target(s)

    def test_rejects_list_column(self):
        s = pl.Series("y", [[1, 2], [3, 4]])
        with pytest.raises(ValueError, match="must be scalar"):
            validate_classification_target(s)


# ── encode_labels ────────────────────────────────────────────────────────────


class TestEncodeLabels:
    def test_string_labels(self):
        s = pl.Series("y", ["cat", "dog", "cat"])
        encoded, mapping = encode_labels(s)
        assert encoded.to_list() == [0, 1, 0]
        assert mapping == {0: "cat", 1: "dog"}

    def test_integer_contiguous(self):
        s = pl.Series("y", [0, 0, 1, 1, 2])
        encoded, mapping = encode_labels(s)
        assert encoded.to_list() == [0, 0, 1, 1, 2]
        assert mapping == {0: "0", 1: "1", 2: "2"}

    def test_integer_non_contiguous(self):
        s = pl.Series("y", [10, 20, 10, 20, 30])
        encoded, mapping = encode_labels(s)
        assert encoded.to_list() == [0, 1, 0, 1, 2]
        assert mapping == {0: "10", 1: "20", 2: "30"}


# ── full transform pipeline ──────────────────────────────────────────────────


class TestTransform:
    def test_binary_last_column(self):
        df = pl.DataFrame({
            "age": [25.0, 30.0, 35.0, 40.0],
            "income": [50_000.0, 60_000.0, 70_000.0, 80_000.0],
            "defaulted": [0, 0, 1, 1],
        })
        result = transform(df, train_ratio=0.75, seed=0)
        assert result["input_dim"] == 2
        assert result["output_dim"] == 2
        assert result["n_train"] + result["n_test"] == 4

    def test_binary_string_labels(self):
        df = pl.DataFrame({
            "f1": [1.0, 2.0, 3.0, 4.0],
            "label": ["no", "no", "yes", "yes"],
        })
        result = transform(df, target_col="label", train_ratio=0.5, seed=0)
        assert result["output_dim"] == 2
        assert set(result["label_map"].values()) == {"no", "yes"}

    def test_multiclass(self):
        df = pl.DataFrame({
            "f1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "f2": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "category": [0, 0, 1, 1, 2, 2],
        })
        result = transform(df, target_col="category", train_ratio=0.5, seed=0)
        assert result["output_dim"] == 3
        assert result["input_dim"] == 2

    def test_normalize_off(self):
        df = pl.DataFrame({
            "x": [10.0, 20.0, 30.0, 40.0],
            "y": [0, 0, 1, 1],
        })
        result = transform(df, target_col="y", normalize=False, train_ratio=0.5, seed=0)
        assert "means" not in result
        assert result["train_images"][0] == [10.0] or result["train_images"][0] == [20.0]

    def test_train_ratio(self):
        df = pl.DataFrame({
            "f": list(range(100)),
            "label": [0] * 50 + [1] * 50,
        })
        result = transform(df, target_col="label", train_ratio=0.5, seed=0)
        assert abs(result["n_train"] / result["n_test"] - 1.0) < 0.3

    def test_feature_cols_override(self):
        df = pl.DataFrame({
            "a": [1.0, 2.0, 3.0, 4.0],
            "b": [5.0, 6.0, 7.0, 8.0],
            "c": [9.0, 10.0, 11.0, 12.0],
            "label": [0, 0, 1, 1],
        })
        result = transform(df, target_col="label", feature_cols=["a", "c"],
                           train_ratio=0.5, seed=0)
        assert result["input_dim"] == 2

    def test_fewer_than_two_labels_raises(self):
        df = pl.DataFrame({"x": [1.0, 2.0], "y": [0, 0]})
        with pytest.raises(ValueError, match="only 1 unique"):
            transform(df, target_col="y")

    def test_non_numeric_feature_raises(self):
        df = pl.DataFrame({
            "name": ["alice", "bob", "carol", "dave"],
            "score": [1.0, 2.0, 3.0, 4.0],
            "label": [0, 0, 1, 1],
        })
        with pytest.raises(ValueError, match="Feature columns must be numeric"):
            transform(df, target_col="label")

    def test_nulls_without_flag_raises(self):
        df = pl.DataFrame({
            "x": [1.0, None, 3.0, 4.0],
            "y": [0, 0, 1, 1],
        })
        with pytest.raises(ValueError, match="null values"):
            transform(df, target_col="y")


# ── reader ───────────────────────────────────────────────────────────────────


class TestReader:
    def test_csv(self):
        df = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
        p = _csv_file(df)
        loaded = read_dataset(p)
        assert loaded.shape == (2, 2)

    def test_parquet(self):
        df = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
        p = _parquet_file(df)
        loaded = read_dataset(p)
        assert loaded.shape == (2, 2)

    def test_unsupported_extension_raises(self):
        p = Path("/tmp/test.xyz")
        p.write_text("dummy")
        with pytest.raises(ValueError, match="Unsupported file extension"):
            read_dataset(p)

    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            read_dataset("/tmp/nonexistent_file_12345.csv")

    def test_supported_extensions_non_empty(self):
        exts = supported_extensions()
        assert ".csv" in exts
        assert len(exts) >= 10

    def test_json_input(self):
        p = Path(tempfile.mkdtemp()) / "data.json"
        import json
        with open(p, "w") as f:
            json.dump([{"a": 1, "b": 4}, {"a": 2, "b": 5}, {"a": 3, "b": 6}], f)
        loaded = read_dataset(p)
        assert loaded.shape == (3, 2)


# ── writer ───────────────────────────────────────────────────────────────────


class TestWriter:
    DATA = {
        "train_images": [[1.0, 2.0], [3.0, 4.0]],
        "train_labels": [0, 0],
        "test_images": [[5.0, 6.0]],
        "test_labels": [1],
        "input_dim": 2,
        "output_dim": 2,
        "n_train": 2,
        "n_test": 1,
        "label_map": {"0": "cat", "1": "dog"},
        "means": {"a": 1.0},
        "stds": {"a": 0.5},
    }

    def test_h5(self):
        p = write(self.DATA, "/tmp/_test_plap", fmt="h5")
        import h5py, json
        with h5py.File(p, "r") as f:
            assert f["train_images"].shape == (2, 2)
            assert json.loads(f.attrs["label_map"]) == {"0": "cat", "1": "dog"}
        p.unlink()

    def test_json(self):
        p = write(self.DATA, "/tmp/_test_plap", fmt="json")
        with open(p) as f:
            d = json.load(f)
        assert d["input_dim"] == 2
        assert d["label_map"] == {"0": "cat", "1": "dog"}
        p.unlink()

    def test_parquet(self):
        p = write(self.DATA, "/tmp/_test_plap", fmt="parquet")
        meta = p / "metadata.json"
        assert meta.exists()
        assert (p / "train_images.parquet").exists()
        with open(meta) as f:
            d = json.load(f)
        assert d["input_dim"] == 2
        import shutil
        shutil.rmtree(p)
