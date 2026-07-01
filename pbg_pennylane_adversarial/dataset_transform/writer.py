"""Write PLAP-formatted data to HDF5, JSON, or Parquet."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl


OutputFormat = Literal["h5", "json", "parquet"]


def write(data: dict, path: str | Path, fmt: OutputFormat = "h5") -> Path:
    """Write PLAP-formatted data to disk.

    Parameters
    ----------
    data : dict
        Must contain ``train_images``, ``train_labels``, ``test_images``,
        ``test_labels``, ``input_dim``, ``output_dim``, ``n_train``,
        ``n_test``, ``label_map``, and optionally ``means`` / ``stds``.
    path : str or Path
        Output path (extension is ignored when ``fmt`` is specified).
    fmt : "h5" | "json" | "parquet"
        Output format.
    """
    p = Path(path)
    if fmt == "h5":
        return _write_h5(data, p.with_suffix(".h5"))
    elif fmt == "json":
        return _write_json(data, p.with_suffix(".json"))
    elif fmt == "parquet":
        return _write_parquet(data, p)
    else:
        raise ValueError(f"Unsupported format: {fmt}")


def _write_h5(data: dict, path: Path) -> Path:
    import h5py

    with h5py.File(path, "w") as f:
        f.create_dataset("train_images",
                         data=np.array(data["train_images"], dtype=np.float64))
        f.create_dataset("train_labels",
                         data=np.array(data["train_labels"], dtype=np.int64))
        f.create_dataset("test_images",
                         data=np.array(data["test_images"], dtype=np.float64))
        f.create_dataset("test_labels",
                         data=np.array(data["test_labels"], dtype=np.int64))
        f.attrs["input_dim"] = data["input_dim"]
        f.attrs["output_dim"] = data["output_dim"]
        f.attrs["n_train"] = data["n_train"]
        f.attrs["n_test"] = data["n_test"]
        f.attrs["label_map"] = json.dumps(data["label_map"])
        if "means" in data:
            f.attrs["means"] = json.dumps(data["means"])
            f.attrs["stds"] = json.dumps(data["stds"])
    return path


def _write_json(data: dict, path: Path) -> Path:
    serializable = {
        k: v for k, v in data.items()
        if k in ("train_images", "train_labels", "test_images", "test_labels",
                 "input_dim", "output_dim", "n_train", "n_test",
                 "label_map", "means", "stds")
    }
    with open(path, "w") as f:
        json.dump(serializable, f)
    return path


def _write_parquet(data: dict, path: Path) -> Path:
    dir_path = path.with_suffix("")
    dir_path.mkdir(parents=True, exist_ok=True)

    pl.DataFrame({
        "feature_vector": [np.array(v, dtype=np.float64) for v in data["train_images"]]
    }).write_parquet(dir_path / "train_images.parquet")

    pl.DataFrame({"label": data["train_labels"]}).write_parquet(
        dir_path / "train_labels.parquet")

    pl.DataFrame({
        "feature_vector": [np.array(v, dtype=np.float64) for v in data["test_images"]]
    }).write_parquet(dir_path / "test_images.parquet")

    pl.DataFrame({"label": data["test_labels"]}).write_parquet(
        dir_path / "test_labels.parquet")

    meta = {k: v for k, v in data.items()
            if k not in ("train_images", "train_labels", "test_images", "test_labels")}
    meta["label_map"] = data["label_map"]
    with open(dir_path / "metadata.json", "w") as f:
        json.dump(meta, f)

    return dir_path
